"""
Simplified PPO Training for Local Testing (Mac Compatible)

This script implements a minimal PPO training flow without Ray and vLLM dependencies.
It's designed to help understand the core PPO workflow on a local machine.

Key simplifications:
- No distributed training (Ray removed)
- Direct model generation (vLLM removed)
- Single process training
- Minimal dataset (for quick testing)

=== PPO Algorithm Overview ===

PPO (Proximal Policy Optimization) optimizes the following objectives:

1. Policy Objective (Actor):
   L^CLIP(θ) = E_t[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]
   where:
   - r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  [probability ratio]
   - A_t is the advantage estimate
   - ε is the clip parameter (typically 0.2)

2. Value Objective (Critic):
   L^VF(φ) = E_t[(V_φ(s_t) - G_t)^2]
   where G_t are the returns (targets)

3. Advantage Estimation (GAE):
   A^GAE_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
   where:
   - δ_t = r_t + γV(s_{t+1}) - V(s_t)  [TD error]
   - γ is discount factor (0.99)
   - λ is GAE parameter (0.95)

4. KL-Penalized Reward:
   r = r_RM - β·D_KL(π_θ || π_θ_old)
   where:
   - r_RM is reward model score
   - β controls KL penalty strength
   - D_KL prevents policy from changing too quickly

Training Loop:
1. Rollout: Sample trajectories using current policy π_θ
2. Evaluate: Compute rewards, values, and advantages
3. Update: Optimize actor and critic for multiple epochs on collected data
4. Repeat
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


# ============================================================================
# Experience Data Structure
# ============================================================================

@dataclass
class Experience:
    """PPO Experience batch containing all necessary tensors for training."""
    sequences: torch.Tensor  # (B, S) - generated sequences
    attention_mask: torch.Tensor  # (B, S) - attention mask
    action_mask: torch.Tensor  # (B, S) - mask for actions (generated tokens)
    action_log_probs: torch.Tensor  # (B, S) - log probs from actor during rollout
    old_action_log_probs: torch.Tensor  # (B, S) - stored for PPO ratio
    values: torch.Tensor  # (B, S) - value estimates from critic
    rewards: torch.Tensor  # (B, S) - rewards (only at last token typically)
    advantages: torch.Tensor  # (B, S) - computed advantages
    returns: torch.Tensor  # (B, S) - computed returns for critic training
    
    def to(self, device):
        """Move all tensors to device."""
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, torch.Tensor):
                setattr(self, field, value.to(device))
        return self


# ============================================================================
# Simple Dataset
# ============================================================================

class SimplePromptDataset(Dataset):
    """Minimal prompt dataset for testing."""
    
    def __init__(self, prompts: List[str]):
        self.prompts = prompts
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx]


# ============================================================================
# Model Wrappers
# ============================================================================

class Actor(torch.nn.Module):
    """Actor model wrapper for policy."""
    
    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        
    def forward(self, input_ids, attention_mask):
        """Forward pass returning logits."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
    @torch.no_grad()
    def generate(self, input_ids, attention_mask, max_new_tokens: int = 20, temperature: float = 1.0):
        """Generate sequences using sampling."""
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Track generated sequences
        generated = input_ids.clone()
        gen_attention_mask = attention_mask.clone()
        
        for _ in range(max_new_tokens):
            # Get logits for next token
            logits = self.forward(generated, gen_attention_mask)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            gen_attention_mask = torch.cat([gen_attention_mask, torch.ones_like(next_token)], dim=1)
            
            # Stop if all sequences generated EOS (simplified - just continue for now)
        
        return generated, gen_attention_mask
    
    def get_log_probs(self, sequences, attention_mask, action_mask):
        """Compute log probabilities for given sequences.
        
        For each position i, computes: log π_θ(a_{i+1} | s_{0:i})
        where a_{i+1} is the actual token at position i+1, and s_{0:i} is the sequence up to i.
        
        This is used to evaluate how likely the model was to generate the actual sequence,
        which is essential for computing the PPO probability ratio: r_t = π_θ / π_θ_old
        """
        # Get logits: shape (B, S, V) where V is vocab size
        logits = self.forward(sequences, attention_mask)
        
        # Convert to log probabilities: log π_θ(·|s_t) for all tokens in vocab
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Extract log prob of actual next token at each position
        # log_probs[:, i] predicts token at position i+1, so we gather sequences[:, i+1]
        token_log_probs = torch.gather(log_probs[:, :-1], dim=-1, 
                                       index=sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
        
        # Pad to match sequence length (first token has no prediction)
        token_log_probs = F.pad(token_log_probs, (1, 0), value=0.0)
        
        return token_log_probs


class Critic(torch.nn.Module):
    """Critic model for value function."""
    
    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # Add value head
        hidden_size = self.model.config.hidden_size
        self.value_head = torch.nn.Linear(hidden_size, 1)
        self.model.to(device)
        self.value_head.to(device)
        
    def forward(self, input_ids, attention_mask):
        """Forward pass returning value estimates."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, 
                            output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Last layer
        values = self.value_head(hidden_states).squeeze(-1)
        return values


class RewardModel(torch.nn.Module):
    """Simple reward model."""
    
    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        hidden_size = self.model.config.hidden_size
        self.reward_head = torch.nn.Linear(hidden_size, 1)
        self.model.to(device)
        self.reward_head.to(device)
        
    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        """Forward pass returning reward scores."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        # Get reward from last token
        last_token_idx = attention_mask.sum(dim=1) - 1
        last_hidden = hidden_states[range(hidden_states.shape[0]), last_token_idx]
        rewards = self.reward_head(last_hidden).squeeze(-1)
        return rewards


# ============================================================================
# PPO Loss Functions
# ============================================================================

class PPOPolicyLoss(torch.nn.Module):
    """PPO clipped surrogate objective.
    
    Implements: L^CLIP(θ) = E_t[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]
    where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) is the probability ratio
    """
    
    def __init__(self, clip_eps: float = 0.2):
        super().__init__()
        self.clip_eps = clip_eps  # ε in the formula above
    
    def forward(self, log_probs, old_log_probs, advantages, action_mask):
        """Compute PPO policy loss.
        
        Args:
            log_probs: log π_θ(a_t|s_t) - current policy
            old_log_probs: log π_θ_old(a_t|s_t) - policy at rollout time
            advantages: A_t - advantage estimates
            action_mask: mask for valid actions
        """
        # Compute probability ratio: r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
        # Using log space: r_t = exp(log π_θ - log π_θ_old)
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Clipped surrogate objective: min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)
        surr1 = ratio * advantages                    # r_t(θ)·A_t
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages  # clip(r_t)·A_t
        
        # Take minimum and negate for loss (we maximize objective, minimize loss)
        policy_loss = -torch.min(surr1, surr2)
        policy_loss = (policy_loss * action_mask).sum() / action_mask.sum()
        
        return policy_loss


class ValueLoss(torch.nn.Module):
    """MSE loss for critic with optional value clipping.
    
    Implements: L^VF(φ) = E_t[max((V_φ(s_t) - G_t)^2, (V_clipped - G_t)^2)]
    where G_t are the returns and V_clipped prevents large value updates
    """
    
    def __init__(self, clip_eps: float = 0.2):
        super().__init__()
        self.clip_eps = clip_eps
    
    def forward(self, values, old_values, returns, action_mask):
        """Compute value loss with optional clipping.
        
        Args:
            values: V_φ(s_t) - current value estimates
            old_values: V_φ_old(s_t) - value estimates at rollout time
            returns: G_t - target returns for training
            action_mask: mask for valid actions
        """
        # Clipped value loss to prevent large updates
        # V_clipped = V_old + clip(V - V_old, -ε, ε)
        values_clipped = old_values + torch.clamp(values - old_values, 
                                                   -self.clip_eps, self.clip_eps)
        
        # L^VF = max((V - G_t)^2, (V_clipped - G_t)^2)
        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        vf_loss = torch.max(vf_loss1, vf_loss2)
        
        vf_loss = (vf_loss * action_mask).sum() / action_mask.sum()
        return vf_loss


# ============================================================================
# Experience Maker
# ============================================================================

class ExperienceMaker:
    """Creates PPO experiences from prompts."""
    
    def __init__(self, actor, critic, reward_model, ref_model, tokenizer, kl_coef: float = 0.1):
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.kl_coef = kl_coef
    
    @torch.no_grad()
    def make_experience(self, prompts: List[str], device: str = "cpu", 
                       max_new_tokens: int = 20) -> Experience:
        """Generate experiences from prompts for PPO training.
        
        This implements the rollout phase where we:
        1. Sample actions from policy: a_t ~ π_θ(·|s_t)
        2. Compute log probabilities: log π_θ(a_t|s_t) and log π_θ_old(a_t|s_t)
        3. Estimate values: V_φ(s_t)
        4. Compute KL-penalized rewards: r = r_RM - β·D_KL(π_θ || π_θ_old)
        5. Calculate advantages using GAE: A^GAE_t
        """
        # Tokenize prompts
        prompt_tensors = self.tokenizer(prompts, return_tensors="pt", 
                                       padding=True, truncation=True)
        input_ids = prompt_tensors["input_ids"].to(device)
        attention_mask = prompt_tensors["attention_mask"].to(device)
        
        prompt_length = input_ids.shape[1]
        
        # === Step 1: Sample actions from current policy ===
        # Generate: a_t ~ π_θ(·|s_t) for each timestep
        sequences, seq_attention_mask = self.actor.generate(
            input_ids, attention_mask, max_new_tokens=max_new_tokens
        )
        
        # Create action mask (only for generated tokens)
        action_mask = torch.zeros_like(sequences, dtype=torch.bool)
        action_mask[:, prompt_length:] = True
        action_mask = action_mask & (seq_attention_mask.bool())
        
        # === Step 2: Compute log π_θ(a_t|s_t) - current policy log probs ===
        action_log_probs = self.actor.get_log_probs(sequences, seq_attention_mask, action_mask)
        
        # === Step 3: Compute log π_θ_old(a_t|s_t) - reference policy log probs ===
        # Used for KL penalty to prevent policy from changing too fast
        ref_log_probs = self.ref_model.get_log_probs(sequences, seq_attention_mask, action_mask)
        
        # === Step 4: Get value estimates V_φ(s_t) from critic ===
        values = self.critic(sequences, seq_attention_mask)
        
        # === Step 5: Get reward scores r_RM from reward model ===
        reward_scores = self.reward_model(sequences, seq_attention_mask)
        
        # === Step 6: Compute KL-penalized rewards ===
        # D_KL(π_θ || π_θ_old) = log π_θ(a|s) - log π_θ_old(a|s)
        kl_divergence = action_log_probs - ref_log_probs
        rewards = torch.zeros_like(sequences, dtype=torch.float)
        
        # Place reward at last generated token
        # Final reward: r = r_RM - β·D_KL where β = kl_coef
        for i in range(sequences.shape[0]):
            last_action_idx = action_mask[i].nonzero(as_tuple=True)[0][-1]
            # Sum KL divergence over all generated tokens
            kl_penalty = (kl_divergence[i] * action_mask[i]).sum()
            # Apply KL-penalized reward: r = r_RM - β·D_KL
            rewards[i, last_action_idx] = reward_scores[i] - self.kl_coef * kl_penalty
        
        # === Step 7: Compute advantages using GAE ===
        # A^GAE_t = Σ(γλ)^l δ_{t+l} where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        # Returns: G_t = A_t + V(s_t)
        advantages, returns = self._compute_advantages(rewards, values, action_mask)
        
        return Experience(
            sequences=sequences,                          # Generated token sequences
            attention_mask=seq_attention_mask,            # Mask for valid tokens
            action_mask=action_mask.float(),              # Mask for generated tokens only
            action_log_probs=action_log_probs,            # log π_θ(a_t|s_t) - current policy
            old_action_log_probs=action_log_probs.clone(), # Stored for computing ratio r_t(θ) = π_θ/π_θ_old
            values=values,                                # V_φ(s_t) - critic's value estimates
            rewards=rewards,                              # r_t with KL penalty applied
            advantages=advantages,                        # A^GAE_t - advantage estimates
            returns=returns                               # G_t = A_t + V(s_t) - targets for critic
        )
    
    def _compute_advantages(self, rewards, values, action_mask, gamma: float = 0.99, 
                           lam: float = 0.95):
        """Compute GAE (Generalized Advantage Estimation) advantages.
        
        GAE Formula: A^GAE_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
        where TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
        
        GAE is a variance-reduction technique that balances bias and variance in advantage estimates.
        It uses a weighted average of n-step advantages, controlled by lambda (lam).
        
        Args:
            rewards: Sparse reward tensor (B, S) - typically only non-zero at final action
            values: Value estimates from critic V_φ(s_t) (B, S) - predictions of future returns
            action_mask: Boolean mask (B, S) - which tokens are actions (vs prompt tokens)
            gamma: Discount factor γ (0.99) - how much we value future rewards
            lam: GAE lambda λ (0.95) - trades off bias vs variance (higher = more variance, less bias)
        
        Returns:
            advantages: A^GAE_t - how much better an action was vs expected (B, S)
            returns: G_t = A_t + V(s_t) - target values for critic training (B, S)
        """
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Process each sequence in the batch
        for i in range(batch_size):
            last_gae = 0  # Accumulator for GAE calculation: A_t = δ_t + γλ·A_{t+1}
            action_indices = action_mask[i].nonzero(as_tuple=True)[0]
            
            # Work backwards through time - GAE requires reverse iteration
            # This allows us to propagate future value estimates backwards
            for t in reversed(action_indices.tolist()):
                # Bootstrap value: estimate of future returns from next state
                if t == action_indices[-1]:  # Terminal state
                    next_value = 0  # No future value at end
                else:
                    next_value = values[i, t + 1]  # V(s_{t+1}) - Use critic's next-state estimate
                
                # TD error (delta): δ_t = r_t + γV(s_{t+1}) - V(s_t)
                # This measures how much better/worse this step was than the critic expected
                delta = rewards[i, t] + gamma * next_value - values[i, t]
                
                # GAE recursion: A_t = δ_t + γλ·A_{t+1}
                # Exponentially-weighted moving average of TD errors
                # Balances between 1-step TD (λ=0, low variance, high bias) 
                # and Monte Carlo (λ=1, high variance, low bias)
                last_gae = delta + gamma * lam * last_gae
                advantages[i, t] = last_gae
                
                # Return = advantage + baseline value: G_t = A_t + V(s_t)
                # This is the target for training the critic (value network)
                returns[i, t] = advantages[i, t] + values[i, t]
        
        # Normalize advantages across the batch for stable training
        # This is a standard trick in PPO to reduce variance
        adv_mean = (advantages * action_mask).sum() / action_mask.sum()
        adv_std = torch.sqrt(((advantages - adv_mean) ** 2 * action_mask).sum() / action_mask.sum())
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)  # 1e-8 prevents division by zero
        
        return advantages, returns


# ============================================================================
# PPO Trainer
# ============================================================================

class SimplePPOTrainer:
    """Simplified PPO Trainer for local testing."""
    
    def __init__(self, actor, critic, reward_model, tokenizer, 
                 lr: float = 1e-5, clip_eps: float = 0.2, kl_coef: float = 0.1,
                 device: str = "cpu"):
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.ref_model = Actor(device=device)  # Frozen copy of initial actor
        self.ref_model.load_state_dict(actor.state_dict())
        self.ref_model.eval()
        
        self.tokenizer = tokenizer
        self.device = device
        
        # Optimizers
        self.actor_optim = AdamW(actor.parameters(), lr=lr)
        self.critic_optim = AdamW(critic.parameters(), lr=lr)
        
        # Loss functions
        self.policy_loss_fn = PPOPolicyLoss(clip_eps=clip_eps)
        self.value_loss_fn = ValueLoss(clip_eps=clip_eps)
        
        # Experience maker
        self.experience_maker = ExperienceMaker(
            actor, critic, reward_model, self.ref_model, tokenizer, kl_coef
        )
        
        self.global_step = 0
    
    def train_step(self, prompts: List[str], max_new_tokens: int = 20, 
                   num_epochs: int = 2) -> dict:
        """Single PPO training step."""
        # 1. Generate experiences
        print(f"[Step {self.global_step}] Generating experiences...")
        experience = self.experience_maker.make_experience(
            prompts, device=self.device, max_new_tokens=max_new_tokens
        )
        
        # Store old values for clipping
        old_values = experience.values.clone()
        
        stats = {}
        
        # 2. PPO updates for multiple epochs
        for epoch in range(num_epochs):
            # Move experience to device
            experience = experience.to(self.device)
            
            # === Critic Update ===
            self.critic.train()
            self.critic_optim.zero_grad()
            
            # Get current values
            current_values = self.critic(experience.sequences, experience.attention_mask)
            
            # Compute value loss
            value_loss = self.value_loss_fn(
                current_values, old_values, experience.returns, experience.action_mask
            )
            
            value_loss.backward()
            self.critic_optim.step()
            
            # === Actor Update ===
            self.actor.train()
            self.actor_optim.zero_grad()
            
            # Get current log probs
            current_log_probs = self.actor.get_log_probs(
                experience.sequences, experience.attention_mask, experience.action_mask
            )
            
            # Compute policy loss
            policy_loss = self.policy_loss_fn(
                current_log_probs, experience.old_action_log_probs, 
                experience.advantages, experience.action_mask
            )
            
            policy_loss.backward()
            self.actor_optim.step()
            
            if epoch == num_epochs - 1:  # Last epoch stats
                stats = {
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "mean_reward": (experience.rewards * experience.action_mask).sum().item() / experience.action_mask.sum().item(),
                    "mean_advantage": experience.advantages.mean().item()
                }
        
        self.global_step += 1
        return stats
    
    def fit(self, prompts: List[str], num_episodes: int = 3, batch_size: int = 2,
           max_new_tokens: int = 20):
        """Main training loop."""
        print(f"Starting PPO Training...")
        print(f"Total prompts: {len(prompts)}, Batch size: {batch_size}, Episodes: {num_episodes}")
        print("=" * 80)
        
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print("-" * 80)
            
            # Process in batches
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                
                # Training step
                stats = self.train_step(batch_prompts, max_new_tokens=max_new_tokens)
                
                # Log stats
                print(f"Batch {i//batch_size + 1}: "
                      f"policy_loss={stats['policy_loss']:.4f}, "
                      f"value_loss={stats['value_loss']:.4f}, "
                      f"mean_reward={stats['mean_reward']:.4f}")
                
                # Generate a sample to show progress
                if i == 0:
                    self._generate_sample(batch_prompts[0])
        
        print("\n" + "=" * 80)
        print("Training complete!")
    
    @torch.no_grad()
    def _generate_sample(self, prompt: str):
        """Generate and print a sample response."""
        self.actor.eval()
        prompt_tensor = self.tokenizer(prompt, return_tensors="pt")
        input_ids = prompt_tensor["input_ids"].to(self.device)
        attention_mask = prompt_tensor["attention_mask"].to(self.device)
        
        generated, _ = self.actor.generate(input_ids, attention_mask, max_new_tokens=20)
        decoded = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        print(f"  Sample: {decoded[:100]}...")


# ============================================================================
# Main Testing Function
# ============================================================================

def main():
    """Main function to test PPO training."""
    print("Initializing Local PPO Training Test")
    print("=" * 80)
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Small model for testing
    model_name = "gpt2"  # Use gpt2 for quick testing
    print(f"Model: {model_name}")
    
    # Initialize tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize models
    print("Loading models...")
    actor = Actor(model_name, device=device)
    critic = Critic(model_name, device=device)
    reward_model = RewardModel(model_name, device=device)
    
    # Sample prompts for testing
    prompts = [
        "Once upon a time",
        "The meaning of life is",
        "In the future, AI will",
        "The best way to learn",
    ]
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = SimplePPOTrainer(
        actor=actor,
        critic=critic,
        reward_model=reward_model,
        tokenizer=tokenizer,
        lr=1e-5,
        clip_eps=0.2,
        kl_coef=0.1,
        device=device
    )
    
    # Train
    print("\n" + "=" * 80)
    trainer.fit(
        prompts=prompts,
        num_episodes=2,
        batch_size=2,
        max_new_tokens=20
    )
    
    print("\nTest complete!")


if __name__ == "__main__":
    main()
