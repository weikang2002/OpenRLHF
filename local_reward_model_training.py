"""
Standalone Reward Model Training Script
Demonstrates the complete RewardModelTrainer flow for local testing on Mac
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
from typing import Optional


# ============================================================================
# Loss Functions (from openrlhf/models/loss.py)
# ============================================================================

class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model using log-sigmoid
    Loss encourages: chosen_reward > reject_reward
    """
    def forward(
        self, 
        chosen_reward: torch.Tensor, 
        reject_reward: torch.Tensor, 
        margin: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if margin is not None:
            # With margin: loss = -log(sigmoid(chosen - rejected - margin))
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            # Standard: loss = -log(sigmoid(chosen - rejected))
            # This minimizes when chosen_reward > reject_reward
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()


class LogExpLoss(nn.Module):
    """
    LogExp Loss for Reward Model
    Loss = log(1 + exp(reject - chosen))
    Minimizes when chosen_reward > reject_reward
    """
    def forward(
        self, 
        chosen_reward: torch.Tensor, 
        reject_reward: torch.Tensor, 
        margin: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward))
        return loss.mean()


# ============================================================================
# Reward Model (Simplified version)
# ============================================================================

class RewardModel(nn.Module):
    """
    Simple Reward Model: Transformer backbone + value head
    """
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.config = self.transformer.config
        
        # Value head: projects hidden states to scalar reward
        hidden_size = self.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Store normalization stats (used during PPO)
        self.config.mean = 0.0
        self.config.std = 1.0
    
    def forward(self, input_ids, attention_mask=None):
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get last hidden state
        last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # Extract the last token's hidden state (where the reward is computed)
        # Find the last non-padding token for each sequence
        if attention_mask is not None:
            # Get the last valid token position for each sequence
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            last_token_hidden = last_hidden[torch.arange(batch_size), sequence_lengths]
        else:
            last_token_hidden = last_hidden[:, -1, :]  # Just use last token
        
        # Compute reward (scalar value)
        rewards = self.value_head(last_token_hidden)  # [batch, 1]
        
        return rewards


# ============================================================================
# Dataset (Synthetic preference pairs)
# ============================================================================

class PreferenceDataset(Dataset):
    """
    Synthetic dataset of preference pairs for demonstration
    In real usage, this would load from JSONL files with human preferences
    """
    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples
        
        # Generate realistic preference pairs (simulating human feedback data)
        self.data = []
        
        # Realistic prompt-response pairs with clear quality differences
        preference_pairs = [
            {
                'prompt': "Explain what machine learning is to a beginner.",
                'chosen': "Machine learning is a branch of artificial intelligence where computers learn from data to make predictions or decisions without being explicitly programmed. Think of it like teaching a child to recognize animals - instead of listing every rule, you show them examples until they learn the patterns. The computer analyzes lots of examples and finds patterns that help it make accurate predictions on new, unseen data.",
                'rejected': "Machine learning is when computers learn stuff. It's like AI and uses algorithms. It's pretty complicated and involves data."
            },
            {
                'prompt': "What are the health benefits of drinking water?",
                'chosen': "Drinking adequate water provides numerous health benefits: 1) Maintains body temperature and lubricates joints, 2) Helps flush out toxins through kidneys, 3) Improves skin hydration and appearance, 4) Aids digestion and prevents constipation, 5) Boosts energy levels and cognitive function, 6) Supports cardiovascular health. Adults should aim for about 8 glasses (2 liters) daily, though needs vary based on activity level, climate, and individual factors.",
                'rejected': "Water is good for you. You should drink it every day. It helps your body work better and keeps you healthy. Drink lots of water."
            },
            {
                'prompt': "How do I fix a leaking faucet?",
                'chosen': "To fix a leaking faucet, follow these steps: 1) Turn off the water supply under the sink, 2) Remove the faucet handle by unscrewing it, 3) Use a wrench to remove the packing nut, 4) Take out the old washer or O-ring - this is usually the culprit, 5) Replace it with a new washer of the same size (available at hardware stores), 6) Reassemble everything in reverse order, 7) Turn water back on and test. If the leak persists, the valve seat might be corroded and need professional attention.",
                'rejected': "Just tighten it with a wrench. If that doesn't work, you probably need to call a plumber or buy a new faucet."
            },
            {
                'prompt': "Write a professional email declining a job offer.",
                'chosen': "Subject: Job Offer - [Position Name]\n\nDear [Hiring Manager's Name],\n\nThank you for offering me the [Position Name] role at [Company]. I sincerely appreciate the time you and your team invested in the interview process and the confidence you've shown in my abilities.\n\nAfter careful consideration, I have decided to decline the offer as I've accepted a position that better aligns with my current career goals. This was not an easy decision, as I was impressed by [specific positive aspect about company].\n\nI hope we might have the opportunity to work together in the future. I wish you and the team continued success.\n\nBest regards,\n[Your Name]",
                'rejected': "Hi, thanks for the offer but I'm going to pass. I got a better opportunity elsewhere. Good luck finding someone.\n\nThanks"
            },
            {
                'prompt': "What causes earthquakes?",
                'chosen': "Earthquakes are caused by the sudden release of energy in Earth's crust, creating seismic waves. The Earth's outer layer consists of tectonic plates that constantly move, though very slowly. When these plates interact at their boundaries - whether colliding, sliding past each other, or pulling apart - stress builds up over time. When this stress exceeds the strength of rocks, they suddenly break or shift, releasing energy as an earthquake. The point where this rupture begins is called the focus (or hypocenter), and the point directly above it on the surface is the epicenter. Most earthquakes occur along plate boundaries, particularly around the Pacific Ring of Fire.",
                'rejected': "Earthquakes happen when the ground shakes. It's because of movements deep in the Earth. They can be dangerous and cause damage."
            },
            {
                'prompt': "What's the difference between Python lists and tuples?",
                'chosen': "Python lists and tuples are both sequence data types, but have key differences:\n\n1) Mutability: Lists are mutable (can be modified after creation) using methods like append(), remove(), or direct assignment. Tuples are immutable - once created, their contents cannot be changed.\n\n2) Syntax: Lists use square brackets [1, 2, 3], tuples use parentheses (1, 2, 3).\n\n3) Performance: Tuples are generally faster and use less memory due to immutability.\n\n4) Use cases: Use lists for collections that need modification; use tuples for fixed data, dictionary keys, or to ensure data integrity.\n\nExample: my_list = [1, 2, 3]; my_list[0] = 5  # Valid\nmy_tuple = (1, 2, 3); my_tuple[0] = 5  # Raises TypeError",
                'rejected': "Lists can be changed but tuples can't. Lists use brackets and tuples use parentheses. That's basically it."
            },
            {
                'prompt': "How can I improve my sleep quality?",
                'chosen': "To improve sleep quality, try these evidence-based strategies:\n\n1) Maintain a consistent sleep schedule, even on weekends\n2) Create a relaxing bedtime routine (reading, gentle stretching, meditation)\n3) Keep your bedroom cool (65-68°F), dark, and quiet\n4) Limit screen time 1-2 hours before bed due to blue light\n5) Avoid caffeine after 2 PM and heavy meals close to bedtime\n6) Exercise regularly, but not within 3 hours of sleep\n7) Manage stress through journaling or relaxation techniques\n8) Consider your mattress quality and replace if needed\n\nIf problems persist beyond 2-3 weeks, consult a healthcare provider to rule out sleep disorders.",
                'rejected': "Go to bed earlier and avoid your phone. Make sure your room is dark. That should help you sleep better."
            },
            {
                'prompt': "Explain the concept of compound interest.",
                'chosen': "Compound interest is interest calculated on both the initial principal and the accumulated interest from previous periods - essentially 'interest on interest.' This creates exponential growth over time.\n\nThe formula is: A = P(1 + r/n)^(nt)\nWhere: A = final amount, P = principal, r = annual rate, n = compounds per year, t = years\n\nExample: $1,000 at 5% annual interest compounded yearly:\nYear 1: $1,050, Year 2: $1,102.50, Year 10: $1,628.89\n\nThe key insight: Your money grows faster over time because you're earning returns on your previous returns. Starting early makes a huge difference - this is why financial advisors emphasize beginning retirement savings young. Einstein allegedly called it 'the eighth wonder of the world.'",
                'rejected': "Compound interest is when you earn interest on your interest. So if you have money in a savings account, you make more money over time. It's better than regular interest."
            },
            {
                'prompt': "What should I do if my computer is running slow?",
                'chosen': "Here's a systematic approach to speed up a slow computer:\n\n1) Check Task Manager (Ctrl+Shift+Esc) to identify resource-heavy programs\n2) Restart your computer - clears temporary memory issues\n3) Remove unused programs via Control Panel > Uninstall\n4) Run disk cleanup to remove temporary files\n5) Disable startup programs that aren't necessary\n6) Check for malware using Windows Defender or reputable antivirus\n7) Free up disk space - aim for 15-20% free for optimal performance\n8) Consider hardware upgrades: adding RAM (most impactful) or switching to an SSD\n9) Update drivers and operating system\n10) If Windows, consider using built-in 'Reset this PC' feature\n\nFor Macs, try clearing cache, checking Activity Monitor, and ensuring sufficient storage.",
                'rejected': "Your computer is probably slow because it's old or has viruses. Try deleting some files and restarting it. You might need to buy a new one."
            },
            {
                'prompt': "How do vaccines work?",
                'chosen': "Vaccines work by training your immune system to recognize and fight specific pathogens without causing the actual disease.\n\nHere's the process:\n1) The vaccine contains a weakened, inactive, or partial form of a pathogen (or instructions to make a harmless piece of it, like mRNA vaccines)\n2) Your immune system recognizes these antigens as foreign invaders\n3) It produces antibodies and activates T-cells specifically designed to combat that pathogen\n4) Crucially, your immune system creates 'memory cells' that remember this pathogen\n5) If you later encounter the real pathogen, your immune system rapidly deploys these memory cells to fight it off before you get sick\n\nThis provides immunity without the risks of actual infection. Some vaccines require boosters to maintain strong immunity over time. Widespread vaccination also creates herd immunity, protecting vulnerable individuals who cannot be vaccinated.",
                'rejected': "Vaccines give you a small amount of the disease so your body learns to fight it. Then if you get exposed to the real disease, you won't get sick because your body knows how to fight it off."
            },
        ]
        
        # Cycle through realistic pairs to generate required number of samples
        for i in range(num_samples):
            pair = preference_pairs[i % len(preference_pairs)]
            
            self.data.append({
                'chosen': pair['prompt'] + " " + pair['chosen'],
                'rejected': pair['prompt'] + " " + pair['rejected'],
                'margin': 0.0  # Optional margin for confidence weighting
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================================
# Collator (Prepares batches for training)
# ============================================================================

class RewardDataCollator:
    """
    Collates preference pairs into batches with proper tokenization
    """
    def __init__(self, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch):
        chosen_texts = [item['chosen'] for item in batch]
        rejected_texts = [item['rejected'] for item in batch]
        margins = [item['margin'] for item in batch]
        
        # Tokenize chosen and rejected separately
        chosen_encodings = self.tokenizer(
            chosen_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        rejected_encodings = self.tokenizer(
            rejected_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'chosen_ids': chosen_encodings['input_ids'],
            'chosen_mask': chosen_encodings['attention_mask'],
            'rejected_ids': rejected_encodings['input_ids'],
            'rejected_mask': rejected_encodings['attention_mask'],
            'margin': margins
        }


# ============================================================================
# Reward Model Trainer (Simplified from openrlhf/trainer/rm_trainer.py)
# ============================================================================

class SimpleRewardModelTrainer:
    """
    Simplified Reward Model Trainer for local testing
    Core logic from RewardModelTrainer adapted for single-device training
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        optimizer,
        scheduler,
        device: str = 'cpu',
        loss_type: str = 'sigmoid',
        max_epochs: int = 3,
        logging_steps: int = 10,
        eval_steps: int = 50,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_epochs = max_epochs
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        
        # Loss function
        if loss_type == 'sigmoid':
            self.loss_fn = PairWiseLoss()
            print("Using PairWiseLoss (log-sigmoid)")
        else:
            self.loss_fn = LogExpLoss()
            print("Using LogExpLoss")
    
    def concatenated_forward(self, chosen_ids, chosen_mask, rejected_ids, rejected_mask):
        """
        KEY OPTIMIZATION: Concatenate chosen and rejected for single forward pass
        This is 2x faster than doing separate forward passes (especially for FSDP)
        
        From openrlhf/trainer/rm_trainer.py lines 304-316
        """
        # Concatenate inputs along batch dimension (dim=0)
        input_ids, attention_mask = self.concatenated_inputs(
            chosen_ids, chosen_mask, rejected_ids, rejected_mask
        )
        
        # Single forward pass on concatenated batch
        all_rewards = self.model(input_ids, attention_mask=attention_mask)
        
        # Split results: first half = chosen, second half = rejected
        batch_size = chosen_ids.shape[0]
        chosen_rewards = all_rewards[:batch_size]      # First half
        rejected_rewards = all_rewards[batch_size:]    # Second half
        
        return chosen_rewards, rejected_rewards
    
    def concatenated_inputs(self, chosen_ids, chosen_mask, rejected_ids, rejected_mask):
        """
        Concatenate chosen and rejected inputs into single tensor
        Pads to same length if needed
        
        From openrlhf/trainer/rm_trainer.py lines 318-350
        """
        # Pad to same length if needed
        max_length = max(chosen_ids.shape[1], rejected_ids.shape[1])
        
        def pad_to_length(tensor, length, pad_value):
            if tensor.size(1) >= length:
                return tensor
            pad_size = length - tensor.size(1)
            padding = torch.full(
                (tensor.size(0), pad_size), 
                pad_value, 
                dtype=tensor.dtype, 
                device=tensor.device
            )
            return torch.cat([padding, tensor], dim=1)  # Left pad
        
        # Pad and concatenate
        chosen_ids_padded = pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id)
        rejected_ids_padded = pad_to_length(rejected_ids, max_length, self.tokenizer.pad_token_id)
        input_ids = torch.cat([chosen_ids_padded, rejected_ids_padded], dim=0)
        
        chosen_mask_padded = pad_to_length(chosen_mask, max_length, 0)
        rejected_mask_padded = pad_to_length(rejected_mask, max_length, 0)
        attention_mask = torch.cat([chosen_mask_padded, rejected_mask_padded], dim=0)
        
        return input_ids, attention_mask
    
    def train_epoch(self, epoch: int):
        """
        Train for one epoch
        From openrlhf/trainer/rm_trainer.py fit() method
        """
        self.model.train()
        total_loss = 0
        total_acc = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.max_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            chosen_ids = batch['chosen_ids'].to(self.device)
            chosen_mask = batch['chosen_mask'].to(self.device)
            rejected_ids = batch['rejected_ids'].to(self.device)
            rejected_mask = batch['rejected_mask'].to(self.device)
            margin = torch.tensor(batch['margin']).to(self.device)
            
            # Forward pass with concatenation
            chosen_rewards, rejected_rewards = self.concatenated_forward(
                chosen_ids, chosen_mask, rejected_ids, rejected_mask
            )
            
            # Compute loss
            loss = self.loss_fn(chosen_rewards, rejected_rewards, margin=None)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Compute accuracy: how often chosen > rejected
            acc = (chosen_rewards > rejected_rewards).float().mean().item()
            
            total_loss += loss.item()
            total_acc += acc
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                'chosen_r': f'{chosen_rewards.mean().item():.3f}',
                'reject_r': f'{rejected_rewards.mean().item():.3f}'
            })
            
            # Logging
            if (step + 1) % self.logging_steps == 0:
                avg_loss = total_loss / self.logging_steps
                avg_acc = total_acc / self.logging_steps
                print(f"\nStep {step+1}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")
                total_loss = 0
                total_acc = 0
            
            # Evaluation
            if (step + 1) % self.eval_steps == 0:
                self.evaluate(step + 1)
                self.model.train()
    
    def evaluate(self, step: int = 0):
        """
        Evaluate on validation set
        From openrlhf/trainer/rm_trainer.py evaluate() method (lines 253-303)
        """
        self.model.eval()
        
        total_loss = 0
        total_acc = 0
        all_rewards = []
        
        print(f"\n{'='*60}")
        print(f"Evaluating at step {step}...")
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluation"):
                chosen_ids = batch['chosen_ids'].to(self.device)
                chosen_mask = batch['chosen_mask'].to(self.device)
                rejected_ids = batch['rejected_ids'].to(self.device)
                rejected_mask = batch['rejected_mask'].to(self.device)
                
                # Forward pass
                chosen_rewards, rejected_rewards = self.concatenated_forward(
                    chosen_ids, chosen_mask, rejected_ids, rejected_mask
                )
                
                # Compute metrics
                loss = self.loss_fn(chosen_rewards, rejected_rewards, margin=None)
                acc = (chosen_rewards > rejected_rewards).float().mean().item()
                
                total_loss += loss.item()
                total_acc += acc
                
                # Collect all rewards for statistics
                all_rewards.append(chosen_rewards)
                all_rewards.append(rejected_rewards)
        
        # Compute averages
        avg_loss = total_loss / len(self.eval_dataloader)
        avg_acc = total_acc / len(self.eval_dataloader)
        
        # Compute reward statistics
        all_rewards = torch.cat(all_rewards, dim=0)
        reward_mean = all_rewards.mean().item()
        reward_std = all_rewards.std().item()
        
        # IMPORTANT: Save normalization stats to model config
        # These are used during PPO to normalize rewards
        self.model.config.mean = reward_mean
        self.model.config.std = max(reward_std, 1e-8)  # Avoid division by zero
        
        print(f"\nEvaluation Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {avg_acc:.4f}")
        print(f"  Reward Mean: {reward_mean:.4f}")
        print(f"  Reward Std: {reward_std:.4f}")
        print(f"  (Saved to model.config for PPO normalization)")
        
        # Reward distribution
        print(f"\nReward Distribution:")
        print(f"  Min: {all_rewards.min().item():.4f}")
        print(f"  25th percentile: {torch.quantile(all_rewards, 0.25).item():.4f}")
        print(f"  Median: {torch.median(all_rewards).item():.4f}")
        print(f"  75th percentile: {torch.quantile(all_rewards, 0.75).item():.4f}")
        print(f"  Max: {all_rewards.max().item():.4f}")
        print(f"{'='*60}\n")
    
    def fit(self):
        """
        Main training loop
        """
        print("\n" + "="*60)
        print("Starting Reward Model Training")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.max_epochs}")
        print(f"Train batches per epoch: {len(self.train_dataloader)}")
        print(f"Eval batches: {len(self.eval_dataloader)}")
        print("="*60 + "\n")
        
        # Initial evaluation
        print("Initial evaluation before training:")
        self.evaluate(step=0)
        
        # Train for all epochs
        for epoch in range(self.max_epochs):
            self.train_epoch(epoch)
        
        # Final evaluation
        print("\nFinal evaluation after training:")
        self.evaluate(step='final')
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """
    Main function to run the complete reward model training flow
    """
    # Configuration
    MODEL_NAME = "distilbert-base-uncased"  # Small model for Mac
    BATCH_SIZE = 4
    NUM_TRAIN_SAMPLES = 200
    NUM_EVAL_SAMPLES = 40
    MAX_EPOCHS = 3
    LEARNING_RATE = 1e-5
    MAX_LENGTH = 128
    
    # Device selection (MPS for Mac M1/M2/M3, or CPU)
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Metal Performance Shaders) on Mac")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("Using CUDA")
    else:
        device = 'cpu'
        print("Using CPU")
    
    print(f"\n{'='*60}")
    print("Reward Model Training Demo - Complete Flow")
    print(f"{'='*60}\n")
    
    # Step 1: Initialize tokenizer
    print("Step 1: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Loaded tokenizer: {MODEL_NAME}\n")
    
    # Step 2: Create datasets
    print("Step 2: Creating synthetic preference datasets...")
    train_dataset = PreferenceDataset(num_samples=NUM_TRAIN_SAMPLES)
    eval_dataset = PreferenceDataset(num_samples=NUM_EVAL_SAMPLES)
    print(f"✓ Train dataset: {len(train_dataset)} samples")
    print(f"✓ Eval dataset: {len(eval_dataset)} samples\n")
    
    # Step 3: Create dataloaders
    print("Step 3: Creating dataloaders...")
    collator = RewardDataCollator(tokenizer, max_length=MAX_LENGTH)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collator
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        collate_fn=collator
    )
    print(f"✓ Train dataloader: {len(train_dataloader)} batches")
    print(f"✓ Eval dataloader: {len(eval_dataloader)} batches\n")
    
    # Step 4: Initialize reward model
    print("Step 4: Initializing reward model...")
    model = RewardModel(model_name=MODEL_NAME)
    print(f"✓ Model initialized: {MODEL_NAME}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    # Step 5: Setup optimizer and scheduler
    print("Step 5: Setting up optimizer and scheduler...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * MAX_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    print(f"✓ Optimizer: AdamW (lr={LEARNING_RATE})")
    print(f"✓ Scheduler: CosineAnnealingLR (total_steps={total_steps})\n")
    
    # Step 6: Create trainer
    print("Step 6: Creating trainer...")
    trainer = SimpleRewardModelTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        loss_type='sigmoid',  # or 'logexp'
        max_epochs=MAX_EPOCHS,
        logging_steps=10,
        eval_steps=50
    )
    print("✓ Trainer created\n")
    
    # Step 7: Train!
    print("Step 7: Starting training...")
    trainer.fit()
    
    # Step 8: Test the trained model
    print("\n" + "="*60)
    print("Step 8: Testing trained model on sample inputs")
    print("="*60)
    
    model.eval()
    test_samples = [
        "What is climate change? Climate change refers to long-term shifts in global temperatures and weather patterns, primarily driven by human activities like burning fossil fuels, which release greenhouse gases that trap heat in the atmosphere.",
        "What is climate change? It's when the weather gets different over time.",
        "How do solar panels work? Solar panels convert sunlight into electricity using photovoltaic cells made of semiconductor materials. When photons hit the cells, they knock electrons loose, creating an electric current that can power homes and businesses.",
        "How do solar panels work? They use the sun to make electricity somehow.",
        "Explain photosynthesis. Photosynthesis is the process by which plants convert light energy into chemical energy, using chlorophyll to capture sunlight and combine CO2 and water to produce glucose and oxygen.",
        "Explain photosynthesis. Plants make food from sunlight.",
    ]
    
    with torch.no_grad():
        for i, text in enumerate(test_samples):
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            reward = model(**inputs)
            print(f"Sample {i+1}: Reward = {reward.item():.4f}")
            print(f"  Text: {text[:80]}...")
            print()
    
    print("="*60)
    print("Complete! The trained model can now be used for PPO training.")
    print("="*60)
    
    return model, tokenizer


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run the complete training flow
    model, tokenizer = main()
