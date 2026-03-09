# Local Reward Model Training - Quick Start Guide

This standalone script demonstrates the complete RewardModelTrainer workflow from OpenRLHF.

## What's Included

The script `local_reward_model_training.py` contains:

1. **Loss Functions** (PairWiseLoss & LogExpLoss)
2. **Reward Model** (Transformer + value head)
3. **Synthetic Dataset** (preference pairs)
4. **Data Collator** (tokenization & batching)
5. **Trainer** (complete training loop with evaluation)
6. **Concatenated Forward Pass** (the key optimization from OpenRLHF)

## Installation

```bash
# Install required packages
pip install torch transformers tqdm
```

## Running the Script

```bash
# Run with default settings (DistilBERT, 3 epochs, synthetic data)
python local_reward_model_training.py
```

The script will:
- ✓ Automatically detect your device (Mac MPS / CUDA / CPU)
- ✓ Download a small model (DistilBERT ~250MB)
- ✓ Generate synthetic preference data
- ✓ Train for 3 epochs with progress bars
- ✓ Evaluate and show metrics
- ✓ Test the trained model on sample inputs

## Expected Output

```
================================================
Reward Model Training Demo - Complete Flow
================================================

Step 1: Loading tokenizer...
✓ Loaded tokenizer: distilbert-base-uncased

Step 2: Creating synthetic preference datasets...
✓ Train dataset: 200 samples
✓ Eval dataset: 40 samples

Step 3: Creating dataloaders...
✓ Train dataloader: 50 batches
✓ Eval dataloader: 10 batches

Step 4: Initializing reward model...
✓ Model initialized: distilbert-base-uncased
  - Parameters: 66,362,369
  - Trainable: 66,362,369

Step 5: Setting up optimizer and scheduler...
✓ Optimizer: AdamW (lr=1e-05)
✓ Scheduler: CosineAnnealingLR (total_steps=150)

Step 6: Creating trainer...
✓ Trainer created

Step 7: Starting training...

============================================================
Starting Reward Model Training
============================================================
Using PairWiseLoss (log-sigmoid)

[Training with progress bars and metrics...]

Epoch 1/3: 100%|████████| 50/50 [00:XX<00:00]
  loss: 0.6234, acc: 0.6250, lr: 9.99e-06
  chosen_r: 0.123, reject_r: -0.234

============================================================
Evaluating at step 50...
============================================================

Evaluation Results:
  Loss: 0.5891
  Accuracy: 0.6875
  Reward Mean: -0.0543
  Reward Std: 0.4321
  (Saved to model.config for PPO normalization)

...
```

## Key Features Demonstrated

### 1. Concatenated Forward Pass (Performance Optimization)

```python
# Instead of 2 forward passes:
chosen_rewards = model(chosen_ids, chosen_mask)    # Pass 1
rejected_rewards = model(rejected_ids, rejected_mask)  # Pass 2

# Do 1 forward pass:
concatenated = torch.cat([chosen_ids, rejected_ids], dim=0)
all_rewards = model(concatenated)
chosen_rewards = all_rewards[:batch_size]      # Split first half
rejected_rewards = all_rewards[batch_size:]    # Split second half
```

**Result**: 2x faster, especially important for distributed training!

### 2. Loss Functions

**PairWiseLoss (Sigmoid)**:
```python
loss = -log(sigmoid(chosen_reward - reject_reward))
```
- Minimized when `chosen_reward > reject_reward`
- Smooth gradients

**LogExpLoss**:
```python
loss = log(1 + exp(reject_reward - chosen_reward))
```
- Alternative formulation with different gradient characteristics

### 3. Reward Normalization

After each evaluation, the script computes and saves:
```python
model.config.mean = reward_mean
model.config.std = reward_std
```

These statistics are **critical for PPO training** to normalize reward signals.

### 4. Training Metrics

- **Loss**: How well the model distinguishes preferences
- **Accuracy**: % of time `chosen_reward > reject_reward`
- **Mean Rewards**: Average scores for chosen/rejected
- **Learning Rate**: Current LR from scheduler

## Customization

Edit the configuration in `main()`:

```python
# Use a different model
MODEL_NAME = "bert-base-uncased"  # or "gpt2", "microsoft/deberta-v3-small"

# Adjust training
BATCH_SIZE = 8
NUM_TRAIN_SAMPLES = 500
MAX_EPOCHS = 5
LEARNING_RATE = 2e-5

# Change loss function
trainer = SimpleRewardModelTrainer(
    ...
    loss_type='logexp',  # 'sigmoid' or 'logexp'
)
```

## What Happens Next?

After training, the reward model can be used in PPO:

1. **Actor (Policy Model)** generates responses
2. **Reward Model** (this!) scores the responses
3. **Critic Model** estimates value functions
4. **PPO Algorithm** updates the actor to maximize rewards

## Differences from Full OpenRLHF

This simplified version removes:
- ❌ Distributed training (DeepSpeed, FSDP, Ray)
- ❌ Ring attention for long sequences
- ❌ Mixtral MoE auxiliary loss
- ❌ Advanced checkpointing
- ❌ W&B/TensorBoard logging

But keeps the **core concepts**:
- ✅ Concatenated forward passes
- ✅ Loss functions (PairWiseLoss, LogExpLoss)
- ✅ Training loop structure
- ✅ Evaluation and normalization
- ✅ Accuracy metrics

## Time Estimate

On a Mac M1/M2/M3:
- Download model: ~30 seconds
- Training (3 epochs, 200 samples): ~2-3 minutes
- Total: ~3-4 minutes

On CPU:
- Training: ~5-10 minutes

## Troubleshooting

**Issue**: Model download fails
**Solution**: Check internet connection, or pre-download model

**Issue**: MPS not available
**Solution**: Script will automatically fall back to CPU

**Issue**: Out of memory
**Solution**: Reduce `BATCH_SIZE` or `MAX_LENGTH` in the config

## Next Steps

1. Try with real preference datasets (Anthropic HH-RLHF, etc.)
2. Experiment with different base models
3. Compare PairWiseLoss vs LogExpLoss
4. Use the trained model in a PPO pipeline

## Files Generated

- `local_reward_model_training.py` - Complete training script
- `LOCAL_RM_TRAINING_README.md` - This guide
