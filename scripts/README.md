# Distributed Training Scripts

## Quick Start

### Single Experiment

```bash
# Train a small model with Muon optimizer
python distributed_train.py \
    --model-size small \
    --optimizer muon \
    --transformer-type normalized \
    --batch-size 8 \
    --learning-rate 15e-4 \
    --max-epochs 1

# Train a medium model with Muon8bit optimizer
python distributed_train.py \
    --model-size medium \
    --optimizer muon_8bit \
    --transformer-type normalized \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --learning-rate 15e-4
```

### Systematic Comparison

```bash
# Run comparison across optimizers and model sizes
python run_experiment.py
```

## Model Sizes

| Size   | Layers | d_model | Heads | Parameters (approx.) |
|--------|--------|---------|-------|---------------------|
| small  | 12     | 768     | 12    | ~117M              |
| medium | 24     | 1024    | 16    | ~345M              |
| large  | 36     | 1280    | 20    | ~774M              |
| xl     | 48     | 1600    | 25    | ~1.3B              |

## Optimizer-Specific Configurations

### Muon/Muon8bit + Normalized Transformer
- **Learning Rate**: 15e-4 (higher than standard)
- **Weight Decay**: 0.0 (disabled)
- **Dropout**: 0.0 (disabled)

### AdamW + Base Transformer
- **Learning Rate**: 3e-4
- **Weight Decay**: 0.1
- **Dropout**: 0.0

## Distributed Training

### FSDP (Recommended)
```bash
python distributed_train.py --strategy fsdp --devices 4
```

### Multi-Node FSDP
```bash
# Node 0
python distributed_train.py --strategy fsdp --devices 8 --num-nodes 2

# Node 1  
python distributed_train.py --strategy fsdp --devices 8 --num-nodes 2
```

### DDP
```bash
python distributed_train.py --strategy ddp --devices 4
```
## Configuration Options

### Core Training Parameters
- `--model-size`: Model architecture size (small/medium/large/xl)
- `--optimizer`: Optimizer choice (muon/muon_8bit/adamw/adam/sgd)
- `--transformer-type`: Architecture type (normalized/base)
- `--batch-size`: Per-device batch size
- `--gradient-accumulation-steps`: Gradient accumulation steps
- `--learning-rate`: Learning rate
- `--sequence-length`: Input sequence length

### Hardware Configuration
- `--devices`: Number of devices or "auto"
- `--num-nodes`: Number of nodes for multi-node training
- `--strategy`: Training strategy (fsdp/ddp/auto)
- `--precision`: Training precision (bf16-mixed/16-mixed/32)

### Experiment Management
- `--project-name`: WandB project name
- `--experiment-name`: Specific experiment name
- `--save-dir`: Output directory for checkpoints and logs
