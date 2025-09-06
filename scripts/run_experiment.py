#!/usr/bin/env python3

"""
Example script demonstrating how to run training experiments with different optimizers and model sizes.
This script shows how to systematically compare Muon and Muon8bit optimizers across different configurations.
"""

import subprocess
import sys
import time
from typing import List, Dict, Any

def run_experiment(config: Dict[str, Any]) -> None:
    """Run a single training experiment with the given configuration."""
    
    cmd = ["python", "distributed_train.py"]
    
    # Add all configuration parameters as command line arguments
    for key, value in config.items():
        if isinstance(value, bool) and value:
            cmd.append(f"--{key.replace('_', '-')}")
        elif not isinstance(value, bool):
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    print(f"Running experiment: {config['experiment_name']}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Experiment {config['experiment_name']} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment {config['experiment_name']} failed")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Run a series of experiments comparing optimizers and model sizes."""
    
    # Base configuration
    base_config = {
        "batch_size": 8,
        "gradient_accumulation_steps": 4,
        "learning_rate": 3e-4,
        "max_epochs": 1,
        "sequence_length": 2048,
        "project_name": "muon-optimizer-comparison",
        "save_dir": "./experiment_outputs",
        "strategy": "fsdp",
        "precision": "bf16-mixed",
        "devices": "auto",
    }
    
    experiments = []    
    model_sizes = ["small", "medium"]
    optimizers = ["muon", "muon_8bit", "adamw"]
    transformer_types = ["base"]
    
    for model_size in model_sizes:
        for optimizer in optimizers:
            for transformer_type in transformer_types:
                config = base_config.copy()
                config.update({
                    "model_size": model_size,
                    "optimizer": optimizer,
                    "transformer_type": transformer_type,
                    "experiment_name": f"{model_size}-{optimizer}-{transformer_type}",
                })
                
                if optimizer in ["muon", "muon_8bit"]:
                    if transformer_type == "normalized":
                        config["learning_rate"] = 15e-4
                    else:
                        config["learning_rate"] = 3e-4
                elif optimizer == "adamw":
                    config["learning_rate"] = 3e-4
                
                experiments.append(config)
    
    print(f"Planning to run {len(experiments)} experiments:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['experiment_name']}")
    
    response = input(f"\nProceed with running {len(experiments)} experiments? (y/n): ")
    if response.lower() not in ['y', 'yes']:
        print("Experiments cancelled.")
        return
    
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    for i, config in enumerate(experiments, 1):
        print(f"\n{'='*60}")
        print(f"Running experiment {i}/{len(experiments)}: {config['experiment_name']}")
        print(f"{'='*60}")
        
        success = run_experiment(config)
        if success:
            successful += 1
        else:
            failed += 1
        
        time.sleep(2)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {duration/60:.1f} minutes")
    print(f"Average time per experiment: {duration/len(experiments)/60:.1f} minutes")
    
    if failed > 0:
        print(f"\n⚠️  {failed} experiments failed. Check logs for details.")
        sys.exit(1)
    else:
        print(f"\n🎉 All experiments completed successfully!")

if __name__ == "__main__":
    main()
