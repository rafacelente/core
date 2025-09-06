#!/usr/bin/env python3

import subprocess
import sys

def main():    
    cmd = [
        "python", "distributed_train.py",
        "--model-size", "small",
        "--optimizer", "muon", 
        "--transformer-type", "normalized",
        "--batch-size", "8",
        "--gradient-accumulation-steps", "4",
        "--learning-rate", "15e-4",
        "--max-epochs", "1",
        "--sequence-length", "2048",
        "--project-name", "muon-research",
        "--experiment-name", "muon-small-normalized",
        "--save-dir", "./muon_outputs",
        "--strategy", "fsdp",
        "--precision", "bf16-mixed",
    ]
    
    print("🚀 Starting Muon training experiment...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Muon training completed successfully!")
    except subprocess.CalledProcessError as e:
        print("❌ Muon training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
