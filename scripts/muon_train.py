#!/usr/bin/env python3

import subprocess
import sys

def main():    
    cmd = [
        "python", "distributed_train.py",
        "--model-size", "small",
        "--optimizer", "muon", 
        "--transformer-type", "base",
        "--batch-size", "8",
        "--gradient-accumulation-steps", "1",
        "--learning-rate", "15e-4",
        "--sequence-length", "2048",
        "--project-name", "muon-8bit",
        "--experiment-name", "muon-small-base-debug",
        "--save-dir", "./muon_outputs",
        "--strategy", "auto",
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
