# scripts/deploy_model.py

import argparse
import subprocess
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Deploy Trained Model")
    parser.add_argument('--model_path', type=str, help='Path to the trained model', required=True)
    parser.add_argument('--deploy_dir', type=str, help='Directory to deploy the model', default='deploy/')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure deployment directory exists
    os.makedirs(args.deploy_dir, exist_ok=True)
    
    # Copy model to deployment directory
    subprocess.run(['cp', args.model_path, args.deploy_dir], check=True)
    
    # Additional deployment steps (e.g., uploading to cloud storage) can be added here
    
    print(f"Model deployed to {args.deploy_dir}")

if __name__ == "__main__":
    main()
