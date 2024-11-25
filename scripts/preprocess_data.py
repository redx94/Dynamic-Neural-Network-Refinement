# scripts/preprocess_data.py

import argparse
import pandas as pd
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Raw Data")
    parser.add_argument('--input', type=str, help='Path to raw data file', required=True)
    parser.add_argument('--output', type=str, help='Path to save processed data', required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load raw data
    df = pd.read_csv(args.input)
    
    # Perform data cleaning, transformation, feature engineering
    # Example: Fill missing values
    df = df.fillna(method='ffill')
    
    # Example: Feature scaling or encoding
    # Add your preprocessing steps here
    
    # Save processed data
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Processed data saved to {args.output}")

if __name__ == "__main__":
    main()
