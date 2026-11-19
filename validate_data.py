import pandas as pd
import sys
import os

def validate_dataset(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist.")
        sys.exit(1)
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        sys.exit(1)
        
    # Display summary statistics and information
    print("=== Dataset Summary ===")
    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")
    print("\n=== Columns and Data Types ===")
    print(df.dtypes)
    
    print("\n=== Missing Values Summary ===")
    print(df.isnull().sum())
    
    # Example expectation validation (can be customized based on known tourism columns)
    expected_columns = ['CustomerID', 'ProdTaken'] 
    missing_cols = [col for col in expected_columns if col not in df.columns]
    
    if missing_cols:
        print(f"\nValidation Warning: Missing expected columns: {missing_cols}")
    else:
        print("\nValidation Success: Critical expected columns are present.")

if __name__ == '__main__':
    data_path = 'visit_with_us_mlops/data/tourism.csv'
    validate_dataset(data_path)
