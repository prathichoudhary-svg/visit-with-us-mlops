import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(filepath, output_dir):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
        
    # 1. Load the dataset
    df = pd.read_csv(filepath)
    print(f"Loaded dataset from {filepath}. Shape: {df.shape}")
    
    # 2. Data cleaning
    # Drop 'Unnamed: 0' (index artifacts) if present, and 'CustomerID' (as it is a unique identifier and not predictive)
    cols_to_drop = [col for col in ['Unnamed: 0', 'CustomerID'] if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped unnecessary columns: {cols_to_drop}")
        
    # 3. Split the dataset (80% Train, 20% Test, stratified by target variable 'ProdTaken')
    if 'ProdTaken' not in df.columns:
        raise ValueError("Target column 'ProdTaken' not found in the dataset.")
        
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['ProdTaken'])
    
    # Create target directory for splits
    os.makedirs(output_dir, exist_ok=True)
    
    # Save locally as CSVs
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Successfully split and saved files:")
    print(f" - Train set: {train_path} (Shape: {train_df.shape})")
    print(f" - Test set: {test_path} (Shape: {test_df.shape})")

if __name__ == '__main__':
    input_csv = 'visit_with_us_mlops/data/tourism.csv'
    output_folder = 'visit_with_us_mlops/data/processed'
    prepare_data(input_csv, output_folder)
