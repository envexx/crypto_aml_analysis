import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set random seed for consistent results
np.random.seed(42)

def detect_file_type_and_load(file_path):
    """
    Detect file type and load data accordingly
    """
    print(f"Attempting to load file: {file_path}")
    try:
        # Try loading as Excel first
        try:
            df = pd.read_excel(file_path)
            print("Successfully loaded as Excel file.")
            return df
        except Exception as excel_error:
            print(f"Excel load failed: {excel_error}")
            
            # Try loading as CSV with different delimiters
            for delimiter in [',', '\t', ';']:
                try:
                    df = pd.read_csv(file_path, delimiter=delimiter)
                    if len(df.columns) > 1:  # Successful parse should have multiple columns
                        print(f"Successfully loaded as CSV with delimiter '{delimiter}'")
                        return df
                except Exception:
                    pass
            
            # If we got here, try one more time with automatic delimiter detection
            try:
                df = pd.read_csv(file_path, sep=None, engine='python')
                print("Successfully loaded as CSV with automatic delimiter detection.")
                return df
            except Exception as csv_error:
                raise Exception(f"Failed to load file as Excel or CSV: {csv_error}")
    except Exception as e:
        raise Exception(f"Error loading file {file_path}: {e}")

def create_synthetic_users(elliptic_classes_path, output_path, num_users=1000):
    """
    Creates a synthetic user dataset that matches transactions in the Elliptic dataset.
    
    Parameters:
    -----------
    elliptic_classes_path : str
        Path to elliptic_txs_classes file
    output_path : str
        Path to save the users.csv file
    num_users : int
        Number of users to generate
    """
    print("Reading Elliptic dataset...")
    try:
        # Read transaction classes file with flexible loading
        classes_df = detect_file_type_and_load(elliptic_classes_path)
        
        # Check data structure
        print(f"Number of transactions read: {len(classes_df)}")
        print("Sample transaction data:")
        print(classes_df.head())
        print("Columns in the dataset:", classes_df.columns.tolist())
        
        # Get all unique txIds
        # First, check if 'txId' column exists in the dataset
        id_column = None
        if 'txId' in classes_df.columns:
            id_column = 'txId'
        elif 'txid' in classes_df.columns:
            id_column = 'txid'
        else:
            # Assume the first column is the transaction ID
            id_column = classes_df.columns[0]
            print(f"Using {id_column} as the transaction ID column")
        
        unique_tx_ids = classes_df[id_column].unique()
        print(f"Number of unique transactions: {len(unique_tx_ids)}")
        
        # Information for creating users
        # KYC levels
        kyc_levels = ['none', 'basic', 'intermediate', 'advanced']
        kyc_weights = [0.05, 0.25, 0.40, 0.30]  # Probability of each level
        
        # Countries (top 20 cryptocurrency user countries)
        countries = ['US', 'JP', 'KR', 'CN', 'UK', 'RU', 'DE', 'FR', 'SG', 'CA', 
                    'AU', 'BR', 'IN', 'ID', 'TR', 'VN', 'NG', 'ZA', 'UA', 'TH']
        country_weights = [0.2, 0.1, 0.1, 0.08, 0.08, 0.06, 0.05, 0.05, 0.04, 0.04,
                          0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01]
        
        # Start date for registration (3 years ago)
        start_date = datetime.now() - timedelta(days=365*3)
        end_date = datetime.now()
        days_range = (end_date - start_date).days
        
        # Ensure days_range is positive
        if days_range <= 0:
            days_range = 365  # Default to 1 year if there's an issue
            print(f"Warning: Calculated days_range was not positive. Using default value: {days_range}")
        
        # Generate unique IDs for users
        user_ids = [f'user_{i:06d}' for i in range(num_users)]
        
        # Generate user data
        users_data = []
        for user_id in user_ids:
            # Registration date (ensure safe random range)
            reg_days_ago = np.random.randint(1, days_range + 1)  # Ensure positive range
            registration_date = end_date - timedelta(days=reg_days_ago)
            
            # Last login date (between registration and now)
            login_days_ago = np.random.randint(0, reg_days_ago)  # This should always be a valid range
            last_login_date = end_date - timedelta(days=login_days_ago)
            
            # KYC level
            kyc_level = np.random.choice(kyc_levels, p=kyc_weights)
            
            # Country
            country_code = np.random.choice(countries, p=country_weights)
            
            # Email (dummy)
            email = f"{user_id}@example.com"
            
            # Risk score (assume random value for now)
            risk_score = round(np.random.uniform(0, 100), 2)
            
            # Verification status
            verification_status = "verified" if kyc_level in ['intermediate', 'advanced'] else "pending"
            
            users_data.append({
                'user_id': user_id,
                'email': email,
                'kyc_level': kyc_level,
                'country_code': country_code,
                'registration_date': registration_date.strftime('%Y-%m-%d %H:%M:%S'),
                'last_login_date': last_login_date.strftime('%Y-%m-%d %H:%M:%S'),
                'verification_status': verification_status,
                'initial_risk_score': risk_score
            })
        
        # Create DataFrame and save to CSV
        users_df = pd.DataFrame(users_data)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        users_df.to_csv(output_path, index=False)
        print(f"User dataset successfully created and saved to {output_path}")
        print(f"Number of users: {len(users_df)}")
        print("Sample user data:")
        print(users_df.head())
        
        return users_df, id_column
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None

# Function to create mapping between transactions and users
def create_transaction_user_mapping(elliptic_classes_path, users_df_path, output_path, id_column=None):
    """
    Creates a mapping between Elliptic transactions and synthetic users.
    
    Parameters:
    -----------
    elliptic_classes_path : str
        Path to elliptic_txs_classes file
    users_df_path : str
        Path to the created users.csv file
    output_path : str
        Path to save tx_user_mapping.csv file
    id_column : str
        Name of the transaction ID column
    """
    try:
        # Read transaction classes file
        classes_df = detect_file_type_and_load(elliptic_classes_path)
        
        # Read users file
        users_df = pd.read_csv(users_df_path)
        
        # Determine ID column if not provided
        if id_column is None:
            if 'txId' in classes_df.columns:
                id_column = 'txId'
            elif 'txid' in classes_df.columns:
                id_column = 'txid'
            else:
                # Assume the first column is the transaction ID
                id_column = classes_df.columns[0]
                print(f"Using {id_column} as the transaction ID column")
        
        # Get all unique txIds
        unique_tx_ids = classes_df[id_column].unique()
        
        # Get all user_ids
        user_ids = users_df['user_id'].tolist()
        
        # Shuffle user_ids to ensure a good distribution
        random.shuffle(user_ids)
        
        # Create Pareto distribution (80-20 rule): 20% of users have 80% of transactions
        num_high_volume_users = int(len(user_ids) * 0.2)
        # Ensure at least one high volume user
        num_high_volume_users = max(1, num_high_volume_users)
        high_volume_users = user_ids[:num_high_volume_users]
        low_volume_users = user_ids[num_high_volume_users:]
        
        # Assign transactions to users
        tx_user_mapping = []
        
        # 80% of transactions for high volume users
        high_volume_tx_count = int(len(unique_tx_ids) * 0.8)
        # Ensure at least one high volume transaction
        high_volume_tx_count = max(1, high_volume_tx_count)
        high_volume_txs = unique_tx_ids[:high_volume_tx_count]
        low_volume_txs = unique_tx_ids[high_volume_tx_count:]
        
        # Distribution for high volume users (they get more transactions)
        if len(high_volume_users) > 0:  # Avoid division by zero
            tx_per_high_volume_user = max(1, len(high_volume_txs) // len(high_volume_users))
            remainder = len(high_volume_txs) % len(high_volume_users)
            
            # Assign transactions to high volume users
            tx_idx = 0
            for user_id in high_volume_users:
                # Number of transactions for this user
                num_tx = tx_per_high_volume_user
                if remainder > 0:
                    num_tx += 1
                    remainder -= 1
                    
                # Assign transactions to this user
                for i in range(num_tx):
                    if tx_idx < len(high_volume_txs):
                        tx_user_mapping.append({
                            id_column: high_volume_txs[tx_idx],
                            'user_id': user_id
                        })
                        tx_idx += 1
        
        # More even distribution for low volume users
        # Each low volume user gets at least 1 transaction
        for i, user_id in enumerate(low_volume_users):
            if i < len(low_volume_txs):
                tx_user_mapping.append({
                    id_column: low_volume_txs[i],
                    'user_id': user_id
                })
        
        # Create DataFrame and save to CSV
        mapping_df = pd.DataFrame(tx_user_mapping)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        mapping_df.to_csv(output_path, index=False)
        print(f"Transaction-user mapping successfully created and saved to {output_path}")
        print(f"Number of mappings: {len(mapping_df)}")
        print("Sample mapping data:")
        print(mapping_df.head())
        
        # Add statistics
        tx_per_user = mapping_df.groupby('user_id').size().reset_index(name='tx_count')
        print("\nStatistics for transactions per user:")
        print(f"Minimum transactions per user: {tx_per_user['tx_count'].min()}")
        print(f"Maximum transactions per user: {tx_per_user['tx_count'].max()}")
        print(f"Average transactions per user: {tx_per_user['tx_count'].mean():.2f}")
        
        return mapping_df
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Path to Elliptic dataset - try both extensions
    if os.path.exists("data/raw/elliptic_txs_classes.csv"):
        elliptic_classes_path = "data/raw/elliptic_txs_classes.csv"
    elif os.path.exists("data/raw/elliptic_txs_classes.xlsx"):
        elliptic_classes_path = "data/raw/elliptic_txs_classes.xlsx"
    else:
        # Ask user for the correct path
        print("Could not find elliptic_txs_classes file automatically.")
        print("Current directory:", os.getcwd())
        print("Files in data/raw directory:", os.listdir("data/raw") if os.path.exists("data/raw") else "directory doesn't exist")
        elliptic_classes_path = input("Please enter the full path to the elliptic_txs_classes file: ")
    
    # Make sure the data/raw directory exists
    os.makedirs("data/raw", exist_ok=True)
    
    # Output path for user dataset
    users_output_path = "data/raw/users.csv"
    
    # Output path for transaction-user mapping
    mapping_output_path = "data/raw/tx_user_mapping.csv"
    
    # Create user dataset
    users_df, id_column = create_synthetic_users(elliptic_classes_path, users_output_path, num_users=1000)
    
    if users_df is not None:
        # Create transaction-user mapping
        create_transaction_user_mapping(elliptic_classes_path, users_output_path, mapping_output_path, id_column)