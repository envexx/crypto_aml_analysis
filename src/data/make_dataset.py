import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(classes_path, features_path, edges_path, users_path, tx_user_mapping_path):
    """
    Load all required datasets
    """
    logger.info("Loading datasets...")
    
    try:
        # Load transaction classes
        classes_df = pd.read_csv(classes_path)
        logger.info(f"Loaded classes data: {classes_df.shape}")
        logger.info(f"Classes columns: {classes_df.columns.tolist()}")
        
        # Load transaction features
        features_df = pd.read_csv(features_path)
        logger.info(f"Loaded features data: {features_df.shape}")
        logger.info(f"Features first few columns: {features_df.columns[:5].tolist()}")
        
        # Load transaction edges
        edges_df = pd.read_csv(edges_path)
        logger.info(f"Loaded edges data: {edges_df.shape}")
        logger.info(f"Edges columns: {edges_df.columns.tolist()}")
        
        # Load synthetic users
        users_df = pd.read_csv(users_path)
        logger.info(f"Loaded users data: {users_df.shape}")
        
        # Load transaction-user mapping
        tx_user_mapping_df = pd.read_csv(tx_user_mapping_path)
        logger.info(f"Loaded tx-user mapping: {tx_user_mapping_df.shape}")
        logger.info(f"Tx-user mapping columns: {tx_user_mapping_df.columns.tolist()}")
        
        return {
            'classes': classes_df,
            'features': features_df,
            'edges': edges_df,
            'users': users_df,
            'tx_user_mapping': tx_user_mapping_df
        }
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def process_transaction_data(data_dict):
    """
    Process transaction data from Elliptic dataset
    """
    logger.info("Processing transaction data...")
    
    try:
        # Extract dataframes
        features_df = data_dict['features']
        classes_df = data_dict['classes']
        tx_user_mapping_df = data_dict['tx_user_mapping']
        users_df = data_dict['users']
        
        # Check column names to ensure they match
        logger.info(f"Features DataFrame first column name: {features_df.columns[0]}")
        logger.info(f"Classes DataFrame first column name: {classes_df.columns[0]}")
        logger.info(f"Transaction-User mapping DataFrame first column name: {tx_user_mapping_df.columns[0]}")
        
        # Ensure column names are consistent for join
        # Assume the first column in each DataFrame is the transaction ID
        tx_id_col_features = features_df.columns[0]
        tx_id_col_classes = classes_df.columns[0]
        tx_id_col_mapping = tx_user_mapping_df.columns[0]
        
        # Rename columns if necessary to ensure consistency
        features_df = features_df.rename(columns={tx_id_col_features: "txId"})
        classes_df = classes_df.rename(columns={tx_id_col_classes: "txId"})
        tx_user_mapping_df = tx_user_mapping_df.rename(columns={tx_id_col_mapping: "txId"})
        
        # Merge features with classes
        tx_df = features_df.merge(classes_df, on='txId', how='left')
        logger.info(f"Merged features and classes: {tx_df.shape}")
        
        # Rename time step column (assuming column '1' is the time step)
        if '1' in tx_df.columns:
            tx_df = tx_df.rename(columns={'1': 'time_step'})
            logger.info("Renamed column '1' to 'time_step'")
        
        # For illustration, rename some key feature columns
        # This might need adjustment based on actual data
        if '2' in tx_df.columns:
            tx_df = tx_df.rename(columns={
                '2': 'total_input',  # Assuming this is the total input/amount
                '3': 'total_output' if '3' in tx_df.columns else None,
                '4': 'fee' if '4' in tx_df.columns else None
            })
            # Remove None values from columns dict
            tx_df = tx_df.rename(columns={k: v for k, v in tx_df.columns.items() if v is not None})
            logger.info("Renamed feature columns")
        
        # Generate synthetic timestamps based on time_step
        # Assuming each time_step is 1 week apart, starting from 1 year ago
        if 'time_step' in tx_df.columns:
            start_date = datetime.now() - timedelta(days=365)  # 1 year ago
            
            time_step_to_date = {}
            for step in tx_df['time_step'].unique():
                # Convert step to int if it's not already
                try:
                    step_int = int(step)
                    time_step_to_date[step] = start_date + timedelta(days=step_int*7)
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert time_step {step} to int")
                    continue
            
            tx_df['transaction_date'] = tx_df['time_step'].map(time_step_to_date)
            logger.info("Added synthetic transaction dates")
        
        # Add user information by merging with tx_user_mapping
        logger.info(f"Merging transactions with user mapping using column 'txId'")
        tx_df = tx_df.merge(tx_user_mapping_df, on='txId', how='left')
        logger.info(f"After merging with user mapping: {tx_df.shape}")
        
        logger.info(f"Merging with users using column 'user_id'")
        tx_df = tx_df.merge(users_df, on='user_id', how='left')
        logger.info(f"After merging with users: {tx_df.shape}")
        logger.info("Added user information to transactions")
        
        # Create a more intuitive 'label' column
        if 'class' in tx_df.columns:
            tx_df['label'] = tx_df['class'].map({
                '1': 'licit',
                1: 'licit',
                '2': 'illicit',
                2: 'illicit',
                'unknown': 'unknown'
            })
            logger.info("Mapped transaction classes to labels")
        
        return tx_df
    
    except Exception as e:
        logger.error(f"Error processing transaction data: {e}")
        logger.error(f"Details of the exception: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def calculate_user_metrics(processed_tx_df):
    """
    Calculate user-level metrics for AML analysis
    """
    logger.info("Calculating user metrics...")
    
    try:
        # Group by user_id
        agg_dict = {
            'txId': 'count',  # transaction count
        }
        
        # Add date-based aggregations if available
        if 'transaction_date' in processed_tx_df.columns:
            agg_dict['transaction_date'] = ['min', 'max']
        
        # Add amount-based aggregations if available
        if 'total_input' in processed_tx_df.columns:
            agg_dict['total_input'] = ['sum', 'mean', 'max']
        
        # Add label-based aggregations if available
        if 'label' in processed_tx_df.columns:
            agg_dict['label'] = [lambda x: (x == 'illicit').sum()]
        
        # Add KYC level if available
        if 'kyc_level' in processed_tx_df.columns:
            agg_dict['kyc_level'] = 'first'
        
        # Perform aggregation
        user_metrics = processed_tx_df.groupby('user_id').agg(agg_dict)
        
        # Flatten MultiIndex columns
        user_metrics.columns = ['_'.join(col).strip() for col in user_metrics.columns.values]
        
        # Rename columns to more readable names
        column_mapping = {
            'txId_count': 'transaction_count'
        }
        
        if 'transaction_date_min' in user_metrics.columns:
            column_mapping['transaction_date_min'] = 'first_transaction'
        if 'transaction_date_max' in user_metrics.columns:
            column_mapping['transaction_date_max'] = 'last_transaction'
        if 'total_input_sum' in user_metrics.columns:
            column_mapping['total_input_sum'] = 'total_amount'
        if 'total_input_mean' in user_metrics.columns:
            column_mapping['total_input_mean'] = 'avg_amount'
        if 'total_input_max' in user_metrics.columns:
            column_mapping['total_input_max'] = 'max_amount'
        if 'label_<lambda_0>' in user_metrics.columns:
            column_mapping['label_<lambda_0>'] = 'illicit_tx_count'
        
        # Apply column renaming
        user_metrics = user_metrics.rename(columns=column_mapping)
        
        # Reset index to make user_id a regular column
        user_metrics = user_metrics.reset_index()
        
        # Calculate illicit transaction percentage
        if 'illicit_tx_count' in user_metrics.columns and 'transaction_count' in user_metrics.columns:
            user_metrics['illicit_tx_percent'] = (user_metrics['illicit_tx_count'] / 
                                              user_metrics['transaction_count'] * 100).fillna(0)
        
        # Calculate activity timespan in days
        if 'first_transaction' in user_metrics.columns and 'last_transaction' in user_metrics.columns:
            user_metrics['activity_timespan_days'] = (
                pd.to_datetime(user_metrics['last_transaction']) - 
                pd.to_datetime(user_metrics['first_transaction'])
            ).dt.days
            
            # Avoid division by zero
            user_metrics['activity_timespan_days'] = user_metrics['activity_timespan_days'].replace(0, 1)
            
            # Calculate transactions per day
            user_metrics['tx_per_day'] = user_metrics['transaction_count'] / user_metrics['activity_timespan_days']
        
        return user_metrics
    
    except Exception as e:
        logger.error(f"Error calculating user metrics: {e}")
        raise

def main(input_dir='data/raw', output_dir='data/processed'):
    """
    Main function to process all data
    """
    logger.info("Starting data processing pipeline...")
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set file paths
    classes_path = os.path.join(input_dir, 'elliptic_txs_classes.csv')
    features_path = os.path.join(input_dir, 'elliptic_txs_features.csv')
    edges_path = os.path.join(input_dir, 'elliptic_txs_edgelist.csv')
    users_path = os.path.join(input_dir, 'users.csv')
    tx_user_mapping_path = os.path.join(input_dir, 'tx_user_mapping.csv')
    
    try:
        # Step 1: Load all data
        data_dict = load_data(
            classes_path, features_path, edges_path, 
            users_path, tx_user_mapping_path
        )
        
        # Step 2: Process transaction data
        processed_tx_df = process_transaction_data(data_dict)
        
        # Step 3: Calculate user metrics
        user_metrics_df = calculate_user_metrics(processed_tx_df)
        
        # Step 4: Save processed data
        processed_tx_path = os.path.join(output_dir, 'processed_transactions.csv')
        user_metrics_path = os.path.join(output_dir, 'user_metrics.csv')
        
        processed_tx_df.to_csv(processed_tx_path, index=False)
        user_metrics_df.to_csv(user_metrics_path, index=False)
        
        logger.info(f"Saved processed transactions to {processed_tx_path}")
        logger.info(f"Saved user metrics to {user_metrics_path}")
        
        logger.info("Data processing completed successfully")
        
        return processed_tx_df, user_metrics_df
    
    except Exception as e:
        logger.error(f"Error in data processing pipeline: {e}")
        raise

if __name__ == "__main__":
    main()