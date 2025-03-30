import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import logging
import time
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Load user metrics data for anomaly detection
    
    Parameters:
    -----------
    file_path : str
        Path to the user metrics CSV file
        
    Returns:
    --------
    pandas.DataFrame
        User metrics dataframe
    """
    logger.info(f"Loading data from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data with shape {df.shape}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def prepare_features(df, feature_names=None):
    """
    Prepare features for anomaly detection
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    feature_names : list, optional
        List of feature names to use. If None, use default features.
        
    Returns:
    --------
    numpy.ndarray
        Scaled feature matrix
    list
        List of feature names used
    """
    logger.info("Preparing features for anomaly detection")
    
    # Default features if none specified
    if feature_names is None:
        feature_names = [
            'transaction_count', 'total_amount', 'avg_amount', 
            'illicit_tx_percent', 'tx_per_day'
        ]
        # Filter to include only available columns
        feature_names = [col for col in feature_names if col in df.columns]
    
    logger.info(f"Using features: {feature_names}")
    
    if len(feature_names) < 2:
        logger.warning("Less than 2 features available. Using additional or dummy features")
        if 'transaction_count' in df.columns and 'transaction_count' not in feature_names:
            feature_names.append('transaction_count')
        # Add dummy feature if still not enough
        if len(feature_names) < 2:
            df['dummy_feature'] = 1
            feature_names.append('dummy_feature')
    
    # Extract feature matrix
    X = df[feature_names].fillna(0).values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logger.info(f"Prepared feature matrix with shape {X_scaled.shape}")
    
    return X_scaled, feature_names

def evaluate_isolation_forest(X, contamination=0.05):
    """
    Evaluate Isolation Forest for anomaly detection
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    contamination : float, optional
        Expected proportion of anomalies
        
    Returns:
    --------
    numpy.ndarray
        Binary anomaly flags (1 for anomalies, 0 for normal)
    float
        Training time in seconds
    """
    logger.info(f"Evaluating Isolation Forest with contamination={contamination}")
    
    start_time = time.time()
    
    # Train model
    model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(X)
    
    # Get predictions
    predictions = model.predict(X)
    
    # Convert to binary flags (1 for anomalies, 0 for normal)
    anomalies = (predictions == -1).astype(int)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    logger.info(f"Isolation Forest found {anomalies.sum()} anomalies in {training_time:.2f} seconds")
    
    return anomalies, training_time

def evaluate_local_outlier_factor(X, contamination=0.05):
    """
    Evaluate Local Outlier Factor for anomaly detection
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    contamination : float, optional
        Expected proportion of anomalies
        
    Returns:
    --------
    numpy.ndarray
        Binary anomaly flags (1 for anomalies, 0 for normal)
    float
        Training time in seconds
    """
    logger.info(f"Evaluating Local Outlier Factor with contamination={contamination}")
    
    start_time = time.time()
    
    # Train model
    model = LocalOutlierFactor(n_neighbors=20, contamination=contamination, n_jobs=-1)
    predictions = model.fit_predict(X)
    
    # Convert to binary flags (1 for anomalies, 0 for normal)
    anomalies = (predictions == -1).astype(int)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    logger.info(f"Local Outlier Factor found {anomalies.sum()} anomalies in {training_time:.2f} seconds")
    
    return anomalies, training_time

def evaluate_one_class_svm(X, contamination=0.05):
    """
    Evaluate One-Class SVM for anomaly detection
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    contamination : float, optional
        Expected proportion of anomalies
        
    Returns:
    --------
    numpy.ndarray
        Binary anomaly flags (1 for anomalies, 0 for normal)
    float
        Training time in seconds
    """
    logger.info(f"Evaluating One-Class SVM with contamination={contamination}")
    
    start_time = time.time()
    
    # Set nu parameter (approximately equal to contamination)
    nu = contamination
    
    # Train model
    model = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
    predictions = model.fit_predict(X)
    
    # Convert to binary flags (1 for anomalies, 0 for normal)
    anomalies = (predictions == -1).astype(int)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    logger.info(f"One-Class SVM found {anomalies.sum()} anomalies in {training_time:.2f} seconds")
    
    return anomalies, training_time

def compare_models(df, feature_names=None, contamination=0.05, output_dir=None):
    """
    Compare different anomaly detection models
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    feature_names : list, optional
        List of feature names to use
    contamination : float, optional
        Expected proportion of anomalies
    output_dir : str, optional
        Directory to save results and plots
        
    Returns:
    --------
    pandas.DataFrame
        Comparison results
    dict
        Dictionary of anomaly predictions from each model
    """
    logger.info("Comparing anomaly detection models")
    
    # Prepare features
    X, features_used = prepare_features(df, feature_names)
    
    # Evaluate models
    results = {}
    predictions = {}
    
    # Isolation Forest
    anomalies_if, time_if = evaluate_isolation_forest(X, contamination)
    results['Isolation Forest'] = {
        'anomalies_count': anomalies_if.sum(),
        'anomalies_percent': anomalies_if.mean() * 100,
        'training_time': time_if
    }
    predictions['Isolation Forest'] = anomalies_if
    
    # Local Outlier Factor
    anomalies_lof, time_lof = evaluate_local_outlier_factor(X, contamination)
    results['Local Outlier Factor'] = {
        'anomalies_count': anomalies_lof.sum(),
        'anomalies_percent': anomalies_lof.mean() * 100,
        'training_time': time_lof
    }
    predictions['Local Outlier Factor'] = anomalies_lof
    
    # One-Class SVM
    anomalies_svm, time_svm = evaluate_one_class_svm(X, contamination)
    results['One-Class SVM'] = {
        'anomalies_count': anomalies_svm.sum(),
        'anomalies_percent': anomalies_svm.mean() * 100,
        'training_time': time_svm
    }
    predictions['One-Class SVM'] = anomalies_svm
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T
    logger.info("\nModel comparison results:")
    logger.info("\n" + str(results_df))
    
    # Calculate agreement between models
    agreement_if_lof = (anomalies_if == anomalies_lof).mean() * 100
    agreement_if_svm = (anomalies_if == anomalies_svm).mean() * 100
    agreement_lof_svm = (anomalies_lof == anomalies_svm).mean() * 100
    
    logger.info(f"\nModel agreement (percentage of same predictions):")
    logger.info(f"Isolation Forest vs LOF: {agreement_if_lof:.2f}%")
    logger.info(f"Isolation Forest vs One-Class SVM: {agreement_if_svm:.2f}%")
    logger.info(f"LOF vs One-Class SVM: {agreement_lof_svm:.2f}%")
    
    # Calculate ensemble prediction (majority voting)
    ensemble_predictions = ((anomalies_if + anomalies_lof + anomalies_svm) >= 2).astype(int)
    predictions['Ensemble (Majority)'] = ensemble_predictions
    
    logger.info(f"Ensemble model found {ensemble_predictions.sum()} anomalies ({ensemble_predictions.mean() * 100:.2f}%)")
    
    # Save results to CSV if output_dir specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comparison results
        results_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'))
        
        # Save predictions
        for model_name, preds in predictions.items():
            df[f'is_anomaly_{model_name.replace(" ", "_")}'] = preds
        
        df.to_csv(os.path.join(output_dir, 'anomaly_detection_results.csv'), index=False)
        
        # Create visualization of model comparison
        create_comparison_visualizations(df, predictions, features_used, output_dir)
    
    return results_df, predictions

def create_comparison_visualizations(df, predictions, features, output_dir):
    """
    Create visualizations comparing different anomaly detection models
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    predictions : dict
        Dictionary of anomaly predictions from each model
    features : list
        List of features used
    output_dir : str
        Directory to save plots
    """
    logger.info("Creating comparison visualizations")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    # Get the first two features for 2D visualization
    if len(features) >= 2:
        feature_x = features[0]
        feature_y = features[1]
    else:
        logger.warning("Not enough features for 2D visualization")
        return
    
    # Create scatter plots for each model
    plt.figure(figsize=(20, 15))
    
    for i, (model_name, anomalies) in enumerate(predictions.items()):
        plt.subplot(2, 2, i+1)
        
        # Plot normal points
        plt.scatter(df.loc[anomalies == 0, feature_x], 
                    df.loc[anomalies == 0, feature_y], 
                    c='blue', label='Normal', alpha=0.5)
        
        # Plot anomalous points
        plt.scatter(df.loc[anomalies == 1, feature_x], 
                    df.loc[anomalies == 1, feature_y], 
                    c='red', label='Anomaly', alpha=0.7)
        
        plt.title(f'{model_name} - {anomalies.sum()} Anomalies')
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'model_comparison_scatter.png'), dpi=300)
    plt.close()
    
    # Create bar chart of anomaly counts
    anomaly_counts = {model: preds.sum() for model, preds in predictions.items()}
    
    plt.figure(figsize=(10, 6))
    plt.bar(anomaly_counts.keys(), anomaly_counts.values(), color='darkred')
    plt.title('Number of Anomalies Detected by Each Model')
    plt.xlabel('Model')
    plt.ylabel('Number of Anomalies')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', 'anomaly_counts.png'), dpi=300)
    plt.close()
    
    # Try to create Venn diagram if matplotlib_venn is available
    try:
        from matplotlib_venn import venn3
        
        if 'Isolation Forest' in predictions and 'Local Outlier Factor' in predictions and 'One-Class SVM' in predictions:
            # Get sets of anomalous indices
            anomalies_if = set(np.where(predictions['Isolation Forest'] == 1)[0])
            anomalies_lof = set(np.where(predictions['Local Outlier Factor'] == 1)[0])
            anomalies_svm = set(np.where(predictions['One-Class SVM'] == 1)[0])
            
            # Create Venn diagram
            plt.figure(figsize=(10, 10))
            venn3([anomalies_if, anomalies_lof, anomalies_svm], 
                ('Isolation Forest', 'Local Outlier Factor', 'One-Class SVM'))
            plt.title('Overlap of Anomalies Detected by Different Models')
            plt.savefig(os.path.join(output_dir, 'figures', 'anomaly_overlap.png'), dpi=300)
            plt.close()
    except ImportError:
        logger.warning("matplotlib_venn not available. Skipping Venn diagram visualization.")
        
        # Alternative visualization for overlap
        plt.figure(figsize=(10, 6))
        
        # Calculate overlaps
        if 'Isolation Forest' in predictions and 'Local Outlier Factor' in predictions and 'One-Class SVM' in predictions:
            anomalies_if = set(np.where(predictions['Isolation Forest'] == 1)[0])
            anomalies_lof = set(np.where(predictions['Local Outlier Factor'] == 1)[0])
            anomalies_svm = set(np.where(predictions['One-Class SVM'] == 1)[0])
            
            # Calculate overlaps
            only_if = len(anomalies_if - anomalies_lof - anomalies_svm)
            only_lof = len(anomalies_lof - anomalies_if - anomalies_svm)
            only_svm = len(anomalies_svm - anomalies_if - anomalies_lof)
            
            if_lof = len(anomalies_if & anomalies_lof - anomalies_svm)
            if_svm = len(anomalies_if & anomalies_svm - anomalies_lof)
            lof_svm = len(anomalies_lof & anomalies_svm - anomalies_if)
            
            all_three = len(anomalies_if & anomalies_lof & anomalies_svm)
            
            # Create bar chart
            categories = ['Only IF', 'Only LOF', 'Only SVM', 'IF & LOF', 'IF & SVM', 'LOF & SVM', 'All Three']
            values = [only_if, only_lof, only_svm, if_lof, if_svm, lof_svm, all_three]
            
            plt.bar(categories, values, color='darkblue')
            plt.title('Overlap of Anomalies Detected by Different Models')
            plt.xlabel('Category')
            plt.ylabel('Number of Anomalies')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'figures', 'anomaly_overlap_bars.png'), dpi=300)
            plt.close()
    
    logger.info(f"Saved visualizations to {os.path.join(output_dir, 'figures')}")

def main(input_path, output_dir):
    """
    Main function to evaluate and compare anomaly detection models
    
    Parameters:
    -----------
    input_path : str
        Path to input CSV file
    output_dir : str
        Directory to save results
    """
    logger.info("Starting anomaly detection model evaluation")
    
    try:
        # Load data
        df = load_data(input_path)
        
        # Compare models
        results_df, predictions = compare_models(
            df, 
            contamination=0.05,
            output_dir=output_dir
        )
        
        logger.info("Model evaluation completed successfully")
        
        return results_df, predictions
    
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate anomaly detection models')
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output', required=True, help='Directory to save results')
    
    args = parser.parse_args()
    
    main(args.input, args.output)