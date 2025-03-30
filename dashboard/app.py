import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Set page configuration
st.set_page_config(
    page_title="Crypto AML Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load data
@st.cache_data
def load_data():
    """Load processed data files"""
    try:
        # Use direct paths to data files
        tx_df = pd.read_csv('data/processed/processed_transactions.csv')
        
        # Try loading enhanced user metrics with risk scores
        try:
            # First try to load the risk score version
            user_metrics_df = pd.read_csv('data/processed/anomaly_detection/user_metrics_with_risk_scores.csv')
            data_source = "Enhanced risk scores"
        except FileNotFoundError:
            # Then try to load the anomaly detection results
            try:
                user_metrics_df = pd.read_csv('data/processed/anomaly_detection/anomaly_detection_results.csv')
                data_source = "Anomaly detection results"
            except FileNotFoundError:
                # Fall back to original user metrics with basic anomaly detection
                try:
                    user_metrics_df = pd.read_csv('data/processed/user_metrics_with_anomalies.csv')
                    data_source = "Basic anomaly detection"
                except FileNotFoundError:
                    # Last resort: original user metrics
                    user_metrics_df = pd.read_csv('data/processed/user_metrics.csv')
                    data_source = "Original metrics without anomaly detection"
                    
                    # Add basic anomaly flag if not present
                    if 'is_anomaly' not in user_metrics_df.columns:
                        if 'transaction_count' in user_metrics_df.columns:
                            threshold = user_metrics_df['transaction_count'].quantile(0.95)
                            user_metrics_df['is_anomaly'] = (user_metrics_df['transaction_count'] > threshold).astype(int)
                        else:
                            user_metrics_df['is_anomaly'] = 0
        
        print(f"Data source for user metrics: {data_source}")
        print(f"Loaded transaction data: {tx_df.shape}")
        print(f"Loaded user metrics data: {user_metrics_df.shape}")
        
        # Convert date columns to datetime
        date_columns = ['transaction_date', 'first_transaction', 'last_transaction']
        for df, name in [(tx_df, 'transactions'), (user_metrics_df, 'user_metrics')]:
            for col in date_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except Exception as e:
                        print(f"Error converting {col} in {name}: {e}")
        
        return tx_df, user_metrics_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None

# Load data
tx_df, user_metrics_df = load_data()

# Title and introduction
st.title("Crypto Transaction Pattern Analysis for AML Compliance")
st.markdown("""
This dashboard presents an analysis of cryptocurrency transaction patterns to support Anti-Money Laundering (AML) compliance.
Use the sidebar for navigation and available filters.
""")

# Sidebar for filters
st.sidebar.title("Filters and Navigation")

# Check if data is loaded successfully
if tx_df is not None and user_metrics_df is not None:
    # Navigation menu
    page = st.sidebar.radio(
        "Select Page", 
        ["Overview", "Transaction Analysis", "User Profiles", "Anomaly Detection", "Risk Assessment"]
    )
    
    # Filter options
    if 'time_step' in tx_df.columns:
        time_steps = sorted(tx_df['time_step'].unique())
        selected_time_steps = st.sidebar.multiselect(
            "Filter by Time Step",
            time_steps,
            default=time_steps[:5] if len(time_steps) > 5 else time_steps
        )
    else:
        selected_time_steps = []
        
    if 'label' in tx_df.columns:
        labels = sorted([label for label in tx_df['label'].unique() if pd.notna(label)])
        selected_labels = st.sidebar.multiselect(
            "Filter by Transaction Label",
            labels,
            default=labels
        )
    else:
        selected_labels = []
    
    if 'kyc_level' in user_metrics_df.columns:
        kyc_levels = sorted([level for level in user_metrics_df['kyc_level'].unique() if pd.notna(level)])
        selected_kyc_levels = st.sidebar.multiselect(
            "Filter by KYC Level",
            kyc_levels,
            default=kyc_levels
        )
    else:
        selected_kyc_levels = []
    
    # Apply filters
    filtered_tx = tx_df.copy()
    if selected_time_steps and 'time_step' in tx_df.columns:
        filtered_tx = filtered_tx[filtered_tx['time_step'].isin(selected_time_steps)]
    if selected_labels and 'label' in tx_df.columns:
        filtered_tx = filtered_tx[filtered_tx['label'].isin(selected_labels)]
    
    filtered_users = user_metrics_df.copy()
    if selected_kyc_levels and 'kyc_level' in user_metrics_df.columns:
        filtered_users = filtered_users[filtered_users['kyc_level'].isin(selected_kyc_levels)]
    
    # Find anomaly column - check for different possible column names
    anomaly_col = None
    for col in ['is_anomaly_Ensemble_(Majority)', 'is_anomaly_Isolation_Forest', 'is_anomaly']:
        if col in filtered_users.columns:
            anomaly_col = col
            break
    
    # Page: Overview
    if page == "Overview":
        st.header("Transaction and User Overview")
        
        # Create metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{len(filtered_tx):,}")
        with col2:
            st.metric("Total Users", f"{len(filtered_users):,}")
        with col3:
            if anomaly_col:
                anomaly_count = filtered_users[anomaly_col].sum()
                st.metric("Suspicious Users", f"{anomaly_count:,}")
        with col4:
            if 'label' in filtered_tx.columns:
                illicit_count = filtered_tx[filtered_tx['label'] == 'illicit'].shape[0]
                st.metric("Suspicious Transactions", f"{illicit_count:,}")
        
        # Create visualizations
        st.subheader("Transaction Label Distribution")
        if 'label' in filtered_tx.columns:
            # Get label counts, handling NaN values
            label_counts = filtered_tx['label'].fillna('Unknown').value_counts().reset_index()
            label_counts.columns = ['Label', 'Count']
            
            if not label_counts.empty:
                fig = px.pie(label_counts, values='Count', names='Label', title='Transaction Label Distribution')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No label data available for visualization")
        
        st.subheader("Transaction Volume Over Time")
        if 'time_step' in filtered_tx.columns:
            tx_by_time = filtered_tx.groupby('time_step').size().reset_index(name='count')
            
            if not tx_by_time.empty:
                fig = px.line(tx_by_time, x='time_step', y='count', title='Transaction Volume Over Time')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No time-based transaction data available for visualization")
        
        st.subheader("User Distribution by KYC Level")
        if 'kyc_level' in filtered_users.columns:
            # Handle NaN values
            kyc_counts = filtered_users['kyc_level'].fillna('Unknown').value_counts().reset_index()
            kyc_counts.columns = ['KYC Level', 'Count']
            
            if not kyc_counts.empty:
                fig = px.bar(kyc_counts, x='KYC Level', y='Count', title='User Distribution by KYC Level')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No KYC level data available for visualization")
    
    # Page: Transaction Analysis
    elif page == "Transaction Analysis":
        st.header("Cryptocurrency Transaction Analysis")
        
        st.subheader("Transaction Amount Distribution")
        amount_col = None
        for col in ['total_input', 'amount', 'total_amount']:
            if col in filtered_tx.columns:
                amount_col = col
                break
                
        if amount_col:
            # Handle invalid values
            valid_amounts = filtered_tx[pd.to_numeric(filtered_tx[amount_col], errors='coerce').notna()]
            
            if not valid_amounts.empty:
                fig = px.histogram(
                    valid_amounts, 
                    x=amount_col,
                    nbins=50,
                    title='Transaction Amount Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid amount data available for visualization")
        else:
            st.info("Transaction amount data not available")
        
        st.subheader("Transaction Amount Comparison by Label")
        if amount_col and 'label' in filtered_tx.columns:
            # Filter for valid labels and amounts
            valid_data = filtered_tx[
                filtered_tx['label'].isin(['licit', 'illicit']) & 
                pd.to_numeric(filtered_tx[amount_col], errors='coerce').notna()
            ]
            
            if not valid_data.empty:
                fig = px.box(
                    valid_data, 
                    x='label', 
                    y=amount_col,
                    title='Transaction Amounts by Label'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid labeled transaction data available for visualization")
        
        st.subheader("Transaction Network Analysis")
        st.info("This section would show network graphs of transactions between wallets, but requires additional network analysis.")
        
        st.subheader("Transaction Details")
        if st.checkbox("Show Transaction Data"):
            st.dataframe(filtered_tx.head(100))
    
    # Page: User Profiles
    elif page == "User Profiles":
        st.header("User Profiles")
        
        st.subheader("Transactions per User Distribution")
        if 'transaction_count' in filtered_users.columns:
            # Handle invalid values
            valid_counts = filtered_users[pd.to_numeric(filtered_users['transaction_count'], errors='coerce').notna()]
            
            if not valid_counts.empty:
                fig = px.histogram(
                    valid_counts,
                    x='transaction_count',
                    nbins=50,
                    title='Transactions per User Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid transaction count data available for visualization")
        
        st.subheader("User Distribution by KYC Level")
        if 'kyc_level' in filtered_users.columns:
            # Handle NaN values
            kyc_counts = filtered_users['kyc_level'].fillna('Unknown').value_counts().reset_index()
            kyc_counts.columns = ['KYC Level', 'Count']
            
            if not kyc_counts.empty:
                fig = px.bar(
                    kyc_counts, 
                    x='KYC Level', 
                    y='Count',
                    title='User Distribution by KYC Level'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No KYC level data available for visualization")
        
        st.subheader("User Country Distribution")
        if 'country_code' in filtered_users.columns:
            # Handle NaN values
            country_counts = filtered_users['country_code'].fillna('Unknown').value_counts().head(10).reset_index()
            country_counts.columns = ['Country', 'Count']
            
            if not country_counts.empty:
                fig = px.bar(
                    country_counts, 
                    x='Country', 
                    y='Count',
                    title='Top User Countries'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No country data available for visualization")
        
        st.subheader("User Details")
        if st.checkbox("Show User Data"):
            st.dataframe(filtered_users.head(100))
    
    # Page: Anomaly Detection
    elif page == "Anomaly Detection":
        st.header("Anomaly Detection")
        
        # Find available anomaly models
        anomaly_models = [col.replace('is_anomaly_', '') for col in filtered_users.columns if col.startswith('is_anomaly_')]
        
        if anomaly_models:
            st.success(f"Multiple anomaly detection models available: {', '.join(anomaly_models)}")
            
            # Let user select which model to view
            selected_model = st.selectbox(
                "Select anomaly detection model to view",
                options=["Ensemble (Majority)"] + [model for model in anomaly_models if model != "Ensemble_(Majority)"],
                index=0
            )
            
            # Convert selection to column name
            if selected_model == "Ensemble (Majority)":
                selected_col = "is_anomaly_Ensemble_(Majority)"
            else:
                selected_col = f"is_anomaly_{selected_model}"
            
            # Check if column exists
            if selected_col in filtered_users.columns:
                st.subheader(f"Normal vs Anomalous User Distribution - {selected_model}")
                anomaly_counts = filtered_users[selected_col].value_counts().reset_index()
                anomaly_counts.columns = ['Is Anomaly', 'Count']
                anomaly_counts['Status'] = anomaly_counts['Is Anomaly'].map({0: 'Normal', 1: 'Anomaly'})
                
                if not anomaly_counts.empty:
                    fig = px.pie(
                        anomaly_counts, 
                        values='Count', 
                        names='Status',
                        title=f'Normal vs Anomalous Users - {selected_model}'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Scatter plot of anomalies
                st.subheader(f"Anomaly Visualization - {selected_model}")
                
                # Find columns for visualization
                x_column = 'transaction_count' if 'transaction_count' in filtered_users.columns else None
                y_column = None
                for col in ['total_amount', 'avg_amount']:
                    if col in filtered_users.columns:
                        y_column = col
                        break
                
                if x_column and y_column:
                    # Map anomaly status
                    filtered_users['Status'] = filtered_users[selected_col].map({0: 'Normal', 1: 'Anomaly'})
                    
                    # Create scatter plot
                    fig = px.scatter(
                        filtered_users,
                        x=x_column,
                        y=y_column,
                        color='Status',
                        hover_name='user_id',
                        title=f'Anomaly Detection: {x_column.replace("_", " ").title()} vs {y_column.replace("_", " ").title()}',
                        color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Model comparison if multiple models available
                if len(anomaly_models) > 1:
                    st.subheader("Model Agreement Analysis")
                    
                    # Calculate how many models flag each user
                    if 'anomaly_count' in filtered_users.columns:
                        # Create histogram of how many models flagged each user
                        fig = px.histogram(
                            filtered_users,
                            x='anomaly_count',
                            title='Distribution of Model Agreement',
                            labels={'anomaly_count': 'Number of Models Flagging as Anomaly'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show users flagged by all models
                        st.subheader("High Confidence Anomalies")
                        st.write("Users flagged as anomalous by all models:")
                        
                        common_anomalies = filtered_users[filtered_users['anomaly_count'] == len(anomaly_models)]
                        if not common_anomalies.empty:
                            st.dataframe(common_anomalies.sort_values('transaction_count', ascending=False))
                        else:
                            st.info("No users were flagged by all models")
            else:
                st.warning(f"Selected model {selected_model} data not found in the dataset")
        elif anomaly_col:
            # Basic anomaly visualization if only simple anomaly column is available
            st.subheader("Normal vs Anomalous User Distribution")
            anomaly_counts = filtered_users[anomaly_col].value_counts().reset_index()
            anomaly_counts.columns = ['Is Anomaly', 'Count']
            anomaly_counts['Status'] = anomaly_counts['Is Anomaly'].map({0: 'Normal', 1: 'Anomaly'})
            
            if not anomaly_counts.empty:
                fig = px.pie(
                    anomaly_counts, 
                    values='Count', 
                    names='Status',
                    title='Normal vs Anomalous User Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Basic scatter plot
            st.subheader("Anomaly Visualization")
            if 'transaction_count' in filtered_users.columns:
                y_column = 'total_amount' if 'total_amount' in filtered_users.columns else 'transaction_count'
                
                # Map anomaly status
                filtered_users['Status'] = filtered_users[anomaly_col].map({0: 'Normal', 1: 'Anomaly'})
                
                # Create scatter plot
                fig = px.scatter(
                    filtered_users,
                    x='transaction_count',
                    y=y_column,
                    color='Status',
                    hover_name='user_id',
                    title=f'Anomaly Detection: Transaction Count vs {y_column.replace("_", " ").title()}',
                    color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No anomaly detection data available. Run the anomaly detection notebook first.")
        
        # Suspicious users table
        st.subheader("Suspicious Users")
        if anomaly_col:
            anomalous_users = filtered_users[filtered_users[anomaly_col] == 1].sort_values(
                'transaction_count', ascending=False
            )
            
            if not anomalous_users.empty:
                st.dataframe(anomalous_users)
                
                # Allow download of anomalous users list
                csv = anomalous_users.to_csv(index=False)
                st.download_button(
                    label="Download Suspicious Users Data",
                    data=csv,
                    file_name="anomalous_users.csv",
                    mime="text/csv"
                )
            else:
                st.info("No suspicious users detected in the filtered dataset")
        else:
            st.info("Anomaly detection data not available")
    
    # NEW PAGE: Risk Assessment
    elif page == "Risk Assessment":
        st.header("AML Risk Assessment")
        
        # Risk score visualization
        if 'aml_risk_score' in filtered_users.columns:
            st.subheader("AML Risk Score Distribution")
            
            # Create histogram of risk scores
            fig = px.histogram(
                filtered_users,
                x='aml_risk_score',
                nbins=20,
                title='Distribution of AML Risk Scores'
            )
            fig.update_layout(xaxis_title="Risk Score (0-100)", yaxis_title="Number of Users")
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk category breakdown
            if 'risk_category' in filtered_users.columns:
                st.subheader("Risk Category Breakdown")
                
                # Count by risk category
                risk_counts = filtered_users['risk_category'].value_counts().reset_index()
                risk_counts.columns = ['Risk Category', 'Count']
                
                # Create order for categories
                category_order = ['Low', 'Medium', 'High', 'Very High']
                if risk_counts['Risk Category'].isin(category_order).all():
                    risk_counts['Risk Category'] = pd.Categorical(
                        risk_counts['Risk Category'], 
                        categories=category_order, 
                        ordered=True
                    )
                    risk_counts = risk_counts.sort_values('Risk Category')
                
                # Create bar chart
                fig = px.bar(
                    risk_counts,
                    x='Risk Category',
                    y='Count',
                    title='Users by Risk Category',
                    color='Risk Category',
                    color_discrete_map={
                        'Low': 'green',
                        'Medium': 'yellow',
                        'High': 'orange',
                        'Very High': 'red'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # High risk users
            st.subheader("High Risk Users")
            
            # Set threshold for high risk
            risk_threshold = st.slider("Risk Score Threshold:", 0, 100, 75)
            
            high_risk_users = filtered_users[filtered_users['aml_risk_score'] >= risk_threshold].sort_values(
                'aml_risk_score', ascending=False
            )
            
            if not high_risk_users.empty:
                st.write(f"Found {len(high_risk_users)} users with risk score >= {risk_threshold}")
                
                # Show KYC level distribution for high risk users
                if 'kyc_level' in high_risk_users.columns:
                    kyc_counts = high_risk_users['kyc_level'].value_counts().reset_index()
                    kyc_counts.columns = ['KYC Level', 'Count']
                    
                    fig = px.pie(
                        kyc_counts,
                        values='Count',
                        names='KYC Level',
                        title='KYC Levels of High Risk Users'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display high risk users
                st.dataframe(high_risk_users)
                
                # Allow download
                csv = high_risk_users.to_csv(index=False)
                st.download_button(
                    label="Download High Risk Users Data",
                    data=csv,
                    file_name="high_risk_users.csv",
                    mime="text/csv"
                )
            else:
                st.info(f"No users found with risk score >= {risk_threshold}")
            
            # Risk factors analysis
            st.subheader("Risk Factors Analysis")
            st.write("Factors contributing to high risk scores:")
            
            # Create scatter matrix for understanding relationship between risk factors
            if len(filtered_users) > 0:
                risk_factors = [
                    'transaction_count', 'total_amount', 'avg_amount', 'tx_per_day', 'aml_risk_score'
                ]
                available_factors = [col for col in risk_factors if col in filtered_users.columns]
                
                if len(available_factors) >= 2:
                    fig = px.scatter_matrix(
                        filtered_users,
                        dimensions=available_factors,
                        color='risk_category' if 'risk_category' in filtered_users.columns else None,
                        title="Relationships Between Risk Factors"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Risk score data not available. Run the anomaly detection notebook first to generate risk scores.")
            
            # If we have anomaly data, we can still provide some insights
            if anomaly_col:
                st.subheader("Basic Risk Assessment (based on anomaly detection)")
                
                # Count anomalies
                anomaly_count = filtered_users[anomaly_col].sum()
                st.write(f"Found {anomaly_count} users flagged as suspicious by anomaly detection")
                
                # Show potentially risky users
                st.subheader("Potentially High Risk Users")
                risky_users = filtered_users[filtered_users[anomaly_col] == 1].sort_values(
                    'transaction_count', ascending=False
                )
                
                if not risky_users.empty:
                    st.dataframe(risky_users)
else:
    st.error("Data could not be loaded. Make sure you've run the data processing script first.")

# Footer
st.markdown("---")
st.markdown("Created as part of a data analysis portfolio for the AML Compliance case study on a crypto platform.")