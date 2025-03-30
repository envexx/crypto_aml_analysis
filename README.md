Crypto Transaction Pattern Analysis for AML Compliance
Show Image
📊 Project Summary
This project develops a comprehensive analytics system to support Anti-Money Laundering (AML) compliance on cryptocurrency transaction platforms. By implementing data engineering techniques, machine learning algorithms, and interactive data visualizations, the system enables the detection of suspicious transaction patterns and high-risk users.
Key Statistics

203,768 Transactions analyzed
1,000 Users profiled
14 Suspicious Users identified
42,019 Suspicious Transactions detected

🔎 Key Features
1. Transaction Label Distribution Analysis
Show Image
Analysis shows that the majority of transactions (77.1%) lack clear labels (unknown), while 20.6% of transactions are identified as illicit, and only 2.23% are classified as licit. The high proportion of unlabeled and illicit transactions highlights the importance of advanced detection systems.
2. Transaction Volume Monitoring
Show Image
The transaction volume exhibits significant fluctuations over time, with peaks reaching approximately 8,000 transactions and troughs around 1,000 transactions. This pattern analysis helps identify unusual spikes that may signal coordinated suspicious activities.
3. User Profiling
Show Image
The distribution of transaction counts per user reveals a distinct bimodal pattern, with most users having very few transactions, while a small group of users execute a significantly higher number of transactions (around 800+). This pattern suggests potential structuring or smurfing activities.
4. Advanced Anomaly Detection
Show Image
Our advanced anomaly detection system identifies 1.4% of users as potentially suspicious, using multiple machine learning algorithms including Isolation Forest, Local Outlier Factor, and One-Class SVM.
5. Suspicious User Identification
Show Image
The system successfully identified 14 suspicious users with anomalous transaction patterns. These users displayed extremely high transaction volumes (816 transactions) completed within very short timeframes (1 day), resulting in unusually high transactions per day metrics.
📈 Methodology
This project employs a multi-faceted approach to AML compliance:

Data Integration - Merging transaction data with user KYC information
Feature Engineering - Creating behavioral indicators for anomaly detection
Machine Learning - Implementing multiple anomaly detection algorithms
Ensemble Techniques - Combining multiple model outputs for higher confidence
Risk Scoring - Developing a comprehensive risk assessment framework
Interactive Visualization - Building a real-time monitoring dashboard

📊 Interactive Dashboard
An interactive Streamlit dashboard was developed to allow compliance officers to:

Filter transactions by time period, label, and KYC level
Visualize transaction patterns and anomalies
Drill down into specific suspicious users
Download reports for further investigation

📑 Detailed Reports
For in-depth analysis, please refer to the following reports:

Anomaly Detection Report - Detailed analysis of anomaly detection methods and findings
Risk Assessment Report - Comprehensive risk scoring methodology and results
User Profiling Report - Behavioral patterns and suspicious activity indicators
Overview Report - Executive summary of key findings

🛠️ Technologies Used

Python - Core programming language
Pandas & NumPy - Data manipulation and analysis
Scikit-learn - Machine learning algorithms
dbt - Data transformation workflows
Plotly & Matplotlib - Data visualization
Streamlit - Interactive dashboard development
SQL - Database queries and data modeling

🚀 Future Enhancements

Real-time Processing - Implementation of stream processing for real-time detection
Network Analysis - Graph-based analysis of transaction networks
Explainable AI - Enhanced explainability of anomaly detection results
Regulatory Reporting - Automated generation of regulatory reports
Alert Management - Case management system for investigating alerts

📞 Contact
For more information about this project, please contact:

Name: Nugrah Salam
Email: ompekp@gmail.com
GitHub: envexx


This project was developed as a portfolio piece demonstrating advanced data analysis capabilities in the cryptocurrency compliance domain.