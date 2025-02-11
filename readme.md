📊 ##Customer Churn Detection

🔍 ##Project Overview

Customer churn refers to the percentage of customers that stop using a company's product or service over a given period. This project aims to develop a machine learning model that predicts whether a customer will churn based on their demographic and financial data.

📂 Dataset

The dataset used is Churn_Modelling.csv

It contains customer details, such as credit score, age, balance, gender, geography, and whether they exited the service (Exited column).

🛠 Project Steps

1️⃣ Reading the Dataset

📥 Load the dataset using pandas.

🔎 Explore the dataset for missing values, inconsistencies, and an overall understanding of the features.

2️⃣ Data Visualization

📊 Generate visualizations to understand:

📈 Distribution of numerical features (Age, CreditScore, Balance, EstimatedSalary).

🏷️ Distribution of categorical features (Gender, Geography).

🔥 Correlation heatmap to identify relationships between features.

🎯 Target variable distribution to check for class imbalance.

3️⃣ Feature Engineering

🔢 Encode categorical variables (e.g., Geography and Gender) using One-Hot Encoding or Label Encoding.

✅ Ensure features are clean, relevant, and properly formatted for modeling.

4️⃣ Splitting the Dataset

✂️ Split data into training (80%) and testing (20%) sets using train_test_split from sklearn.

5️⃣ Handling Class Imbalance

⚖️ If there is a class imbalance in the Exited column, apply Synthetic Minority Oversampling Technique (SMOTE) to balance the classes.

6️⃣ Feature Standardization

🎚 Normalize numerical features to ensure they are on the same scale (mean = 0, standard deviation = 1), which is crucial for certain algorithms like SVM and KNN.

7️⃣ Model Training and Evaluation

🤖 Train the following machine learning models:

🔍 K-Nearest Neighbors (KNN)

📖 Naive Bayes

💡 Support Vector Machine (SVM)

🌳 Decision Tree (DT)

📉 Evaluate models using:

✅ Accuracy

🎯 Precision and Recall

📏 F1-Score

📊 ROC-AUC Score

8️⃣ Model Comparison

⚖️ Compare all models based on performance metrics and identify the best-performing model.

🎯 Deliverables

🤖 A trained model capable of predicting customer churn.

📊 A comparative analysis of different machine learning models.

🖼️ Visual representations of results and insights from the dataset.

📁 Repository Structure

|-- Customer_Churn_Detection/
    |-- data/
        |-- Churn_Modelling.csv
    |-- file.ipynb
    |-- README.md

🚀 How to Run the Project

🔽 Clone the repository:

git clone https://github.com/ayaatef11/Customer_Churn_Detection.git
cd Customer_Churn_Detection

📦 Install required dependencies:

pip install -r requirements.txt

🏗 Run the preprocessing and training scripts:

python src/data_processing.py
python src/model_training.py

📊 View results and model evaluation:

python src/evaluation.py

🛠 Dependencies

🐍 Python 3.x

📊 Pandas

🔢 NumPy

🏗 Scikit-learn

📈 Matplotlib

🎨 Seaborn

⚖️ Imbalanced-learn (for SMOTE, if needed)

📈 Results and Findings

📊 The project evaluates four models to determine the best one for predicting customer churn.

🏆 The best-performing model is selected based on accuracy, precision, recall, F1-score, and ROC-AUC score.

📉 Visualizations and charts are provided to illustrate key findings.

👥 Contributors

Your Name

📜 License

This project is licensed under the MIT License.

