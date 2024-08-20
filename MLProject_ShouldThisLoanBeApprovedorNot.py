import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Data Preprocessing
# Specify data types while reading the CSV file
dtype_mapping = {
    'SBA_Appv': float,  # You can specify data types for other columns as needed
}

# Define a function to clean currency columns and convert to float
def clean_currency(x):
    try:
        # Remove leading and trailing white spaces, dollar signs, and commas
        x = x.strip().replace('$', '').replace(',', '')
        # Attempt to convert to float
        return float(x)
    except ValueError:
        # Handle any values that can't be converted to float (e.g., 'N/A')
        return np.nan  # You can choose to replace with another value or drop these rows later

# Read the CSV file with the updated clean_currency function
df = pd.read_csv("SBAnational.csv", dtype=dtype_mapping, low_memory=False, converters={'GrAppv': clean_currency})

# Handle missing values and data cleaning (You can customize this based on your dataset)
# For simplicity, we'll drop rows with missing values in this example
df.dropna(inplace=True)

# Rest of the code for EDA, feature engineering, model building, and evaluation



# Rest of the code for EDA, feature engineering, model building, and evaluation


# Step 2: Exploratory Data Analysis (EDA)
# Visualize the data to gain insights
plt.figure(figsize=(12, 6))

# Distribution of Loan Status
plt.subplot(1, 2, 1)
sns.countplot(x='MIS_Status', data=df)
plt.title('Distribution of Loan Status')

# Distribution of Loan Amounts
plt.subplot(1, 2, 2)
sns.histplot(data=df, x='GrAppv', bins=20, kde=True)
plt.title('Distribution of Loan Amounts')

plt.tight_layout()
plt.show()

# Explore relationships between features and target
plt.figure(figsize=(12, 6))

# Loan Term vs Loan Status
plt.subplot(2, 2, 1)
sns.boxplot(x='MIS_Status', y='LoanTerm', data=df)
plt.title('Loan Term vs Loan Status')

# IsFranchise vs Loan Status
plt.subplot(2, 2, 2)
sns.countplot(x='IsFranchise', hue='MIS_Status', data=df)
plt.title('IsFranchise vs Loan Status')

# ApprovalRatio vs Loan Status
plt.subplot(2, 2, 3)
sns.boxplot(x='MIS_Status', y='ApprovalRatio', data=df)
plt.title('ApprovalRatio vs Loan Status')

plt.tight_layout()
plt.show()

# Step 3: Feature Engineering
# Here are some common feature engineering techniques:

# Feature 1: Loan Term
df['LoanTerm'] = (df['DisbursementDate'] - df['ApprovalDate']).dt.days

# Feature 2: Create a binary feature indicating if the borrower is a franchise business
df['IsFranchise'] = df['FranchiseCode'].apply(lambda x: 1 if x != 0 else 0)

# Feature 3: Total amount approved as a percentage of the loan amount requested
df['ApprovalRatio'] = df['SBA_Appv'] / df['GrAppv']

# Step 4: Model Building
# Split the dataset
X = df.drop(columns=['MIS_Status'])
y = df['MIS_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a model (e.g., Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

