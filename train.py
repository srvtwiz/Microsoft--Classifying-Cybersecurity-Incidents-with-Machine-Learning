import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import DMatrix
import xgboost as xgb

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

data=pd.read_csv("C:/Users/srvik/Desktop/Train.csv")
print (data)
print(data.isnull().sum())
data=data.drop(['MitreTechniques','ActionGrouped','ActionGranular','EmailClusterId','ThreatFamily','ResourceType','Roles',
                'AntispamDirection','SuspicionLevel','Timestamp','LastVerdict'],axis=1)
print(data.info())
print(data['IncidentGrade'].value_counts())
print(data['EvidenceRole'].value_counts())
print(data['EntityType'].value_counts())
print(data['Category'].value_counts())

numeric_col= data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_col.corr()
print(correlation_matrix)
plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()

threshold = 0.6
high_corr = correlation_matrix[correlation_matrix > threshold]
print(high_corr)

data=data.drop(['ApplicationName','FileName','FolderPath','State','City','OSVersion','OSFamily','DeviceName',
                'RegistryValueData','RegistryKey','AccountName','AccountUpn','AccountSid','RegistryValueData','RegistryKey'],axis=1)
final_df=data.dropna()

X=final_df.drop(columns=['IncidentGrade'])
Y=final_df['IncidentGrade']

label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# Select column names for numeric and categorical columns
numeric_col = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_col = X.select_dtypes(include='object').columns.tolist()

# Create transformations for numeric and categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_col),
        
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_col)
    ]
)

label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# Now proceed with splitting and model evaluation
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_encoded, test_size=0.2, stratify=Y_encoded, random_state=69)

# List of models and hyperparameter grids
models = [
    {
        "name": "LogisticRegression",
        "model": LogisticRegression(random_state=42, max_iter=1000),
        "params": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__penalty": ['l2']
        }
    },
    {
        "name": "RandomForest",
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [None, 10, 20, 30],
            "classifier__min_samples_split": [2, 5, 10]
        }
    },
    {
        "name": "XGBoost",
        "model": XGBClassifier(tree_method='hist', device = "cuda", random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2]
        }
    }
]

# Rest of the code continues...


# Function to preprocess and move data to GPU
def preprocess_to_gpu(pipeline, X):
    # Preprocess data using the pipeline's preprocessor
    X_processed = pipeline.named_steps['preprocessor'].transform(X)
    
    # Return processed data as a regular NumPy array (not DMatrix)
    return X_processed

# Evaluate models without hyperparameter tuning
results = []
best_model_name = None
best_macro_f1 = 0
best_model = None

for m in models:
    print(f"Training {m['name']} with default parameters...")
    
    # Create a pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', m['model'])
    ])
    
    if m['name'] == "XGBoost":
        # Preprocess and move data to GPU (or keep it on CPU if desired)
        X_train_gpu = preprocess_to_gpu(pipeline, X_train)
        X_test_gpu = preprocess_to_gpu(pipeline, X_test)

        # Train XGBoost (pass processed data directly)
        pipeline.named_steps['classifier'].fit(X_train_gpu, Y_train)
        Y_pred = pipeline.named_steps['classifier'].predict(X_test_gpu)
    else:
        # Train other models
        pipeline.fit(X_train, Y_train)
        Y_pred = pipeline.predict(X_test)
    
    # Metrics calculation
    macro_f1 = f1_score(Y_test, Y_pred, average='macro')
    precision = precision_score(Y_test, Y_pred, average='macro')
    recall = recall_score(Y_test, Y_pred, average='macro')
    
    print(f"Classification Report for {m['name']}:")
    print(classification_report(Y_test, Y_pred))
    
    # Store results
    results.append({
        "Model": m['name'],
        "Macro F1": macro_f1,
        "Precision": precision,
        "Recall": recall
    })
    
    # Check if this is the best model so far
    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_model_name = m['name']
        best_model = m

# Display results
print("\nModel Evaluation Results:")
for result in results:
    print(result)

print(f"\nBest Model: {best_model_name} with Macro F1: {best_macro_f1}")

# Hyperparameter tuning for the best model
if best_model:
    print(f"\nStarting hyperparameter tuning for the best model: {best_model_name}...")
    
    # Create pipeline for the best model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', best_model['model'])
    ])
    
    if best_model_name == "XGBoost":
        # Preprocess data for XGBoost and keep data as a NumPy array
        X_train_gpu = preprocess_to_gpu(pipeline, X_train)
        param_grid = best_model['params']

        # Hyperparameter tuning with GridSearchCV
        grid_search = GridSearchCV(
            estimator=pipeline.named_steps['classifier'],
            param_grid=param_grid,
            scoring='f1_macro',
            cv=5,
            verbose=1,
            n_jobs=-1
        )
        grid_search.fit(X_train_gpu, Y_train)
        tuned_model = grid_search.best_estimator_
    else:
        # Hyperparameter tuning for other models
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=best_model['params'],
            scoring='f1_macro',
            cv=5,
            verbose=1,
            n_jobs=-1
        )
        grid_search.fit(X_train, Y_train)
        tuned_model = grid_search.best_estimator_
    
    # Best model evaluation
    Y_pred = tuned_model.predict(X_test)
    
    # Metrics calculation
    macro_f1 = f1_score(Y_test, Y_pred, average='macro')
    precision = precision_score(Y_test, Y_pred, average='macro')
    recall = recall_score(Y_test, Y_pred, average='macro')
    
    print(f"\nBest parameters for {best_model_name}: {grid_search.best_params_}")
    print(f"Classification Report for {best_model_name} after hyperparameter tuning:")
    print(classification_report(Y_test, Y_pred))
    
    # Display final results
    print("\nFinal Tuned Model Results:")
    print({
        "Model": best_model_name,
        "Best Params": grid_search.best_params_,
        "Macro F1": macro_f1,
        "Precision": precision,
        "Recall": recall
    })
