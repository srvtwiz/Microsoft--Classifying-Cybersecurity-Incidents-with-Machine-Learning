import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV,train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import precision_score,f1_score,recall_score,classification_report,accuracy_score
import xgboost as xgb



data=pd.read_csv("C:/Users/srvik/Desktop/Train.csv")
data=data.drop(['MitreTechniques','ActionGrouped','ActionGranular','EmailClusterId','ThreatFamily','ResourceType','Roles',
                'AntispamDirection','SuspicionLevel','Timestamp','LastVerdict'],axis=1)
#dropped after correlation checking
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
preprocessor=ColumnTransformer(transformers=[('num',StandardScaler(),numeric_col),('cat',OneHotEncoder(handle_unknown='ignore'),categorical_col)])



X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.2, stratify=Y_encoded, random_state=69)


# Transform the training and testing data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)


# Convert the processed data to a DMatrix structure and enable GPU
dtrain = xgb.DMatrix(X_train_transformed, label=Y_train)
dtest = xgb.DMatrix(X_test_transformed, label=Y_test)

# Initialize the model with GPU support
model = xgb.train(
    params = {
    'objective': 'multi:softmax',  # Assuming multi-class classification
    'num_class': len(label_encoder.classes_),  # Number of unique classes
    'tree_method': 'hist',          # Use histogram-based algorithm
    'device': 'cuda',               # Set device to GPU (CUDA)
    'max_depth': 8,
    'learning_rate': 0.1,
    'random_state': 100},
    dtrain=dtrain,
    num_boost_round=200
)

# Make predictions using GPU-based DMatrix
pred1 = model.predict(dtrain)
pred2 = model.predict(dtest)

# Convert predictions back to integer labels
pred1 = pred1.astype(int)
pred2 = pred2.astype(int)

# Metrics function remains the same
def metrics(actual, predicted):
    acc = accuracy_score(actual, predicted) * 100
    prec = precision_score(actual, predicted, average="macro") * 100
    recall = recall_score(actual, predicted, average="macro") * 100
    macro_f1 = f1_score(actual, predicted, average="macro") * 100

    return pd.DataFrame({
        "Metrics": ["Accuracy", "Precision", "Recall", "Macro F1"],
        "Values": [acc, prec, recall, macro_f1]
    }).set_index("Metrics")

# Evaluate metrics
train_metrics = metrics(Y_train, pred1)
test_metrics = metrics(Y_test, pred2)

print(pd.DataFrame({
    "Training": train_metrics["Values"],
    "Testing": test_metrics["Values"]
}).reset_index())




# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 1, 5]
}



# Initialize the model with GPU support
base_model = xgb.XGBClassifier(
    random_state=100,
    n_jobs=-1,
    tree_method='hist',  # Enables GPU
    device = "cuda"  # Use the first GPU. Adjust if using multiple GPUs.
)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=5,  # 5-fold cross-validation
    verbose=3,
    n_jobs=-1
)

# Perform the grid search using the transformed training data
grid_search.fit(X_train_transformed, Y_train)

# Retrieve the best model and parameters
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Train the best model on the entire training set
best_model.fit(dtrain, Y_train)

# Make predictions
pred1 = best_model.predict(X_train_transformed)
pred2 = best_model.predict(X_test_transformed)

# Evaluate metrics
train_metrics = metrics(Y_train, pred1)
test_metrics = metrics(Y_test, pred2)

# Display metrics
print(pd.DataFrame({
    "Training": train_metrics["Values"],
    "Testing": test_metrics["Values"]
}).reset_index())
