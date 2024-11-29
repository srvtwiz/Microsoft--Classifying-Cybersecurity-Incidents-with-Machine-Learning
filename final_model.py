#import libaries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV,train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import precision_score,f1_score,recall_score,classification_report,accuracy_score
import xgboost as xgb
import pickle

#read the csv file
train_data=pd.read_csv("C:/Users/srvik/Desktop/github/microsoft data/GUIDE_Train.csv", low_memory=False)
test_data=pd.read_csv("C:/Users/srvik/Desktop/github/microsoft data/GUIDE_Test.csv", low_memory=False)

print(train_data)
print(test_data)

#checking data for null values
print(train_data.isnull().sum())
print(test_data.isnull().sum())

#drop null value cloumn with more than 80%data is null
train_data=train_data.drop(['MitreTechniques','ActionGrouped','ActionGranular','EmailClusterId','ThreatFamily','ResourceType','Roles',
                'AntispamDirection','SuspicionLevel','LastVerdict'],axis=1)
test_data=test_data.drop(['MitreTechniques','ActionGrouped','ActionGranular','EmailClusterId','ThreatFamily','ResourceType','Roles',
                'AntispamDirection','SuspicionLevel','LastVerdict'],axis=1)

#check correlation with heatmap
numeric_col= train_data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_col.corr()
print(correlation_matrix)
plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap for train')
plt.show()

threshold = 0.6
high_corr = correlation_matrix[correlation_matrix > threshold]
print(high_corr)

numeric_col= test_data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_col.corr()
print(correlation_matrix)
plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap for test')
plt.show()

threshold = 0.6
high_corr = correlation_matrix[correlation_matrix > threshold]
print(high_corr)

#dropping 60% correlated column 
train_data=train_data.drop(['ApplicationName','FileName','FolderPath','State','City','OSVersion','OSFamily','DeviceName',
                'RegistryValueData','RegistryKey','AccountName','AccountUpn','AccountSid','RegistryValueData','RegistryKey'],axis=1)
test_data=test_data.drop(['ApplicationName','FileName','FolderPath','State','City','OSVersion','OSFamily','DeviceName',
                'RegistryValueData','RegistryKey','AccountName','AccountUpn','AccountSid','RegistryValueData','RegistryKey'],axis=1)

#feature engineeering
train_data['Timestamp'] = pd.to_datetime(train_data['Timestamp'], errors='coerce')
test_data['Timestamp'] = pd.to_datetime(test_data['Timestamp'], errors='coerce')

train_data['Timestamp'].head()
train_data["Day"]=train_data["Timestamp"].dt.day
train_data["Month"]=train_data["Timestamp"].dt.month
train_data["Year"]=train_data["Timestamp"].dt.year
train_data["Hour"]=train_data["Timestamp"].dt.hour

test_data['Timestamp'].head()
test_data["Day"]=test_data["Timestamp"].dt.day
test_data["Month"]=test_data["Timestamp"].dt.month
test_data["Year"]=test_data["Timestamp"].dt.year
test_data["Hour"]=test_data["Timestamp"].dt.hour

print(train_data.isnull().sum())
print(test_data.isnull().sum())
print(train_data.info())

train_data=train_data.dropna()

# Separate features and target from training data and test data
X_train = train_data.drop(columns=['IncidentGrade'])
y_train = train_data['IncidentGrade']
X_test = test_data.drop(columns=['IncidentGrade'])
y_test = test_data['IncidentGrade']

#encoding target
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.fit_transform(y_test)

# Select column names for numeric and categorical columns
train_numeric_col = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
train_categorical_col = X_train.select_dtypes(include='object').columns.tolist()

test_numeric_col = X_test.select_dtypes(include=['float64', 'int64']).columns.tolist()
test_categorical_col = X_test.select_dtypes(include='object').columns.tolist()

# Create transformations for numeric and categorical columns
train_preprocessor=ColumnTransformer(transformers=[('num',StandardScaler(),train_numeric_col),('cat',OneHotEncoder(handle_unknown='ignore'),train_categorical_col)])
test_preprocessor=ColumnTransformer(transformers=[('num',StandardScaler(),test_numeric_col),('cat',OneHotEncoder(handle_unknown='ignore'),test_categorical_col)])

# Transform the training and testing data
X_train_transformed = train_preprocessor.fit_transform(X_train)
X_test_transformed = train_preprocessor.transform(X_test)


# Convert the processed data to a DMatrix structure and enable GPU
dtrain = xgb.DMatrix(X_train_transformed, label=y_train_encoded)
dtest = xgb.DMatrix(X_test_transformed, label=y_test_encoded)

#training the model
model = xgb.train(
    params = {
    'objective': 'multi:softmax',  # Assuming multi-class classification
    'num_class': len(label_encoder.classes_),  # Number of unique classes
    'tree_method': 'hist',          # Use histogram-based algorithm
    'device': 'cuda',               # Set device to GPU (CUDA)
    'max_depth': 8,
    'learning_rate': 0.1,
    'random_state': 100
    ,'gamma' : 0
    ,'colsample_bytree':0.8
    ,'learning_rate':0.2
    ,'subsample':1.0
    ,'max_depth':10
    ,'random_state':100}
    ,
    dtrain=dtrain,
    num_boost_round=200
)

# Make predictions using GPU-based DMatrix
pred1 = model.predict(dtrain)
pred2 = model.predict(dtest)

# Convert predictions back to integer labels
pred1 = pred1.astype(int)
pred2 = pred2.astype(int)

#function for metrics
def metrics(trained, tested):
    accuracy = accuracy_score(trained, tested) * 100
    precision = precision_score(trained, tested, average="macro") * 100
    recall = recall_score(trained, tested, average="macro") * 100
    f1 = f1_score(trained, tested, average="macro") * 100

    return pd.DataFrame({
        "Metrics": ["Accuracy", "Precision", "Recall", "Macro F1"],
        "Values": [accuracy, precision, recall, f1]
    }).set_index("Metrics")
    
    # Evaluate metrics
train_metrics = metrics(y_train_encoded, pred1)
test_metrics = metrics(y_test_encoded, pred2)

print(pd.DataFrame({
    "Training": train_metrics["Values"],
    "Testing": test_metrics["Values"]
}).reset_index())


# Save the trained model to a pickle file
with open("xgboost_model.pkl", "wb") as model_file:pickle.dump(model, model_file)



