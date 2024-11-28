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
preprocessor=ColumnTransformer(transformers=[('num',StandardScaler(),numeric_col),('cat',OneHotEncoder(),categorical_col)])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.2, stratify=Y_encoded, random_state=69)

model=xgb.XGBClassifier(n_estimators=200,learning_rate=0.1,random_state=100,n_jobs=-1,max_depth=8)
model.fit(X_train,Y_train)

def metrics(actual, predicted):
    acc = accuracy_score(actual, predicted) * 100
    prec = precision_score(actual, predicted, average="macro") * 100
    recall = recall_score(actual, predicted, average="macro") * 100
    macro_f1 = f1_score(actual, predicted, average="macro") * 100

    return pd.DataFrame({
        "Metrics": ["Accuracy", "Precision", "Recall", "Macro F1"],
        "Values": [acc, prec, recall, macro_f1]
    }).set_index("Metrics")

pred1 =model.predict(X_train)
pred2 =model.predict(X_test)

train_metrics=metrics(Y_train,pred1)
test_metrics=metrics(Y_test,pred2)

pd.DataFrame({
    "Training":train_metrics["Values"],
    "Testing":test_metrics["Values"]
}).reset_index()