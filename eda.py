import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
train_data=pd.read_csv('data/GUIDE_Train.csv')
train_df=train_data.drop(['MitreTechniques','ActionGrouped','ActionGranular','EmailClusterId','ThreatFamily','ResourceType','Roles','AntispamDirection','SuspicionLevel','LastVerdict'],
  axis=1)

plt.figure(figsize=(8, 5))
sns.countplot(data=train_df, x='IncidentGrade')
plt.title("Distribution of IncidentGrade (Target Variable)")
plt.xlabel("Incident Grade")
plt.ylabel("Count")
plt.show()


categorical_features = ['Category', 'EntityType', 'EvidenceRole']

# Plot distributions
fig, axs = plt.subplots(1, len(categorical_features), figsize=(20, 5))
for i, feature in enumerate(categorical_features):
    sns.countplot(data=train_df, x=feature, ax=axs[i])
    axs[i].set_title(f"Distribution of {feature}")
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel("Count")
    axs[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()



numerical_features = ['IncidentId', 'AlertId', 'OrgId']  # Update with relevant numerical features

# Plot histograms
fig, axs = plt.subplots(len(numerical_features), 1, figsize=(10, len(numerical_features) * 4))
for i, feature in enumerate(numerical_features):
    sns.histplot(data=train_df, x=feature, kde=True, ax=axs[i])
    axs[i].set_title(f"Distribution of {feature}")
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
