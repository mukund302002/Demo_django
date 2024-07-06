import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# load dataset
dataset = pd.read_csv("C:\\Users\\Mukund Agarwalla\\Downloads\\archive\\train.csv", encoding='latin-1')
dataset = dataset.rename(columns=lambda x: x.strip().lower())

# cleaning missing values
dataset = dataset[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',  'survived']]
dataset['sex'] = dataset['sex'].map({'male': 0, 'female': 1})
dataset['age'] = pd.to_numeric(dataset['age'], errors='coerce')
dataset['age'] = dataset['age'].fillna(np.mean(dataset['age']))

# fill missing values for fare and embarked
dataset['fare'] = dataset['fare'].fillna(np.mean(dataset['fare']))



# Check for any remaining NaN values and fill them
dataset.fillna(dataset.mean(), inplace=True)

X = dataset.drop(['survived'], axis=1)
y = dataset['survived']

# Ensure all column names are strings
X.columns = X.columns.astype(str)

# scaling features 
sc = MinMaxScaler(feature_range=(0, 1))
X_scaled = sc.fit_transform(X)

# model fit
log_model = LogisticRegression(C=1)
log_model.fit(X_scaled, y)

# saving model as a pickle
pickle.dump(log_model, open("titanic_survival_ml_model.sav", "wb"))
pickle.dump(sc, open("scaler.sav", "wb"))
