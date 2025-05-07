import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('dataset.csv')

# Basic preprocessing
df.dropna(inplace=True)
X = df.drop('Accident_Severity', axis=1)
y = df['Accident_Severity']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
