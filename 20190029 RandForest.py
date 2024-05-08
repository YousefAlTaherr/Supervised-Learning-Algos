import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
data = pd.read_csv('Train1.csv')
mean_val = data.iloc[:, 2:-1].mean()
data.iloc[:, 2:-1] = data.iloc[:, 2:-1].fillna(mean_val)
x = data.drop(columns=['INCIDENT_ID', 'MALICIOUS_OFFENSE', 'DATE'], axis=1)
y = data['MALICIOUS_OFFENSE']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)
model = RandomForestClassifier(n_estimators=100, random_state=41)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
train_accuracy = model.score(x_train, y_train)
test_accuracy = model.score(x_test, y_test)

print("Algorithm: Random Forest")
print("Hyperparameters: n_estimators=100")
print(f"Train Set Accuracy: {train_accuracy:.3f}")
print(f"Test Set Accuracy: {test_accuracy:.3f}")