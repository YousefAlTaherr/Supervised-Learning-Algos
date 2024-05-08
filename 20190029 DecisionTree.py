import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

train = pd.read_csv('Train1.csv')

# Fill missing values with mean
mean_val = train.iloc[:, 2:-1].mean()
train.iloc[:, 2:-1] = train.iloc[:, 2:-1].fillna(mean_val)

# Prepare features and target
x = train.drop(columns=['INCIDENT_ID', 'MALICIOUS_OFFENSE', 'DATE'], axis=1)
y = train['MALICIOUS_OFFENSE']

# Split train into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)

# Create and train the Decision Tree model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
train_accuracy = model.score(x_train, y_train)
test_accuracy = model.score(x_test, y_test)

print("Algorithm: Decision Tree")
print("Hyperparameters: max_depth=3")
print(f"Train Set Accuracy: {train_accuracy:.3f}")
print(f"Test Set Accuracy: {test_accuracy:.3f}")
