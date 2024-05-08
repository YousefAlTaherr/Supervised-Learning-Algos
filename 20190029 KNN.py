import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
data = pd.read_csv('Train1.csv')
mean_val=data.iloc[:, 2:-1].mean()
data.iloc[:, 2:-1]=data.iloc[:, 2:-1].fillna(mean_val)
x = data.drop(columns=['INCIDENT_ID', 'MALICIOUS_OFFENSE','DATE'], axis=1)
y=data['MALICIOUS_OFFENSE']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=4)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
train_accuracy = model.score(x_train, y_train)
test_accuracy = model.score(x_test, y_test)
algorithm = 'KNN'
hyperparameters = 'n_neighbors=3'
print("Algorithm: KNN")
print("Hyperparameters: n_neighbors =3")
print(f"Accuracy train: {train_accuracy:.3f}")
print(f"Accuracy test: {test_accuracy:.3f}")
#i found that model.score gets the model specific score instead of accuracy_score being generalized