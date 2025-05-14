import pandas as pd


# Carga dataset
dataset = pd.read_csv('loan_data.csv') 
X = dataset.iloc[:,[0,3,4,6,8,9,10,11]].values # características: todas las variables continuas
y = dataset['loan_status'].values # objetivo: (préstamo aprobado?) 

# Divide dataset (train 75%, test 25%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Entrena modelo
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Clasifica el conjunto de prueba
y_pred = classifier.predict(X_test)


# Evalúa modelo
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(cm)
print("Precisión del modelo:")
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
