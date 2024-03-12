# Importando las bibliotecas necesarias
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

df_train = pd.read_csv('blood_samples.csv')
df_test = pd.read_csv('blood_samples_test.csv')

X_train = df_train.drop('Disease', axis=1)  # Características
y_train = df_train['Disease']  # Etiquetas

X_test = df_test.drop('Disease', axis=1)
y_test = df_test['Disease']

# Codificacion etiquetas
le = LabelEncoder()

# Quitar etiquetas de test que no estan en train para que no haya errores
all_labels = pd.concat([df_train['Disease'], df_test['Disease']])
le.fit(all_labels)

y_train = le.transform(y_train)
y_test = le.transform(y_test)

model = svm.SVC()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Reporte de clasificación:\n", classification_report(y_test, predictions))
print("Matriz de confusión:\n", confusion_matrix(y_test, predictions))
