import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, classification_report,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("./blood_samples.csv")

df.Disease = df.Disease.map({
    "Anemia":0,
    "Healthy":1,
    "Diabetes":2,
    "Thalasse":3,
    "Thromboc":4,
    "Heart Di":5
})

X = df.drop(["Disease"], axis=1)
y = df.Disease

xtrain, xtest, ytrain, ytest = train_test_split(X,y,test_size = 0.2, random_state=0)

model = LogisticRegression(max_iter = 60) #Mas de 60 iteraciones no cambia el accuracy
model.fit(xtrain, ytrain)

model.score(xtest, ytest)

predictions = model.predict(xtest)
cm = confusion_matrix(ytest, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["Anemia","Healthy","Diabetes","Thalasse","Thromboc","Heart Di"])
disp.plot()
plt.show()

print("Reporte de clasificación:\n", classification_report(ytest, predictions))