import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("./blood_samples.csv")
df_test = pd.read_csv("./blood_samples_test.csv")

df.Disease = df.Disease.map({
    "Anemia":0,
    "Healthy":1,
    "Diabetes":2,
    "Thalasse":3,
    "Thromboc":4
})

X = df.drop(["Disease"], axis=1)
y = df.Disease

xtrain, xtest, ytrain, ytest = train_test_split(X,y,test_size=0.3, random_state=0)

model = LogisticRegression(max_iter=1000) 
model.fit(xtrain, ytrain)

model.score(xtest, ytest)

preds = model.predict(xtest)
cm = confusion_matrix(ytest, preds)
cmdf = pd.DataFrame(cm, index=["Anemia","Healthy","Diabetes","Thalasse","Thromboc"], columns=["Anemia","Healthy","Diabetes","Thalasse","Thromboc"])
plt.figure(figsize=(8,6))
sns.heatmap(cmdf)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()