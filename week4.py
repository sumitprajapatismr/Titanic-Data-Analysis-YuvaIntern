import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("train.csv")

# -------------------------
# Data Cleaning
# -------------------------
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns=["Cabin"])
df = df.drop_duplicates()

# -------------------------
# Convert Categorical Data
# -------------------------
df["Sex"] = df["Sex"].map({"male":0,"female":1})
df["Embarked"] = df["Embarked"].map({"S":0,"C":1,"Q":2})

# -------------------------
# Features and Target
# -------------------------
X = df[["Pclass","Sex","Age","Fare","SibSp","Parch","Embarked"]]
y = df["Survived"]

# -------------------------
# Train Test Split
# -------------------------
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# -------------------------
# Logistic Regression
# -------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

# -------------------------
# Prediction
# -------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print("Model Accuracy:",accuracy)

# -------------------------
# Confusion Matrix
# -------------------------
cm = confusion_matrix(y_test,y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix")
plt.savefig("week4_confusion_matrix.png")
plt.close()

# -------------------------
# ROC Curve
# -------------------------
y_prob = model.predict_proba(X_test)[:,1]

fpr,tpr,thresholds = roc_curve(y_test,y_prob)

plt.figure(figsize=(6,4))
plt.plot(fpr,tpr,label="ROC Curve")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

auc = roc_auc_score(y_test,y_prob)

plt.legend([f"AUC = {auc:.3f}"])

plt.savefig("week4_roc_curve.png")
plt.close()

print("ROC AUC:",auc)

print("\nWeek 4 Project Completed Successfully!")