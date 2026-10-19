import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("train.csv")

# Dataset information
print(df.head())
print(df.info())

# Data Cleaning
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
# Visualization 1
df["Survived"].value_counts().plot(kind="bar", title="Survival Count")
plt.savefig("survival.png")
plt.close()

# Visualization 2
df["Age"].plot(kind="hist", bins=20, title="Age Distribution")
plt.savefig("age.png")
plt.close()

# Visualization 3
df.groupby("Sex")["Survived"].mean().plot(kind="bar", title="Survival Rate by Gender")
plt.savefig("gender.png")
plt.close()

print("Project Completed Successfully!")