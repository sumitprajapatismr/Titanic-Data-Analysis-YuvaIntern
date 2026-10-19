import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("train.csv")

# Display First 5 Rows
print(df.head())

# Dataset Information
print(df.info())

# -----------------------------
# Data Cleaning
# -----------------------------

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Remove Cabin column
df = df.drop(columns=["Cabin"])

# Remove duplicate rows
df = df.drop_duplicates()

# -----------------------------
# Visualization 1
# Survival Count
# -----------------------------
plt.figure(figsize=(6,4))
df["Survived"].value_counts().plot(kind="bar", color=["red", "green"])
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Number of Passengers")
plt.savefig("survival.png")
plt.close()

# -----------------------------
# Visualization 2
# Age Distribution
# -----------------------------
plt.figure(figsize=(6,4))
plt.hist(df["Age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.savefig("age.png")
plt.close()

# -----------------------------
# Visualization 3
# Survival Rate by Gender
# -----------------------------
plt.figure(figsize=(6,4))
df.groupby("Sex")["Survived"].mean().plot(kind="bar")
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.savefig("gender.png")
plt.close()

print("\nWeek 1 Project Completed Successfully!")