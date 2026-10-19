import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("train.csv")

# Data Cleaning
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns=["Cabin"])
df = df.drop_duplicates()

# 1. Survival Count
plt.figure(figsize=(6,4))
df["Survived"].value_counts().plot(kind="bar")
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Passengers")
plt.savefig("visual1_survival_count.png")
plt.close()

# 2. Age Distribution
plt.figure(figsize=(6,4))
plt.hist(df["Age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("visual2_age_distribution.png")
plt.close()

# 3. Survival by Gender
plt.figure(figsize=(6,4))
df.groupby("Sex")["Survived"].mean().plot(kind="bar")
plt.title("Survival Rate by Gender")
plt.ylabel("Survival Rate")
plt.savefig("visual3_gender_survival.png")
plt.close()

# 4. Passenger Class Distribution
plt.figure(figsize=(6,4))
df["Pclass"].value_counts().sort_index().plot(kind="bar")
plt.title("Passenger Class Distribution")
plt.xlabel("Class")
plt.ylabel("Passengers")
plt.savefig("visual4_passenger_class.png")
plt.close()

# 5. Average Fare by Class
plt.figure(figsize=(6,4))
df.groupby("Pclass")["Fare"].mean().plot(kind="bar")
plt.title("Average Fare by Passenger Class")
plt.xlabel("Class")
plt.ylabel("Average Fare")
plt.savefig("visual5_fare_by_class.png")
plt.close()

print("Week 2 Project Completed Successfully!")