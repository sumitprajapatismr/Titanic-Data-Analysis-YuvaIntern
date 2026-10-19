import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chi2_contingency

# Load Dataset
df = pd.read_csv("train.csv")

# -----------------------------
# Data Cleaning
# -----------------------------
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns=["Cabin"])
df = df.drop_duplicates()

# ====================================================
# Hypothesis 1
# H0: Mean age of male and female passengers is equal.
# H1: Mean age of male and female passengers is different.
# ====================================================

male_age = df[df["Sex"] == "male"]["Age"]
female_age = df[df["Sex"] == "female"]["Age"]

t_stat, p_value = ttest_ind(male_age, female_age)

print("========== T-TEST ==========")
print(f"T Statistic : {t_stat:.4f}")
print(f"P-Value     : {p_value:.6f}")

if p_value < 0.05:
    print("Result : Reject Null Hypothesis")
else:
    print("Result : Fail to Reject Null Hypothesis")

# ====================================================
# Hypothesis 2
# H0: Survival is independent of Gender.
# H1: Survival depends on Gender.
# ====================================================

table = pd.crosstab(df["Sex"], df["Survived"])

chi2, p, dof, expected = chi2_contingency(table)

print("\n========== CHI-SQUARE TEST ==========")
print(f"Chi2 Statistic : {chi2:.4f}")
print(f"P-Value        : {p:.6f}")

if p < 0.05:
    print("Result : Reject Null Hypothesis")
else:
    print("Result : Fail to Reject Null Hypothesis")

# ====================================================
# Visualization 1
# ====================================================

plt.figure(figsize=(6,4))
plt.hist(df["Age"], bins=20, edgecolor="black")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.savefig("week3_age_histogram.png")
plt.close()

# ====================================================
# Visualization 2
# ====================================================

plt.figure(figsize=(6,4))
df.groupby("Sex")["Survived"].mean().plot(kind="bar")
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.savefig("week3_gender_survival.png")
plt.close()

# ====================================================
# Visualization 3
# ====================================================

plt.figure(figsize=(6,4))
df["Pclass"].value_counts().sort_index().plot(kind="bar")
plt.title("Passenger Class Distribution")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.savefig("week3_passenger_class.png")
plt.close()

print("\nWeek 3 Project Completed Successfully!")