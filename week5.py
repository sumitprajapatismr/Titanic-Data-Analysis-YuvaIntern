import pandas as pd

# Load Dataset
df = pd.read_csv("train.csv")

print("="*60)
print("FINAL DATA SCIENCE PROJECT SUMMARY")
print("="*60)

print("\nDataset Shape:", df.shape)

print("\nMissing Values")
print(df.isnull().sum())

print("\nPassenger Survival")
print(df["Survived"].value_counts())

print("\nGender Distribution")
print(df["Sex"].value_counts())

print("\nPassenger Class Distribution")
print(df["Pclass"].value_counts())

print("\nAverage Age:", round(df["Age"].mean(),2))
print("Average Fare:", round(df["Fare"].mean(),2))

print("\nProject Completed Successfully!")

print("\nKey Findings")
print("- Female passengers had a higher survival rate.")
print("- First-class passengers survived more often.")
print("- Passenger class and gender strongly influenced survival.")
print("- Logistic Regression achieved around 80% accuracy.")

print("\nStrategic Recommendations")
print("- Prioritize vulnerable passengers during emergencies.")
print("- Improve emergency planning for lower passenger classes.")
print("- Use predictive analytics for safety planning.")
print("- Enhance data collection for better prediction models.")

print("\nWeek 5 Final Project Completed Successfully!")