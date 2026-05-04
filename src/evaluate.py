import pandas as pd

# Load predictions file
results = pd.read_csv("data/spam_predictions.csv")

# Count spam vs ham
print("Counts:")
print(results['predicted_label'].value_counts())

# Percentages
print("\nPercentages:")
print(results['predicted_label'].value_counts(normalize=True) * 100)
