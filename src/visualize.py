import pandas as pd
import matplotlib.pyplot as plt

# Load predictions
results = pd.read_csv("data/spam_predictions.csv")

# Count spam vs ham
counts = results['predicted_label'].value_counts()

# Plot bar chart
counts.plot(kind='bar', color=['skyblue','salmon'])
plt.title("Spam vs Ham Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.savefig("data/spam_distribution.png")
plt.show()
