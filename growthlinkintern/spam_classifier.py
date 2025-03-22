# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset & Detect Column Names
df = pd.read_csv("mail_data.csv", encoding="latin-1")

# Print columns to check correct names
print("Available Columns in CSV:", df.columns)

# Auto-detect spam label & message columns
for col in df.columns:
    if "label" in col.lower() or "category" in col.lower() or "v1" in col.lower():
        label_col = col
    if "message" in col.lower() or "text" in col.lower() or "v2" in col.lower():
        message_col = col

# Rename columns for consistency
df = df[[label_col, message_col]]
df.columns = ["Label", "Message"]

# Convert labels to binary (spam = 1, ham = 0)
df["Label"] = df["Label"].map({"ham": 0, "spam": 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["Message"], df["Label"], test_size=0.2, random_state=42)

# Create Model Pipeline (TF-IDF + Naïve Bayes)
model = make_pipeline(TfidfVectorizer(stop_words="english"), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Real-time Prediction
def predict_spam(message):
    prediction = model.predict([message])[0]
    return "SPAM" if prediction == 1 else "HAM"

# Interactive Console Input
while True:
    msg = input("\nEnter a message to check (or type 'exit' to quit): ")
    if msg.lower() == "exit":
        break
    print(f"Prediction: {predict_spam(msg)}")