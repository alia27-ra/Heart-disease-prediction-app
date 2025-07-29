import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Step 1: Load the dataset
data = data = pd.read_csv("heart_disease_uci.csv")
  # Make sure this file exists in the same folder

# Step 2: Split into features and target
X = data.drop("target", axis=1)
y = data["target"]

# Step 3: Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Split for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Save the model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Training complete. model.pkl and scaler.pkl saved.")
