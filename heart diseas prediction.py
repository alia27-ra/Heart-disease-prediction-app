import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load dataset
df = pd.read_csv("heart_disease_uci.csv")

# Step 2: Drop unnecessary columns (if any)
if 'id' in df.columns:
    df = df.drop('id', axis=1)
if 'dataset' in df.columns:
    df = df.drop('dataset', axis=1)

# Step 3: Encode categorical columns if present
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

# Step 4: Convert 'num' column to binary (0 = no disease, 1 = has disease)
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# Step 5: Split features and target
X = df.drop('num', axis=1)
y = df['num']

# Step 6: Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Prediction
y_pred = model.predict(X_test)

# Step 10: Evaluation
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%\n")
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
import pickle

# Save trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ model.pkl and scaler.pkl saved successfully.")

