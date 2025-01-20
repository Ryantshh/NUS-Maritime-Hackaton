# Import Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib  # For saving and loading the model
from collections import Counter

# Step 2: Load and Preprocess the Data
train_data = pd.read_csv('psc_severity_train.csv')  # Load the dataset

# Filter out invalid severity entries
valid_severities = ['High', 'Medium', 'Low']
train_data_clean = train_data[train_data['annotation_severity'].isin(valid_severities)]
train_data_clean.reset_index(drop=True, inplace=True)

# Step 3: Apply Majority Voting on 'deficiency_code'
def majority_vote(group):
    counts = Counter(group['annotation_severity'])
    severity_order = {'High': 2, 'Medium': 1, 'Low': 0}
    most_common = counts.most_common()
    most_common_sorted = sorted(most_common, key=lambda x: (-x[1], -severity_order[x[0]]))
    return most_common_sorted[0][0]

# Perform majority voting for each 'deficiency_code'
consensus_severity = train_data_clean.groupby('deficiency_code').apply(majority_vote).reset_index()
consensus_severity.columns = ['deficiency_code', 'majority_severity']

# Merge majority severity back to the dataset
train_data_clean = pd.merge(train_data_clean, consensus_severity, on='deficiency_code')

# Step 4: Apply TF-IDF Vectorization on 'def_text'
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(train_data_clean['def_text'])

# Step 5: Encode the Majority Severity for Training
label_encoder = LabelEncoder()
train_data_clean['majority_severity_encoded'] = label_encoder.fit_transform(train_data_clean['majority_severity'])

# Step 6: Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X, train_data_clean['majority_severity_encoded'])

# Step 7: Predict Severity Based on Text Analysis
predictions = model.predict(X)
predicted_severity = label_encoder.inverse_transform(predictions)
train_data_clean['predicted_severity'] = predicted_severity

# Step 8: Final Consensus Severity Using Both SME Votes and Text Analysis
final_consensus = train_data_clean.groupby('deficiency_code')['predicted_severity'].apply(lambda x: Counter(x).most_common(1)[0][0]).reset_index()
final_consensus.columns = ['deficiency_code', 'consensus_severity']

# Step 9: Replace 'annotation_severity' with Consensus Severity
train_data_final = pd.merge(train_data, final_consensus, on='deficiency_code')
train_data_final['annotation_severity'] = train_data_final['consensus_severity']
train_data_final.drop(columns=['consensus_severity'], inplace=True)

# Step 10: Save the Final Result
train_data_final.to_csv('final_consensus_full_data.csv', index=False)
print("Final dataset with updated consensus severity saved as 'final_consensus_full_data.csv'.")

# Load Datasets
train_data = pd.read_csv('final_consensus_full_data.csv')  # Training dataset
test_data = pd.read_csv('psc_severity_test.csv')           # Provided test dataset

# Data Cleaning: Keep Only Valid Severity Ratings
valid_severities = ['High', 'Medium', 'Low']
train_data = train_data[train_data['annotation_severity'].isin(valid_severities)]
train_data.reset_index(drop=True, inplace=True)

# Encode the Target Variable (annotation_severity)
label_encoder = LabelEncoder()
train_data['severity_encoded'] = label_encoder.fit_transform(train_data['annotation_severity'])

# Split the Data into Training (80%) and Validation (20%)
X_train, X_val, y_train, y_val = train_test_split(
    train_data['def_text'], 
    train_data['severity_encoded'], 
    test_size=0.2, 
    random_state=42
)

# TF-IDF Vectorization of def_text
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(test_data['def_text'])  # Vectorize test data

# Train Models

# Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# Gradient Boosting Classifier
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train_tfidf, y_train)

# Hybrid Ensemble (Voting Classifier)
hybrid_model = VotingClassifier(estimators=[
    ('lr', lr_model),
    ('gb', gb_model)
], voting='hard')

# Train the Hybrid Ensemble Model
hybrid_model.fit(X_train_tfidf, y_train)

# Validate the Model on the Validation Set
hybrid_val_preds = hybrid_model.predict(X_val_tfidf)
hybrid_val_report = classification_report(y_val, hybrid_val_preds, target_names=label_encoder.classes_)
hybrid_val_accuracy = accuracy_score(y_val, hybrid_val_preds)

# Print Validation Results
print("Hybrid Model Validation Classification Report:")
print(hybrid_val_report)
print(f"Hybrid Model Validation Accuracy: {hybrid_val_accuracy:.2f}")

# Save the Trained Model and Vectorizer
joblib.dump(hybrid_model, 'hybrid_severity_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Model and vectorizer have been saved successfully.")

# Predict on the Provided Test Dataset
hybrid_test_preds = hybrid_model.predict(X_test_tfidf)
test_data['predicted_severity'] = label_encoder.inverse_transform(hybrid_test_preds)

# FINAL STEP: Keep Only Required Columns and Save
final_test_data = test_data[['PscInspectionId', 'deficiency_code', 'predicted_severity']]
final_test_data.to_csv('predicted_severity_test.csv', index=False)

print("Final 'predicted_severity_test.csv' generated with required columns.")
