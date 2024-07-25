import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
file_path = 'Brain_Tumor.csv'
df = pd.read_csv(file_path)

# Data preprocessing
df.drop("Image", inplace=True, axis=1)
df_clean = df.drop_duplicates().dropna()

# Check class distribution
print("Class distribution in the dataset:")
print(df_clean['Class'].value_counts())

# Feature and target separation
X = df_clean.drop('Class', axis=1)
y = df_clean['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA for Decision Tree
pca_dt = PCA(n_components=8)
X_train_pca_dt = pca_dt.fit_transform(X_train_scaled)
X_test_pca_dt = pca_dt.transform(X_test_scaled)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_pca_dt, y_train)

# Predictions for Decision Tree
y_train_pred_dt = dt_classifier.predict(X_train_pca_dt)
y_test_pred_dt = dt_classifier.predict(X_test_pca_dt)
accuracy_dt_train = accuracy_score(y_train, y_train_pred_dt)
accuracy_dt_test = accuracy_score(y_test, y_test_pred_dt)

print("\nDecision Tree Accuracy:")
print(f"Training Accuracy: {accuracy_dt_train}")
print(f"Test Accuracy: {accuracy_dt_test}")

# PCA for Naive Bayes
pca_nb = PCA(n_components=3)
X_train_pca_nb = pca_nb.fit_transform(X_train_scaled)
X_test_pca_nb = pca_nb.transform(X_test_scaled)

# Naive Bayes Classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train_pca_nb, y_train)

# Predictions for Naive Bayes
y_train_pred_nb = nb_classifier.predict(X_train_pca_nb)
y_test_pred_nb = nb_classifier.predict(X_test_pca_nb)
accuracy_nb_train = accuracy_score(y_train, y_train_pred_nb)
accuracy_nb_test = accuracy_score(y_test, y_test_pred_nb)

print("\nNaive Bayes Accuracy:")
print(f"Training Accuracy: {accuracy_nb_train}")
print(f"Test Accuracy: {accuracy_nb_test}")

# Determine which classifier to use based on accuracy
if accuracy_dt_test > accuracy_nb_test:
    final_model = dt_classifier
    final_pca = pca_dt
    final_accuracy = accuracy_dt_test
    print("Decision Tree has higher accuracy. Using Decision Tree.")
else:
    final_model = nb_classifier
    final_pca = pca_nb
    final_accuracy = accuracy_nb_test
    print("Naive Bayes has higher accuracy. Using Naive Bayes.")

# Save the selected model and PCA
model_data = {
    'dt_model': final_model,
    'nb_model': nb_classifier,  # Saving NB model as well for consistency
    'scaler': scaler,
    'final_pca_dt': final_pca,
    'final_pca_nb': pca_nb,  # Saving NB PCA as well for consistency
    'dt_accuracy': accuracy_dt_test,
    'nb_accuracy': accuracy_nb_test
}

with open('model_data.pkl', 'wb') as model_file:
    pickle.dump(model_data, model_file)

print("Model and PCA data have been saved as model_data.pkl")
