import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from collections import Counter
from collections import Counter
# Load dataset
file_path = 'Brain_Tumor.csv'
df = pd.read_csv(file_path)

# Define the feature columns
feature_columns = ['Mean', 'Variance', 'Standard Deviation', 'Entropy', 'Skewness', 'Kurtosis',
                   'Contrast', 'Energy', 'ASM', 'Homogeneity', 'Dissimilarity', 'Correlation', 'Coarseness']

# Calculate the range of values for each feature for tumors (Class = 1)
tumor_data = df[df['Class'] == 1]
tumor_range = tumor_data[feature_columns].agg(['min', 'max'])

# Calculate the range of values for each feature for non-tumors (Class = 0)
non_tumor_data = df[df['Class'] == 0]
non_tumor_range = non_tumor_data[feature_columns].agg(['min', 'max'])

print("Tumor Data Range:")
print(tumor_range)

print("\nNon-Tumor Data Range:")
print(non_tumor_range)

# Data preprocessing
df = df.rename(columns={'Coarseness': 'Level'})
df.drop("Image", inplace=True, axis=1)

# Handle duplicates and missing values
df.duplicated().sum()
df.isna().sum()
dfClean = df.drop_duplicates(inplace=False)
dfClean = dfClean.dropna()

# Function to remove outliers
def outlierDEL(column):
    q1 = dfClean[column].quantile(0.25)
    q3 = dfClean[column].quantile(0.75)
    IQR = q3-q1
    lowBound = q1-1.5*IQR
    upBound = q3+1.5*IQR

    dfClean[column] = dfClean[column].apply(lambda x: lowBound if x<lowBound else (upBound if x> upBound else x))
# Remove outliers for specified columns
def outlierDEL(column):

    print("Processing outliers for column:", column)


outliersCOL = [ 'Class', 'Mean', 'Variance', 'Standard Deviation',
         'Entropy', 'Kurtosis', 'Contrast', 'Energy', 'ASM',
         'Homogeneity', 'Dissimilarity', 'Correlation', 'Skewness', 'Level']

for col in outliersCOL:
    outlierDEL(col)

# Feature and target separation
X = dfClean.drop('Class', axis = 1)
y = dfClean['Class']

y.head()

print("Jumlah sampel sebelum oversampling:")
print(Counter(y))
# SMOTE for handling imbalanced dataset
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
n_components = 10
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Decision Tree Classifier with GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy'],
}

dt_classifier = DecisionTreeClassifier(random_state=42)

grid_search_cv = GridSearchCV(dt_classifier, param_grid=param_grid, cv=5)
grid_search_cv.fit(X_train_pca, y_train)
best_dt_classifier = grid_search_cv.best_estimator_

cv_scores = cross_val_score(best_dt_classifier, X_train_pca, y_train, cv=5)

print("Cross Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())
print("Std CV Score:", cv_scores.std())

best_dt_classifier.fit(X_train_pca, y_train)

y_train_pred = best_dt_classifier.predict(X_train_pca)
y_test_pred = best_dt_classifier.predict(X_test_pca)

print("Classification Report for Training Data:")
print(classification_report(y_train, y_train_pred))
print("Classification Report for Test Data:")
print(classification_report(y_test, y_test_pred))

train_accuracy = best_dt_classifier.score(X_train_pca, y_train)
test_accuracy = best_dt_classifier.score(X_test_pca, y_test)
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

with open('dt.pkl', 'wb') as file:
    pickle.dump(best_dt_classifier, file)

