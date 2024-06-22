# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_iris

# Loading the Iris dataset
iris = load_iris()

# Creating a DataFrame
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                  columns= iris['feature_names'] + ['target'])

# Data preprocessing
# No missing values to handle in this dataset

# Splitting data into features and target
X = df.drop('target', axis=1)
y = df['target']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Exploratory Data Analysis (EDA)
# Pairplot to visualize relationships between variables
sns.pairplot(df, hue='target')
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

# Modeling with Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluation
y_pred = rf_classifier.predict(X_test)

# Confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Interpretation of Results
# Feature importances
feature_importances = pd.DataFrame(rf_classifier.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print("\nFeature Importances:\n", feature_importances)

# Example prediction
example_pred = rf_classifier.predict([[5.1, 3.5, 1.4, 0.2]])

# Convert example_pred to integer index
predicted_class_index = int(example_pred[0])

# Print the predicted class name
print("\nExample Prediction:", iris['target_names'][predicted_class_index])

