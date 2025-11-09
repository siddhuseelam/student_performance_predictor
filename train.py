# train.py — Train Decision Tree model on Student Performance dataset

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv('data/StudentsPerformance.csv')

# -----------------------
# Target Creation
# -----------------------
df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
df['pass'] = (df['average_score'] >= 60).astype(int)
df.drop(['math score', 'reading score', 'writing score', 'average_score'], axis=1, inplace=True)

# -----------------------
# Encode Categorical Variables
# -----------------------
cat_cols = df.select_dtypes('object').columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# -----------------------
# Split Data
# -----------------------
X = df.drop('pass', axis=1)
y = df['pass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------
# Train Decision Tree
# -----------------------
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Training Accuracy:", accuracy_score(y_train, model.predict(X_train)))
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------
# Cross-Validation
# -----------------------
scores = cross_val_score(model, X, y, cv=5)
print("\nCross-validation accuracy:", scores.mean())

# -----------------------
# Simplified Tree (Pruning)
# -----------------------
pruned = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
pruned.fit(X_train, y_train)
print("\nPruned Tree Accuracy:", accuracy_score(y_test, pruned.predict(X_test)))

# -----------------------
# Rule Extraction
# -----------------------
print("\nExtracted Rules:\n")
print(export_text(pruned, feature_names=list(X.columns)))

# -----------------------
# Bias Check (Gender / Race)
# -----------------------
X_reduced = X.drop(['gender', 'race/ethnicity'], axis=1)
reduced_score = cross_val_score(DecisionTreeClassifier(random_state=42), X_reduced, y, cv=5)
print("\nAccuracy without gender & race:", reduced_score.mean())

# -----------------------
# Association Rules
# -----------------------
# Use a fresh copy of the raw CSV and one-hot encode categorical cols so apriori only sees 0/1
raw = pd.read_csv('data/StudentsPerformance.csv')
raw['average_score'] = raw[['math score', 'reading score', 'writing score']].mean(axis=1)
raw['pass'] = (raw['average_score'] >= 60).astype(int)
raw = raw.drop(['math score', 'reading score', 'writing score', 'average_score'], axis=1)

# One-hot encode categorical features (this guarantees 0/1 columns for apriori)
encoded = pd.get_dummies(raw.drop('pass', axis=1))
encoded = encoded.astype(int)

try:
    frequent_items = apriori(encoded, min_support=0.3, use_colnames=True)
    rules = association_rules(frequent_items, metric="lift", min_threshold=1.0)
    print("\nTop Association Rules:\n", rules.sort_values('lift', ascending=False).head())
except ValueError as e:
    print("\nAssociation rules generation failed:", e)

# -----------------------
# Clustering (KMeans)
# -----------------------
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
df['Cluster'] = clusters
sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=clusters, palette='Set2')
plt.title("Student Clusters")
plt.show()

# -----------------------
# Save Model
# -----------------------
pickle.dump(pruned, open('models/decision_tree.pkl', 'wb'))
print("\nModel saved successfully to models/decision_tree.pkl")
