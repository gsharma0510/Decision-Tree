import pandas as pd
from ydata_profiling import ProfileReport as pr
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
from sklearn.tree import _tree

df = pd.read_excel("D:\MS BAnDS\Spring 2024\Predictive Analytics\Week 4\KRITIK Exercise 2\MCR_HomeHlth.xlsx")

print(df.head(5))

#description of data
print(df.describe())


# Generate a profile report
profile = pr(df, title='Medical Data', explorative=True)


# Display the profile report
profile.to_notebook_iframe()


# Step 1: Read the dataset from Excel file
data = pd.read_excel("D:\MS BAnDS\Spring 2024\Predictive Analytics\Week 4\KRITIK Exercise 2\MCR_HomeHlth.xlsx")

# Step 2: Prepare data for modeling
X = data.drop(columns=['profit_b', 'State', 'Agency_Name'])  # Features
y = data['profit_b']  # Target variable

# Step 3: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

# Step 4: Build Decision Tree model
model = DecisionTreeClassifier(max_depth=8, min_samples_leaf=8, min_samples_split=8)
model.fit(X_train, y_train)

# Step 5: Visualize the Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=X.columns, class_names=[str(i) for i in model.classes_], filled=True)
plt.show()

# Step 6: Evaluate the model
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)


# Step 7: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 8: Print classification report
print(classification_report(y_test, y_pred))


#Predict probabilities for the test set
y_prob = model.predict_proba(X_test)[:, 1]

# Sort instances by predicted probabilities
sorted_indices = np.argsort(y_prob)
y_test_sorted = y_test.iloc[sorted_indices]

# Divide the dataset into deciles
n = len(y_test_sorted)
deciles = np.array_split(y_test_sorted, 10)

# Calculate proportion of positive instances in each decile
positive_proportions = [np.mean(decile) for decile in deciles]

# Calculate overall proportion of positive instances
overall_positive_proportion = np.mean(y_test)

# Calculate lift and gain for each decile
lift = [positive_proportion / overall_positive_proportion for positive_proportion in positive_proportions]
gain = np.cumsum(positive_proportions) / np.sum(y_test)

# Normalize gain values
normalized_gain = gain / gain[-1]

# Print lift and normalized gain for each decile
for i, (lift_value, normalized_gain_value) in enumerate(zip(lift, normalized_gain), 1):
    print(f"Decile {i}: Lift = {lift_value}, Normalized Gain = {normalized_gain_value:.4f}")


# Step 14: Understand decision rules
tree_rules = []

# Function to recursively traverse the tree
def traverse_tree(tree, node, rules, feature_names):
    if tree.feature[node] != _tree.TREE_UNDEFINED:
        name = feature_names[tree.feature[node]]
        threshold = tree.threshold[node]
        left_node = tree.children_left[node]
        right_node = tree.children_right[node]
        traverse_tree(tree, left_node, rules + [f"{name} <= {threshold:.2f}"], feature_names)
        traverse_tree(tree, right_node, rules + [f"{name} > {threshold:.2f}"], feature_names)
    else:
        target = np.argmax(tree.value[node])
        rules.append(f"Class: {target}")
        tree_rules.append(" ".join(rules))

traverse_tree(model.tree_, 0, [], X.columns)
for rule in tree_rules:
    print(rule)


# In[27]:


# Get feature importance
feature_importances = model.feature_importances_

# Create a DataFrame to display feature importance
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display the top features
print(importance_df.head())


#Find important features
feature_importance = model.feature_importances_
feature_names = X.columns

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plotting feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.gca().invert_yaxis()  # Invert y-axis to display most important features on top
plt.show()

print(feature_importance_df)




