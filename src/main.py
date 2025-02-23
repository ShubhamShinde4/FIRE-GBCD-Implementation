# src/main.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fire_model import FIRE
from gbcd_optimizer import GBCDOptimizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, r2_score

# Function to visualize rule importance
def plot_rule_importance(weights, title):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(weights)), weights)
    plt.xlabel("Rule Index")
    plt.ylabel("Optimized Weight")
    plt.title(title)
    plt.show()


# ----------------- Titanic Dataset -------------------
print("----- Titanic Dataset Results -----")

# Load cleaned Titanic dataset
titanic_data = pd.read_csv('../data/cleaned_titanic_train.csv')
X_titanic = titanic_data.drop('Survived', axis=1)  # Features
y_titanic = titanic_data['Survived']  # Target

# Train-Test Split for Titanic
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_titanic, y_titanic, test_size=0.2, random_state=42)

# Initialize and train FIRE model for Titanic
fire_titanic = FIRE(max_depth=3, n_estimators=100)
fire_titanic.fit(X_train_t, y_train_t)
fire_titanic.print_rules()

# Optimize Titanic Rules with GPU-Accelerated GBCD
rule_weights_titanic = np.random.rand(len(fire_titanic.rules))
optimizer = GBCDOptimizer(sparsity_penalty=0.2, fusion_penalty=0.05, learning_rate=0.01)
optimized_weights_titanic = optimizer.optimize(rule_weights_titanic)

# Predictions and Evaluation for Titanic
y_pred_titanic = fire_titanic.predict(X_test_t)
titanic_mse = mean_squared_error(y_test_t, y_pred_titanic)
titanic_accuracy = accuracy_score(y_test_t, y_pred_titanic)
titanic_f1 = f1_score(y_test_t, y_pred_titanic)

# Print Evaluation Results
print(f"Titanic Dataset - Test MSE: {titanic_mse}")
print(f"Titanic Dataset - Accuracy: {titanic_accuracy}")
print(f"Titanic Dataset - F1 Score: {titanic_f1}\n")

# Visualize Rule Importance for Titanic
plot_rule_importance(optimized_weights_titanic, "Titanic Dataset Rule Importance")


# ----------------- Wine Quality Dataset -------------------
print("----- Wine Quality Dataset Results -----")

# Load cleaned Wine Quality dataset
wine_data = pd.read_csv('../data/cleaned_wine_red.csv')
X_wine = wine_data.drop('quality', axis=1)  # Features
y_wine = wine_data['quality']  # Target

# Train-Test Split for Wine
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)

# Initialize and train FIRE model for Wine dataset
fire_wine = FIRE(max_depth=4, n_estimators=100)
fire_wine.fit(X_train_w, y_train_w)
fire_wine.print_rules()

# Optimize Wine Rules with GPU-Accelerated GBCD
rule_weights_wine = np.random.rand(len(fire_wine.rules))
optimized_weights_wine = optimizer.optimize(rule_weights_wine)

# Predictions and Evaluation for Wine
y_pred_wine = fire_wine.predict(X_test_w)
wine_mse = mean_squared_error(y_test_w, y_pred_wine)
wine_r2 = r2_score(y_test_w, y_pred_wine)

# Print Evaluation Results
print(f"Wine Dataset - Test MSE: {wine_mse}")
print(f"Wine Dataset - R² Score: {wine_r2}\n")

# Visualize Rule Importance for Wine Quality
plot_rule_importance(optimized_weights_wine, "Wine Quality Dataset Rule Importance")


# ----------------- Display Optimized Weights -------------------
print("Optimized Rule Weights for Titanic:\n", optimized_weights_titanic)
print("Optimized Rule Weights for Wine:\n", optimized_weights_wine)
