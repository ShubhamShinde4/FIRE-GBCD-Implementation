import numpy as np
from sklearn.ensemble import RandomForestClassifier

class FIRE:
    def __init__(self, max_depth=3, n_estimators=100):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth)
        self.rules = []
        self.feature_names = None

    def fit(self, X, y):
        self.feature_names = X.columns  # Store feature names
        self.model.fit(X, y)  # Train Random Forest model
        self.extract_rules()  # Extract decision rules from the trained trees

    def extract_rules(self):
        for tree in self.model.estimators_:
            self.rules.extend(self.get_rules_from_tree(tree))

    def get_rules_from_tree(self, tree):
        rules = []
        tree_ = tree.tree_

        def recurse(node, conditions):
            if tree_.feature[node] != -2:  # Not a leaf node
                feature = self.feature_names[tree_.feature[node]]
                threshold = tree_.threshold[node]
                recurse(tree_.children_left[node], conditions + [f"{feature} <= {threshold:.2f}"])
                recurse(tree_.children_right[node], conditions + [f"{feature} > {threshold:.2f}"])
            else:
                rules.append(" AND ".join(conditions))  # Leaf node: Append the condition as a rule

        recurse(0, [])
        return rules

    def print_rules(self):
        for i, rule in enumerate(self.rules[:10]):  # Print top 10 rules
            print(f"Rule {i + 1}: {rule}")

    def predict(self, X):
        return self.model.predict(X)
