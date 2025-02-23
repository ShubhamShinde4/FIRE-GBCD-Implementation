import pandas as pd

# Load Titanic Dataset
titanic_train = pd.read_csv('../data/train.csv')
titanic_test = pd.read_csv('../data/test.csv')

# Load Wine Quality Dataset
wine_red = pd.read_csv('../data/winequality-red.csv', sep=';')
wine_white = pd.read_csv('../data/winequality-white.csv', sep=';')

# Display the first few rows to verify
print("Titanic Train Data:")
print(titanic_train.head())

print("\nWine Quality (Red) Data:")
print(wine_red.head())

print("\nWine Quality (White) Data:")
print(wine_white.head())

# Handling missing values for Titanic Dataset
titanic_train.fillna({'Age': titanic_train['Age'].median(), 'Embarked': 'S'}, inplace=True)
titanic_test.fillna({'Age': titanic_test['Age'].median(), 'Fare': titanic_test['Fare'].median(), 'Embarked': 'S'}, inplace=True)

# Convert categorical variables into numeric
titanic_train['Sex'] = titanic_train['Sex'].map({'male': 0, 'female': 1})
titanic_test['Sex'] = titanic_test['Sex'].map({'male': 0, 'female': 1})

titanic_train = pd.get_dummies(titanic_train, columns=['Embarked'], drop_first=True)
titanic_test = pd.get_dummies(titanic_test, columns=['Embarked'], drop_first=True)

# Drop irrelevant features
titanic_train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
titanic_test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Save the cleaned data
titanic_train.to_csv('../data/cleaned_titanic_train.csv', index=False)
titanic_test.to_csv('../data/cleaned_titanic_test.csv', index=False)

# Normalize Wine Quality Dataset
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Normalize features for Red Wine
wine_red_scaled = pd.DataFrame(
    scaler.fit_transform(wine_red.drop('quality', axis=1)),
    columns=wine_red.columns[:-1]
)
wine_red_scaled['quality'] = wine_red['quality']

# Normalize features for White Wine
wine_white_scaled = pd.DataFrame(
    scaler.fit_transform(wine_white.drop('quality', axis=1)),
    columns=wine_white.columns[:-1]
)
wine_white_scaled['quality'] = wine_white['quality']

# Save the cleaned data
wine_red_scaled.to_csv('../data/cleaned_wine_red.csv', index=False)
wine_white_scaled.to_csv('../data/cleaned_wine_white.csv', index=False)
