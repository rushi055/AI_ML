# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset from CSV
data = pd.read_csv('titanic.csv')

# Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
# print(data.head())
print(data[['Name', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']].head())


#------------------------------------------------------------------------------------------------------
# Check for missing values
print("Missing values in the dataset:")
print(data.isnull().sum())

# Fill missing 'Age' values with the median age
# data['Age'].fillna(data['Age'].median(),inplace=True)
data['Age'] = data['Age'].fillna(data['Age'].median())

# Fill missing 'Embarked' values with the mode (most common value)
# data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True) #we can write inplace= True or not it doesnt affect the output
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Drop the 'Cabin' column as it has too many missing values
data.drop(columns=['Cabin'], inplace=True)

# Drop the 'Name', 'Ticket' columns as they are not useful for prediction
data.drop(columns=['Name', 'Ticket'], inplace=True)

# Convert categorical columns 'Sex' and 'Embarked' into numerical format
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Display the cleaned dataset
print("Cleaned dataset:")
print(data.head())

#------------------------------------------------------------------------------------------------------

# Define features (X) and labels (y)
X = data.drop(columns=['Survived'])  # All columns except 'Survived'
y = data['Survived']  # The target column

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


#-------------------------------------------------------------------------------------------

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=500)

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

