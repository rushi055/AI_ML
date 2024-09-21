import pandas as pd

def load_data(file_path):
    # Load the dataset from CSV
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    # Fill missing 'Age' values with the median age (Avoid using inplace=True)
    data['Age'] = data['Age'].fillna(data['Age'].median())
    
    # Fill missing 'Embarked' values with the mode (most common value)
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    
    # Drop the 'Cabin' and 'Ticket' columns
    data = data.drop(columns=['Cabin', 'Ticket'])

    return data
