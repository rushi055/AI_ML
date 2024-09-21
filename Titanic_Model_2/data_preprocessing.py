from sklearn.model_selection import train_test_split

def preprocess_data(data):
    # Convert categorical columns 'Sex' and 'Embarked' into numerical format
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Define features (X) and labels (y)
    X = data.drop(columns=['Survived', 'Name'])  # Drop 'Survived' and 'Name'
    y = data['Survived']  # The target column

    # Split the dataset into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
