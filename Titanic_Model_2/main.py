import data_cleaning 
import data_preprocessing 
import model_training 

# Define the file path for the dataset
file_path = 'titanic.csv'

def main():
    # Step 1: Load and Clean Data
    data = data_cleaning.load_data(file_path)
    data_cleaned = data_cleaning.clean_data(data)

    # Step 2: Preprocess Data
    X_train, X_test, y_train, y_test = data_preprocessing.preprocess_data(data_cleaned)

    # Step 3: Train the Model
    model = model_training.train_model(X_train, y_train)

    # Step 4: Evaluate the Model
    accuracy = model_training.evaluate_model(model, X_test, y_test)
    
    print(f"Model accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
