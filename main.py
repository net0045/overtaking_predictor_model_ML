import os
from logic.data_creator import DataCreator
from logic.model_trainer import ModelTrainer

def test_model():
    pass

def validate_model():
    pass

def model_training(name):
    model_trainer = ModelTrainer(name)
    model_trainer.train_model()


def generate_data():
    os.makedirs("data/features", exist_ok=True)
    dtcreator = DataCreator(2000)
    dtcreator.generate_overtaking_data()
    
    X_train, X_val, X_test, y_train, y_val, y_test = dtcreator.get_split_data()
    
    train_full = X_train.copy()
    train_full['target'] = y_train
    
    valid_full = X_val.copy()
    valid_full['target'] = y_val
    
    test_full = X_test.copy()
    test_full['target'] = y_test
    
    train_full.to_csv("data/train.csv", index=False)
    valid_full.to_csv("data/valid.csv", index=False)
    test_full.to_csv("data/test.csv", index=False)
    
    print("Datasets created successfully in 'data/features' folder.")


def main():
    print("Possible operations:")
    print("G - generate data")
    print("MT - model training")
    print("V - validate model")
    print("T - test model")

    operation = input("Enter which operation you want to do (G / MT / V / T)! ")

    if operation == "G":
        print("Starting generating the dataset values...")
        generate_data()
    elif operation == "MT":
        print("Starting Model training...")
        model_name = input("Name your model: ")
        model_training(model_name)
    elif operation == "V":
        print("Starting Model validation...")
    elif operation == "T":
        print("Starting Model testing...")
    else:
        raise ValueError("Entered operation is not supported")

if __name__ == "__main__":
    main()