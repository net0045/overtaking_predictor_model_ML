import os
from logic.data_creator import DataCreator
from logic.model_trainer import ModelTrainer
from logic.model_nn_trainer import ModelNNTrainer
from gui.gui_configurator import GUIConfigurator

def test_model():
    pass

def validate_model():
    pass

def model_training(name):
    model_trainer = ModelTrainer(name)
    model_trainer.train_model()

def model_nn_training(name):
    model_trainer = ModelNNTrainer(name)
    model_trainer.train_nn_model()


def generate_data():
    os.makedirs("data/features", exist_ok=True)
    dtcreator = DataCreator(3000)
    dtcreator.generate_overtaking_data_manual()
    
    X_train, X_val, X_test, y_train, y_val, y_test = dtcreator.get_split_data()
    
    train_full = X_train.copy()
    train_full['target'] = y_train
    
    valid_full = X_val.copy()
    valid_full['target'] = y_val
    
    test_full = X_test.copy()
    test_full['target'] = y_test
    
    train_full.to_csv("data/features/train3k_relV.csv", index=False)
    valid_full.to_csv("data/features/valid3k_relV.csv", index=False)
    test_full.to_csv("data/features/test3k_relV.csv", index=False)
    
    print("Datasets created successfully in 'data/features' folder.")


def main():
    app = GUIConfigurator(ModelTrainer("rf_model"), ModelNNTrainer("nn_model"))
    app.mainloop()
    """print("Possible operations:")
    print("G - generate data")
    print("MT - model training")
    print("NN - neural network model training")
    print("V - validate model")
    print("T - test model")

    operation = input("Enter which operation you want to do (G / MT / NN / V / T)! ")

    if operation == "G":
        print("Starting generating the dataset values...")
        generate_data()
    elif operation == "MT":
        print("Starting Model training...")
        model_name = input("Name your model: ")
        model_training(model_name)
    elif operation == "NN":
        print("Starting Neural Network Model training...")
        model_name = input("Name your model: ")
        model_nn_training(model_name)
    elif operation == "V":
        print("Starting Model validation...")
    elif operation == "T":
        print("Starting Model testing...")
    else:
        raise ValueError("Entered operation is not supported")
        """

if __name__ == "__main__":
    main()