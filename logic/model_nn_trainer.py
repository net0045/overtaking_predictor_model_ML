from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

class ModelNNTrainer():
    def __init__(self, name = "NN_model"):
        self.name = name
        self.path = f"data/models/{name}"

    def scale_data(self, X_train, X_valid):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        
        joblib.dump(scaler, f'{self.path}/scaler.pkl')

        return X_train_scaled, X_valid_scaled
    
    def show_correlation_matrix(self, df, is_valid=False):
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        if is_valid == False:
            file_path = os.path.join(self.path, f"correlation_{self.name}.png")
            plt.title(f'Correlation Matrix NN Model: {self.name}')
            plt.savefig(file_path)
        else:
            file_path = os.path.join(self.path, f"valid_correlation_{self.name}.png")
            plt.title(f'Validation Correlation Matrix NN Model: {self.name}')
            plt.savefig(file_path)
        

    def show_confusion_matrix(self, matrix):
        file_path = os.path.join(self.path, f"confusion_{self.name}.png")
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='g', cmap='Reds', 
                    xticklabels=['Danger', 'Safe'], 
                    yticklabels=['Danger', 'Safe'])

        plt.title(f'Confusion Matrix Model: {self.name}')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig(file_path)
        plt.show()
    
    def show_history(self, history):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{self.path}/training_history.png')
        plt.show()
    
    def train_nn_model_config(self, threshold, name, train_path, valid_path):
        self.name = name
        self.path = os.path.join("data/models", name)
        os.makedirs(self.path, exist_ok=True)

        df = pd.read_csv(train_path)
        df_valid = pd.read_csv(valid_path)

        if df is None or df_valid is None:
            raise ValueError("Some of the dataset is not loaded")
        
        feats_to_drop = ['target']

        X_train = df.drop(feats_to_drop, axis=1)
        y_train = df['target']

        X_valid = df_valid.drop(feats_to_drop, axis=1)
        y_valid = df_valid['target']

        # Scale the data
        X_train_scaled, X_valid_scaled = self.scale_data(X_train, X_valid)

        self.show_correlation_matrix(df)
        self.show_correlation_matrix(df_valid, True)

        print("Started training Neural Network model...")
        nn = MyNeuralNetwork()
        model = nn.build_model(X_train_scaled.shape[1])
        history = model.fit(
            X_train_scaled, y_train, 
            epochs=50, 
            batch_size=32, 
            validation_data=(X_valid_scaled, y_valid),
            verbose=1
        )

        y_pred = (model.predict(X_valid_scaled) > threshold).astype(int)
        conf_matrix = confusion_matrix(y_valid, y_pred)
       
        self.show_history(history)
        self.show_confusion_matrix(conf_matrix)

        model_save_path = os.path.join(self.path, f'{self.name}.keras')
        model.save(model_save_path)
        print(f"\nModel and Scaler stored in: {self.path}")
        
        return history



    def train_nn_model(self, threshold=0.9):
        os.makedirs(self.path, exist_ok=True)
        # Load the data
        df = pd.read_csv("./data/features/train3k_relV.csv")
        df_valid = pd.read_csv("./data/features/valid3k_relV.csv")

        if df is None or df_valid is None:
            raise ValueError("Some of the dataset is not loaded")

        # Split the data 
        # feature to drop
        feats_to_drop = ['target']

        X_train = df.drop(feats_to_drop, axis=1)
        y_train = df['target']

        X_valid = df_valid.drop(feats_to_drop, axis=1)
        y_valid = df_valid['target']

        # Scale the data
        X_train_scaled, X_valid_scaled = self.scale_data(X_train, X_valid)

        self.show_correlation_matrix(df)
        self.show_correlation_matrix(df_valid, True)

        print("Started training Neural Network model...")
        nn = MyNeuralNetwork()
        model = nn.build_model(X_train_scaled.shape[1])
        history = model.fit(
            X_train_scaled, y_train, 
            epochs=50, 
            batch_size=32, 
            validation_data=(X_valid_scaled, y_valid),
            verbose=1
        )

        y_pred = (model.predict(X_valid_scaled) > threshold).astype(int)
        conf_matrix = confusion_matrix(y_valid, y_pred)
       
        self.show_history(history)
        self.show_confusion_matrix(conf_matrix)

        model_save_path = os.path.join(self.path, f'{self.name}.keras')
        model.save(model_save_path)
        print(f"\nModel and Scaler stored in: {self.path}")
        
        return history
    

class MyNeuralNetwork():
    def build_model(self, inp_shape):
        model = Sequential()
        model.add(Dense(16, activation='relu', input_shape=(inp_shape,)))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model