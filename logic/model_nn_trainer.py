from pyexpat import model

from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

class ModelNNTrainer():
    def __init__(self, name = "NN_model"):
        self.name = name
        self.path = f"data/models/{name}"
        self.layers = None

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
    
    def train_nn_model(self, threshold, name, train_path, valid_path, epochs, batch_size, log_function):
        self.name = name
        self.path = os.path.join("data/models", name)

        def log(msg):
            log_function("end", f">>> [NN training] {msg}\n")

        gui_log_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: log(f"Epoch {epoch+1}/{epochs} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}"))
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        os.makedirs(self.path, exist_ok=True)
        log(f"Loading datasets from {train_path} and {valid_path}...")
        df = pd.read_csv(train_path)
        df_valid = pd.read_csv(valid_path)

        if df is None or df_valid is None:
            log(f"[ERR] Failed to load datasets!")
            return
        
        feats_to_drop = ['target']

        X_train = df.drop(feats_to_drop, axis=1)
        y_train = df['target']

        X_valid = df_valid.drop(feats_to_drop, axis=1)
        y_valid = df_valid['target']

        # Scale the data
        X_train_scaled, X_valid_scaled = self.scale_data(X_train, X_valid)

        self.show_correlation_matrix(df)
        self.show_correlation_matrix(df_valid, True)

        model_nn = self.build_nn_architecture(X_train_scaled.shape[1], self.layers, log_function)
        log(f"Training model {name} with threshold {threshold}...")
        history = model_nn.fit(
            X_train_scaled, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=(X_valid_scaled, y_valid),
            callbacks=[gui_log_callback, early_stop_callback],
            verbose=0
        )

        y_pred = (model_nn.predict(X_valid_scaled) > threshold).astype(int)
        conf_matrix = confusion_matrix(y_valid, y_pred)
       
        self.show_history(history)
        self.show_confusion_matrix(conf_matrix)

        model_save_path = os.path.join(self.path, f'{self.name}.keras')
        model_nn.save(model_save_path)
        log(f"\nModel and Scaler stored in: {self.path}")
        
        return history
    
    
    def build_nn_architecture(self, input_shape, layers, log_function):
        model = Sequential()
        index = 1
        output_layer = [l for l in layers if l["output"]]
        only_hidden = [l for l in layers if not l["output"]]
        sorted_hidden = sorted(only_hidden, key=lambda x: x["id"])

        def get_val(obj):
            return obj.get() if hasattr(obj, 'get') else obj
        
        for layer in sorted_hidden:
            layer_neurons = get_val(layer['neurons'])
            layer_activation = get_val(layer['activation'])
            log_function("end", f"Building layer with {layer_neurons} neurons and {layer_activation} activation function\n")
            if layer["id"] != index:
                log_function("end", f"Error: There has been an unordered layer configuration. Please check the layers order and try again.")
                return None
            
            if layer["id"] == 1:
                model.add(Dense(int(layer_neurons), activation=layer_activation, input_shape=(input_shape,)))
                index += 1
            else:
                model.add(Dense(int(layer_neurons), activation=layer_activation))
                index += 1

        out_act = get_val(output_layer[0]['activation'])
    
        log_function("end", f"Adding Output layer (1 neuron, {out_act})\n")
        model.add(Dense(1, activation=out_act))
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

