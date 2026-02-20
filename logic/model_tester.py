from cv2 import threshold
import pandas as pd
import tensorflow as tf
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class ModelTester():
    def __init__(self, name = "Basic_model"):
        self.name = name
        self.path = os.path.join("data", "models", "test_results", name)
           
    def show_confusion_matrix(self, matrix):
        os.makedirs(self.path, exist_ok=True)
        file_path = os.path.join(self.path, f"test_confusion_{self.name}.png")
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='g', cmap='Reds', 
                    xticklabels=['Danger', 'Safe'], 
                    yticklabels=['Danger', 'Safe'])

        plt.title(f'Confusion Matrix Model (TEST): {self.name}')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig(file_path)
        plt.show()

    def test_model_rf(self, model_path, test_path, model_name, log_function):
        self.name = model_name
        self.path = os.path.join("data/models/test_results", model_name)

        model = joblib.load(model_path)
        df_test = pd.read_csv(test_path)

        def log(msg):
            log_function("end", f">>> [TESTER] {msg}\n")

        if df_test is None:
            log(f"Failed to load test dataset from {test_path}!")
            return

        feats_to_drop = ['target']
        X_test = df_test.drop(feats_to_drop, axis=1)
        y_test = df_test['target']

        try:
            y_pred = model.predict(X_test)
        except ValueError as e:
            if "feature names" in str(e).lower():
                log("ERROR: Mismatch in feature names between training and test datasets.")
                log("Please ensure the test dataset has the same columns as the training dataset and picked right model.")
                log(f"Details: {str(e).split('fit.')[-1].strip()}") 
            else:
                log(f"Error during prediction: {str(e)}")
            return
        
        acc = accuracy_score(y_test, y_pred)
        log(f"Test Accuracy: {acc:.4f}")

        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        log(f"Classification Report:\n{class_report}")
        
        self.show_confusion_matrix(conf_matrix)
        saved_path = self.save_results_to_txt(acc, class_report)
        log(f"Test results saved to: {saved_path}")

    def test_model_nn(self, model_path, scaler_path, test_path, model_name, log_function):
        self.name = model_name
        self.path = os.path.join("data/models/test_results", model_name)

        try:
            model = tf.keras.models.load_model(model_path)
            scaler = joblib.load(scaler_path)
            df_test = pd.read_csv(test_path)

            def log(msg):
                log_function("end", f">>> [TESTER] {msg}\n")

            if df_test is None:
                log(f"Failed to load test dataset from {test_path}!")
                return

            feats_to_drop = ['target']
            X_test = df_test.drop(feats_to_drop, axis=1)
            y_test = df_test['target']

            X_test_scaled = scaler.transform(X_test)

            try:
                y_pred_prob = model.predict(X_test_scaled)
                y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            except ValueError as e:
                if "feature names" in str(e).lower():
                    log("ERROR: Mismatch in feature names between training and test datasets.")
                    log("Please ensure the test dataset has the same columns as the training dataset and picked right model.")
                    log(f"Details: {str(e).split('fit.')[-1].strip()}") 
                else:
                    log(f"Error during prediction: {str(e)}")
                return
            
            acc = accuracy_score(y_test, y_pred)
            log(f"Test Accuracy: {acc:.4f}")

            class_report = classification_report(y_test, y_pred)
            log(f"Classification Report:\n{class_report}")

            conf_matrix = confusion_matrix(y_test, y_pred)
            self.show_confusion_matrix(conf_matrix)
            saved_path = self.save_results_to_txt(acc, class_report)
            log(f"Test results saved to: {saved_path}")
        except Exception as e:
            log(f"Critical error during testing: {str(e)}")

    def save_results_to_txt(self, accuracy, class_report):
        os.makedirs(self.path, exist_ok=True)
        file_path = os.path.join(self.path, f"test_results_{self.name}.txt")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"### TEST RESULTS SUMMARY ###\n")
                f.write(f"############################\n")
                f.write(f"Model Name: {self.name}\n")
                f.write(f"Accuracy:   {accuracy:.4f} ({round(accuracy*100, 2)}%)\n\n")
                f.write(f"CLASSIFICATION REPORT:\n")
                f.write(f"----------------------\n")
                f.write(class_report)
            return file_path
        except Exception as e:
            print(f"Error saving file: {e}")
            return None