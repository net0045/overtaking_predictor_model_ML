import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class ModelTrainer():
    def __init__(self, name = "Basic_model"):
        self.name = name
        self.path = f"data/models/{name}"

    def show_correlation_matrix(self, data_frame, isValidation = False):
        plt.figure(figsize=(10, 8))
        sns.heatmap(data_frame.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        if isValidation == False:
            file_path = os.path.join(self.path, f"correlation_{self.name}.png")
            plt.title(f'Correlation Matrix Model: {self.name}')
            plt.savefig(file_path)
        else:
            file_path = os.path.join(self.path, f"valid_correlation_{self.name}.png")
            plt.title(f'Validation Correlation Matrix Model: {self.name}')
            plt.savefig(file_path)
        
        plt.show()

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

    
    def train_model(self):
        os.makedirs(self.path, exist_ok=True)
        # Load the data
        df = pd.read_csv("./data/features/train.csv")
        df_valid = pd.read_csv("./data/features/valid.csv")

        if df is None or df_valid is None:
            raise ValueError("Some of the dataset is not loaded")

        # Split the data 
        X_train = df.drop('target', axis=1)
        y_train = df['target']

        X_valid = df_valid.drop('target', axis=1)
        y_valid = df_valid['target']

        # Show correlation matrices
        self.show_correlation_matrix(df)
        self.show_correlation_matrix(df_valid, True)

        print("Started training RandomForest model...")
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)

        # Evaluates on validation data
        y_pred = classifier.predict(X_valid)
        accuracy = accuracy_score(y_valid, y_pred)
        conf_matrix = confusion_matrix(y_valid, y_pred)

        self.show_confusion_matrix(conf_matrix)
        
        print("\n#Result against validation data#")
        print("Accuracy (%): ", accuracy)
        print(classification_report(y_valid, y_pred))

        importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': classifier.feature_importances_
        }).sort_values('importance', ascending=False)

        importances.to_csv(os.path.join(self.path, "importances.csv"), index=False)

        joblib.dump(classifier, f'data/models/{self.name}.pkl')
        print("\nModel stored to data/models/..")







