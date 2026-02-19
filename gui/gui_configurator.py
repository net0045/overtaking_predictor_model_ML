import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
from logic.data_creator import DataCreator
from logic.model_trainer import ModelTrainer
from logic.model_nn_trainer import ModelNNTrainer
import os
from tkinter import filedialog

class GUIConfigurator(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.model_trainer = ModelTrainer()
        self.model_nn_trainer = ModelNNTrainer()
        self.data_generator = DataCreator()
        self.create_gui()
        
    def create_gui(self):
        self.title("Overtaking Predictor Model GUI")
        self.geometry("1280x800")
        ctk.set_appearance_mode("dark")

        header_label = ctk.CTkLabel(self, text="Overtaking Predictor Control Center", font=("Arial", 24, "bold"))
        header_label.pack(pady=20)

        self.seg_button = ctk.CTkSegmentedButton(
            self, 
            values=["Data generation", "Random Forest training", "Neural Network training", "Model testing"],
            command=self.handle_operation 
        )
        self.seg_button.pack(pady=10)

        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)

        self.left_panel = ctk.CTkFrame(self.main_container, width=600)
        self.left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        self.left_panel_label = ctk.CTkLabel(self.left_panel, text="Select an operation to see options", text_color="gray")
        self.left_panel_label.pack(expand=True)

        self.right_panel = ctk.CTkFrame(self.main_container, width=600)
        self.right_panel.pack(side="right", fill="both", expand=True, padx=(10, 0))

        ctk.CTkLabel(self.right_panel, text="System Logs", font=("Arial", 14, "bold")).pack(pady=5)
        self.log_output = ctk.CTkTextbox(self.right_panel, font=("Consolas", 12))
        self.log_output.pack(fill="both", expand=True, padx=10, pady=10)

    def handle_operation(self, value):
        self.log_output.insert("end", f">>> Operation selected: {value}\n")
        self.log_output.see("end")
        
        for widget in self.left_panel.winfo_children():
            widget.destroy()

        match value:
            case "Data generation":
                self.data_generation_ui() 
            case "Random Forest training":
                self.randon_forest_training_ui()
            case "Neural Network training":
                self.neural_network_training_ui()
            case "Model testing":
                self.log_output.insert("end", "Status: Preparing testing environment...\n")
    
    """Train a random Forest Model GUI""" 
    def randon_forest_training_ui(self):
        ctk.CTkLabel(self.left_panel, text="Random Forest Training Configuration", font=("Arial", 16, "bold")).pack(pady=10)

        # Model name and confidence threshold
        row1 = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=5)

        name_container = ctk.CTkFrame(row1, fg_color="transparent")
        name_container.pack(side="left", expand=True, fill="x")
        ctk.CTkLabel(name_container, text="Model Name").pack(side="top", anchor="w", padx=5)
        self.model_rf_name = ctk.CTkEntry(name_container, placeholder_text="Give your model a name", width=250)
        self.model_rf_name.pack(side="top", fill="x", padx=5)

        conf_container = ctk.CTkFrame(row1, fg_color="transparent")
        conf_container.pack(side="left", expand=True, fill="x")
        ctk.CTkLabel(conf_container, text="Confidence Threshold (%)").pack(side="top", anchor="w", padx=5)
        self.confidence_threshold_rf = ctk.CTkEntry(conf_container, placeholder_text="e.g. 80", width=50)
        self.confidence_threshold_rf.pack(side="top", fill="x", padx=5)
        
        estim_container = ctk.CTkFrame(row1, fg_color="transparent")
        estim_container.pack(side="left", expand=True, fill="x")
        ctk.CTkLabel(estim_container, text="Number of Estimators").pack(side="top", anchor="w", padx=5)
        self.n_estimators = ctk.CTkEntry(estim_container, placeholder_text="e.g. 100 (max 1000)", width=50)
        self.n_estimators.pack(side="top", fill="x", padx=5)

        # CSV paths choosing
        row2 = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=(15, 5))

        self.train_data_rf = ctk.StringVar(value="No file selected")
        train_path_label = ctk.CTkEntry(row2, textvariable=self.train_data_rf, state="disabled", width=300)
        train_path_label.pack(side="left", expand=True, fill="x", padx=5)

        self.validation_data_rf = ctk.StringVar(value="No file selected")
        val_path_label = ctk.CTkEntry(row2, textvariable=self.validation_data_rf, state="disabled", width=300)
        val_path_label.pack(side="left", expand=True, fill="x", padx=5)


        # Buttons for browsing CSV files
        row3 = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        row3.pack(fill="x", padx=10, pady=5)

        self.btn_browse_train = ctk.CTkButton(row3, text="Browse Train Data", command=lambda: self.browse_file(False, False), fg_color="#34495e")
        self.btn_browse_train.pack(side="left", expand=True, padx=5)

        self.btn_browse_validation = ctk.CTkButton(row3, text="Browse Validation Data", command=lambda: self.browse_file(True, False), fg_color="#34495e")
        self.btn_browse_validation.pack(side="left", expand=True, padx=5)

        self.buttonStart = ctk.CTkButton(self.left_panel, text="Start Training", width=200, height=40, 
                                          command=self.start_rf_training, fg_color="green", hover_color="#27ae60")
        self.buttonStart.pack(pady=30)
    

    def start_rf_training(self):
        self.log_output.insert("end", "Starting Random Forest training...\n")
        model_name = self.model_rf_name.get() if self.model_rf_name.get() else "rf_model"
        train_path = self.train_data_rf.get()
        valid_path = self.validation_data_rf.get()
        conf_thr = self.confidence_threshold_rf.get() if self.confidence_threshold_rf.get() else "70"
        estimators = self.n_estimators.get() if self.n_estimators.get() else 100
        if not conf_thr.isdigit():
            self.log_output.insert("end", "ERROR: Please enter a valid number for confidence threshold.\n")
            return
        if not estimators.isdigit():
            self.log_output.insert("end", "ERROR: Please enter a valid number for estimators.\n")
            return
        if int(estimators) > 1000:
            self.log_output.insert("end", "ERROR: Please enter a number of estimators less than or equal to 1000.\n")
            return
        else:
            estimators = int(estimators)
        if train_path == "No file selected" or valid_path == "No file selected":
            self.log_output.insert("end", "ERROR: Please select both training and validation datasets.\n")
            return
        
        self.model_trainer.train_model(float(conf_thr)/100, model_name, train_path, valid_path, self.log_output.insert, estimators=estimators)
     

 
    """Model NN trainig UI"""
    def neural_network_training_ui(self):
        ctk.CTkLabel(self.left_panel, text="Neural Network Training Configuration", font=("Arial", 16, "bold")).pack(pady=10)

        # Model name and confidence threshold
        row1 = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=5)

        name_container = ctk.CTkFrame(row1, fg_color="transparent")
        name_container.pack(side="left", expand=True, fill="x")
        ctk.CTkLabel(name_container, text="Model Name").pack(side="top", anchor="w", padx=5)
        self.model_nn_name = ctk.CTkEntry(name_container, placeholder_text="Give your model a name", width=250)
        self.model_nn_name.pack(side="top", fill="x", padx=5)

        conf_container = ctk.CTkFrame(row1, fg_color="transparent")
        conf_container.pack(side="left", expand=True, fill="x")
        ctk.CTkLabel(conf_container, text="Confidence Threshold (%)").pack(side="top", anchor="w", padx=5)
        self.confidence_threshold_nn = ctk.CTkEntry(conf_container, placeholder_text="e.g. 80", width=250)
        self.confidence_threshold_nn.pack(side="top", fill="x", padx=5)

        # CSV paths choosing
        row2 = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=(15, 5))

        self.train_data_nn = ctk.StringVar(value="No file selected")
        train_path_label = ctk.CTkEntry(row2, textvariable=self.train_data_nn, state="disabled", width=300)
        train_path_label.pack(side="left", expand=True, fill="x", padx=5)

        self.validation_data_nn = ctk.StringVar(value="No file selected")
        val_path_label = ctk.CTkEntry(row2, textvariable=self.validation_data_nn, state="disabled", width=300)
        val_path_label.pack(side="left", expand=True, fill="x", padx=5)

        # Buttons for browsing CSV files
        row3 = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        row3.pack(fill="x", padx=10, pady=5)

        self.btn_browse_train = ctk.CTkButton(row3, text="Browse Train Data", command=lambda: self.browse_file(False, True), fg_color="#34495e")
        self.btn_browse_train.pack(side="left", expand=True, padx=5)

        self.btn_browse_validation = ctk.CTkButton(row3, text="Browse Validation Data", command=lambda: self.browse_file(True, True), fg_color="#34495e")
        self.btn_browse_validation.pack(side="left", expand=True, padx=5)

        self.buttonStart = ctk.CTkButton(self.left_panel, text="Start Training", width=200, height=40, 
                                          command=self.start_nn_training, fg_color="green", hover_color="#27ae60")
        self.buttonStart.pack(pady=30)

    def start_nn_training(self):
        pass



    """Data Generation UI - enable user to define features and their ranges"""
    def data_generation_ui(self):
        ctk.CTkLabel(self.left_panel, text="Data Generation Configuration", font=("Arial", 16, "bold")).pack(pady=10)

        self.features_frame = ctk.CTkScrollableFrame(self.left_panel, height=400, label_text="Define Features")
        self.features_frame.pack(fill="x", pady=10, padx=10)

        self.feature_rows = []

        header_frame = ctk.CTkFrame(self.features_frame)
        header_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(header_frame, text="Feature Name", width=150).pack(side="left", padx=5)
        ctk.CTkLabel(header_frame, text="Min", width=80).pack(side="left", padx=5)
        ctk.CTkLabel(header_frame, text="Max", width=80).pack(side="left", padx=5)

        control_container = ctk.CTkFrame(self.left_panel, fg_color="transparent") 
        control_container.pack(pady=10)

        self.file_name = ctk.CTkEntry(control_container, placeholder_text="e.g. my_datasetname", width=200)
        self.file_name.pack(side="left", padx=5)

        self.samples_number = ctk.CTkEntry(control_container, placeholder_text="Number of samples e.g. 1000", width=200)
        self.samples_number.pack(side="left", padx=5)
        
        self.add_btn = ctk.CTkButton(control_container, text="+ Feature", width=100, command=self.add_feature_row)
        self.add_btn.pack(side="left", padx=5)

        self.remove_btn = ctk.CTkButton(control_container, text="- Feature", width=100, command=self.remove_feature_row)
        self.remove_btn.pack(side="left", padx=5)

        self.buttonGD = ctk.CTkButton(control_container, text="Generate", width=100, command=self.generate_data, fg_color="green", hover_color="#27ae60")
        self.buttonGD.pack(side="left", padx=5)

        self.load_default_config()

    def load_default_config(self):
        while self.feature_rows:
            self.remove_feature_row()
        
        default_features = [
            {"name": "v_ego", "min": 50, "max": 130, "description": "Speed of ego vehicle in km/h"},
            {"name": "v_followed", "min": 40, "max": 110, "description": "Speed of vehicle before ego vehicle in km/h"},
            {"name": "v_oncoming", "min": 50, "max": 130, "description": "Speed of oncoming vehicle in km/h"},
            {"name": "dist_flw_m", "min": 10, "max": 50, "description": "Distance to vehicle before ego vehicle in meters"},
            {"name": "dist_oncom_m", "min": 100, "max": 1500, "description": "Distance to oncoming vehicle in meters"},
            {"name": "engine_power_kw", "min": 75, "max": 350, "description": "Engine power in kW"},
            {"name": "weight_kg", "min": 1300, "max": 3000, "description": "Weight of the vehicle in kg"},
            {"name": "road_friction", "min": 0.2, "max": 1.0, "description": "Road friction 1.0 = dry asphalt, 0.5 = wet road, 0.2 = icy road"}
        ]

        for feature in default_features:
            self.add_feature_row()
            last_row = self.feature_rows[-1]
            last_row["name"].insert(0, feature["name"])
            last_row["name"].configure(state="disabled", fg_color="#2b2b2b")
            last_row["min"].insert(0, str(feature["min"]))
            last_row["max"].insert(0, str(feature["max"]))
            last_row["description"].configure(text=feature["description"])
        
        self.log_output.insert("end", "Standard Configuration loaded...\n")

    def remove_feature_row(self):
        if self.feature_rows and len(self.feature_rows) > 8:
            last_row = self.feature_rows.pop()
            last_row["frame"].destroy()

    def add_feature_row(self):
        row_frame = ctk.CTkFrame(self.features_frame)
        row_frame.pack(fill="x", padx=5, pady=2)

        name_entry = ctk.CTkEntry(row_frame, placeholder_text="e.g. v_ego", width=150)
        name_entry.pack(side="left", padx=5)

        min_entry = ctk.CTkEntry(row_frame, placeholder_text="0", width=100)
        min_entry.pack(side="left", padx=5)

        max_entry = ctk.CTkEntry(row_frame, placeholder_text="100", width=100)
        max_entry.pack(side="left", padx=5)

        description_label = ctk.CTkLabel(row_frame, text="", width=500)
        description_label.pack(side="left", padx=5)

        self.feature_rows.append({
            "frame": row_frame,
            "name": name_entry,
            "min": min_entry,
            "max": max_entry,
            "description": description_label
        })
    
    def generate_data(self):
        self.log_output.insert("end", "Gathering parameters...\n")
        
        final_config = []
        for row in self.feature_rows:
            f_name = row["name"].get()
            f_min = row["min"].get()
            f_max = row["max"].get()
            if f_name and f_min and f_max:
                final_config.append({"name": f_name, "min": f_min, "max": f_max})
        
        try:
            samples = int(self.samples_number.get())
        except:
            samples = 1000

        self.log_output.insert("end", f"Starting generation of {samples} samples...\n")
        result = self.data_generator.configurate_overtaking_data(samples, final_config)

        X_train, X_val, X_test, y_train, y_val, y_test = self.data_generator.get_split_data()
    
        train_full = X_train.copy()
        train_full['target'] = y_train
        
        valid_full = X_val.copy()
        valid_full['target'] = y_val
        
        test_full = X_test.copy()
        test_full['target'] = y_test
        
        file_name = self.file_name.get() if self.file_name.get() else "default"
        train_full.to_csv(f"data/features/train_{file_name}.csv", index=False)
        valid_full.to_csv(f"data/features/valid_{file_name}.csv", index=False)
        test_full.to_csv(f"data/features/test_{file_name}.csv", index=False)

        self.log_output.insert("end", f"SUCCESS: Datasets saved to data/features/\n")
        self.log_output.insert("end", f"Train size: {len(train_full)}, Val size: {len(valid_full)}, Test size: {len(test_full)}\n")
        self.log_output.see("end")


    def browse_file(self, is_valid = False, is_nn=False):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            if is_nn:
                if is_valid: self.validation_data_nn.set(path)
                else: self.train_data_nn.set(path)
            else:
                if is_valid: self.validation_data_rf.set(path) 
                else: self.train_data_rf.set(path)
    
