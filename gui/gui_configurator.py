import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
from logic.data_creator import DataCreator
import os

class GUIConfigurator(ctk.CTk):
    def __init__(self, model_trainer, model_nn_trainer):
        super().__init__()
        self.model_trainer = model_trainer
        self.model_nn_trainer = model_nn_trainer
        self.data_generator = DataCreator()
        self.create_gui()
        
    def create_gui(self):
        self.title("Overtaking Predictor Model GUI")
        self.geometry("1280x800")
        ctk.set_appearance_mode("dark")

        # Horní část - Nadpis a výběr operace
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
                self.log_output.insert("end", "Status: Preparing RF training layout...\n")
                # Zde můžeš přidat metodu self.rf_training_ui()
            case "Neural Network training":
                self.log_output.insert("end", "Status: Preparing NN training layout...\n")
            case "Model testing":
                self.log_output.insert("end", "Status: Preparing testing environment...\n")

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

