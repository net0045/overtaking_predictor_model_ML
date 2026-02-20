from os import path

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
from logic.data_creator import DataCreator
from logic.model_trainer import ModelTrainer
from logic.model_nn_trainer import ModelNNTrainer
from logic.model_tester import ModelTester
import os
from tkinter import filedialog

class GUIConfigurator(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.model_trainer = ModelTrainer()
        self.model_nn_trainer = ModelNNTrainer()
        self.data_generator = DataCreator()
        self.model_tester = ModelTester()
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

        self.left_panel = ctk.CTkScrollableFrame(self.main_container, width=600)
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
                self.model_testing_ui()
    
    """Train a random Forest Model GUI""" 
    def randon_forest_training_ui(self):
        ctk.CTkLabel(self.left_panel, text="Random Forest Training Configuration", font=("Arial", 16, "bold")).pack(pady=10)

        # Model name and confidence threshold
        row1 = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=5)

        name_container = ctk.CTkFrame(row1, fg_color="transparent")
        name_container.pack(side="left", expand=True, fill="x")
        ctk.CTkLabel(name_container, text="Model Name").pack(side="top", anchor="w", padx=5)
        self.model_rf_name = ctk.CTkEntry(name_container, placeholder_text="Give your model a name", width=200)
        self.model_rf_name.pack(side="top", fill="x", padx=5)

        conf_container = ctk.CTkFrame(row1, fg_color="transparent")
        conf_container.pack(side="left", expand=True, fill="x")
        ctk.CTkLabel(conf_container, text="Confidence Threshold (%)").pack(side="top", anchor="w", padx=5)
        self.confidence_threshold_rf = ctk.CTkEntry(conf_container, placeholder_text="e.g. 80", width=25)
        self.confidence_threshold_rf.pack(side="top", fill="x", padx=5)
        
        estim_container = ctk.CTkFrame(row1, fg_color="transparent")
        estim_container.pack(side="left", expand=True, fill="x")
        ctk.CTkLabel(estim_container, text="Number of Estimators").pack(side="top", anchor="w", padx=5)
        self.n_estimators = ctk.CTkEntry(estim_container, placeholder_text="e.g. 100 (max 1000)", width=25)
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
        self.model_nn_name = ctk.CTkEntry(name_container, placeholder_text="Give your model a name", width=200)
        self.model_nn_name.pack(side="top", fill="x", padx=5)

        conf_container = ctk.CTkFrame(row1, fg_color="transparent")
        conf_container.pack(side="left", expand=True, fill="x")
        ctk.CTkLabel(conf_container, text="Confidence Threshold (%)").pack(side="top", anchor="w", padx=5)
        self.confidence_threshold_nn = ctk.CTkEntry(conf_container, placeholder_text="e.g. 80", width=25)
        self.confidence_threshold_nn.pack(side="top", fill="x", padx=5)

        epoch_container = ctk.CTkFrame(row1, fg_color="transparent")
        epoch_container.pack(side="left", expand=True, fill="x")
        ctk.CTkLabel(epoch_container, text="Number of Epochs").pack(side="top", anchor="w", padx=5)
        self.epoch_nn = ctk.CTkEntry(epoch_container, placeholder_text="e.g. 100 (max 500)", width=25)
        self.epoch_nn.pack(side="top", fill="x", padx=5)

        batch_container = ctk.CTkFrame(row1, fg_color="transparent")
        batch_container.pack(side="left", expand=True, fill="x")
        ctk.CTkLabel(batch_container, text="Batch Size").pack(side="top", anchor="w", padx=5)
        self.batch_size_nn = ctk.CTkEntry(batch_container, placeholder_text="e.g. 32", width=25)
        self.batch_size_nn.pack(side="top", fill="x", padx=5)

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

        row4 = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        row4.pack(fill="x", padx=10, pady=5)

        self.nn_vizualizer_container = ctk.CTkFrame(row4, fg_color="transparent")
        self.nn_vizualizer_container.pack(fill="x")

        self.nn_seg_btn = ctk.CTkSegmentedButton(
            self.left_panel, 
            values=["Prebuild NN Architecture", "Custom NN Architecture"],
            command=lambda v: self.handle_chosen_nn_architecture(v, self.nn_vizualizer_container) 
        )
        self.nn_seg_btn.pack(pady=10)
        self.nn_seg_btn.set("Prebuild NN Architecture")

        self.handle_chosen_nn_architecture("Prebuild NN Architecture", self.nn_vizualizer_container)

        self.buttonStart = ctk.CTkButton(self.left_panel, text="Start Training", width=200, height=40, 
                                          command=self.start_nn_training, fg_color="green", hover_color="#27ae60")
        self.buttonStart.pack(pady=30)
    
    def handle_chosen_nn_architecture(self, value, container):
        for widget in container.winfo_children():
            widget.destroy()
        if value == "Prebuild NN Architecture":
            ctk.CTkLabel(container, text="Using optimized ADAS architecture (3 layers: 64, 32, 16)", text_color="gray").pack(pady=10)
            self.nn_architecture_ui(container, True)
            #self.model_nn_trainer.use_prebuilt_architecture = True
        else:
            self.nn_architecture_ui(container)
            #self.model_nn_trainer.use_prebuilt_architecture = False
    
    def nn_architecture_ui(self, container, prebuilded = False):
        self.nn_hid_layers_data = [] 
        self.layers_visualizer_frame = ctk.CTkScrollableFrame(
            container, 
            height=200, 
            orientation="horizontal", 
            label_text="Sequential Dense Neural Network Architecture",
            label_font=("Arial", 12, "bold")
        )
        self.layers_visualizer_frame.pack(fill="x", padx=5, pady=5)
        self.add_input_layer()

        if prebuilded:
            self.add_prebuild_nn_layer(16, "relu", 1)
            self.add_prebuild_nn_layer(8, "relu", 2)
            self.add_output_layer("sigmoid", prebuilded=True)
        else:
            ctrl_frame = ctk.CTkFrame(container, fg_color="transparent")
            ctrl_frame.pack(fill="x", pady=(5, 10))
            
            ctk.CTkButton(ctrl_frame, text="+ Add Layer", width=120, command=self.add_custom_nn_layer, fg_color="#267cb6", hover_color="#1d5d8a").pack(side="left", padx=5)
            ctk.CTkButton(ctrl_frame, text="- Remove Layer", width=120, command=self.remove_custom_nn_layer, fg_color="#c02741", hover_color="#921d32").pack(side="left", padx=5)
            
            self.add_custom_nn_layer()
            self.add_output_layer("sigmoid")
        
    # Adds a fixed input layer with disabled entry for number of neurons (equal to number of features)
    def add_input_layer(self):
        layer_box = ctk.CTkFrame(self.layers_visualizer_frame, width=140, height=180, border_width=2, border_color="#cc2115", corner_radius=20, fg_color="#944E4B")
        layer_box.pack(side="left", padx=10, pady=5)
        layer_box.pack_propagate(False)
        ctk.CTkLabel(layer_box, text="Input Layer", font=("Arial", 11, "bold")).pack(pady=2)

        ctk.CTkLabel(layer_box, text="Neurons:", font=("Arial", 10)).pack()
        neurons_entry = ctk.CTkEntry(layer_box, width=80)
        neurons_entry.insert(0, "Features") 
        neurons_entry.configure(state="disabled")
        neurons_entry.pack(pady=1)

    # Adds an output layer with 1 neuron and selectable activation function, entries are disabled if prebuilded architecture is chosen
    def add_output_layer(self, activation, prebuilded = False):
        layer_box = ctk.CTkFrame(self.layers_visualizer_frame, width=140, height=180, border_width=2, border_color="#2a8d3a", corner_radius=20, fg_color="#2a553f")
        layer_box.pack(side="left", padx=10, pady=5)
        layer_box.pack_propagate(False)
        ctk.CTkLabel(layer_box, text="Output Layer", font=("Arial", 11, "bold")).pack(pady=2)

        ctk.CTkLabel(layer_box, text="Neurons:", font=("Arial", 10)).pack()
        neurons_entry = ctk.CTkEntry(layer_box, width=80)
        neurons_entry.insert(0, str(1)) 
        neurons_entry.configure(state="disabled")
        neurons_entry.pack(pady=1)

        ctk.CTkLabel(layer_box, text="Activation:", font=("Arial", 10)).pack()
        activation_combo = ctk.CTkComboBox(layer_box, width=100, values=["relu", "sigmoid", "tanh", "linear"])
        activation_combo.set(activation)
        if prebuilded:
            activation_combo.configure(state="disabled")
        activation_combo.pack(pady=1)

        self.nn_hid_layers_data.append({
            "id": "output",
            "neurons": 1,
            "activation": activation_combo,
            "output": True,
            "frame": layer_box
        })
    
    # Adds a hidden layer with predefined number of neurons and activation function
    def add_prebuild_nn_layer(self, neurons, activation, index):
        self.nn_hid_layers_data.append({
            "id": index,
            "neurons": neurons,
            "activation": activation,
            "output": False
        })

        layer_box = ctk.CTkFrame(self.layers_visualizer_frame, width=140, height=180, border_width=2, border_color="#122f4d", fg_color="#2a3f55")
        layer_box.pack(side="left", padx=10, pady=5)
        layer_box.pack_propagate(False)

        ctk.CTkLabel(layer_box, text=f"Hid. Layer {index}", font=("Arial", 11, "bold")).pack(pady=2)

        ctk.CTkLabel(layer_box, text="Neurons:", font=("Arial", 10)).pack()
        neurons_entry = ctk.CTkEntry(layer_box, width=80, placeholder_text="e.g. 64")
        neurons_entry.insert(0, str(neurons)) 
        neurons_entry.configure(state="disabled")
        neurons_entry.pack(pady=1)

        ctk.CTkLabel(layer_box, text="Activation:", font=("Arial", 10)).pack()
        activation_entry = ctk.CTkEntry(layer_box, width=80, placeholder_text="e.g. relu")
        activation_entry.insert(0, str(activation)) 
        activation_entry.configure(state="disabled")
        activation_entry.pack(pady=1)
    
    def add_custom_nn_layer(self):
        if len(self.nn_hid_layers_data) >= 7:
            self.log_output.insert("end", "Warning: Maximum 6 hidden layers allowed.\n")
            return
    
        output_layer = next((item for item in self.nn_hid_layers_data if item["output"]), None)
        if output_layer:
            output_layer["frame"].pack_forget()

        # Main box
        layer_id = len([l for l in self.nn_hid_layers_data if not l["output"]]) + 1
        layer_box = ctk.CTkFrame(self.layers_visualizer_frame, width=140, height=180, border_width=2, border_color="#122f4d", fg_color="#2a3f55")
        layer_box.pack(side="left", padx=10, pady=5)
        layer_box.pack_propagate(False)

        ctk.CTkLabel(layer_box, text=f"Hid. Layer {layer_id}", font=("Arial", 11, "bold")).pack(pady=2)
        ctk.CTkLabel(layer_box, text="Neurons:", font=("Arial", 10)).pack()
        neurons = ctk.CTkEntry(layer_box, width=80)
        neurons.insert(0, "64") 
        neurons.pack(pady=1)
        ctk.CTkLabel(layer_box, text="Activation:", font=("Arial", 10)).pack()
        activation_combo = ctk.CTkComboBox(layer_box, width=100, values=["relu", "sigmoid", "tanh", "linear"])
        activation_combo.set("relu")
        activation_combo.pack(pady=1)

        new_layer_data = {
            "id": layer_id,
            "neurons": neurons,
            "activation": activation_combo,
            "output": False,
            "frame": layer_box
        }
        
        if output_layer:
            self.nn_hid_layers_data.insert(len(self.nn_hid_layers_data)-1, new_layer_data)
            output_layer["frame"].pack(side="left", padx=10, pady=5)
        else:
            self.nn_hid_layers_data.append(new_layer_data)

    
    def remove_custom_nn_layer(self):
        hidden_layers = [l for l in self.nn_hid_layers_data if not l["output"]]
        len_hidden = len(hidden_layers)
        print(f"Hidden layers before removing: {len_hidden}")
        
        if len(hidden_layers) <= 1:
            self.log_output.insert("end", "Warning: At least one hidden layer is required.\n")
            return

        last_hidden = hidden_layers[-1]
        self.nn_hid_layers_data.pop(self.nn_hid_layers_data.index(last_hidden))
        last_hidden["frame"].destroy()

    def logger_wrapper(self, index, text):
        self.log_output.insert(index, text)
        self.log_output.see("end") 
        self.update_idletasks()

    def start_nn_training(self):
        self.log_output.insert("end", "Starting Neural Network training...\n")
        model_name = self.model_nn_name.get() if self.model_nn_name.get() else "nn_model"
        train_path = self.train_data_nn.get()
        valid_path = self.validation_data_nn.get()
        conf_thr = self.confidence_threshold_nn.get() if self.confidence_threshold_nn.get() else "70"
        epochs = self.epoch_nn.get() if self.epoch_nn.get() else "100"
        batch_size = self.batch_size_nn.get() if self.batch_size_nn.get() else "32"
        if not epochs.isdigit():
            self.log_output.insert("end", "ERROR: Please enter a valid number for epochs.\n")
            return
        if int(epochs) > 500:
            self.log_output.insert("end", "ERROR: Please enter a number of epochs less than or equal to 500.\n")
            return
        else:
            epochs = int(epochs)
        if not batch_size.isdigit():
            self.log_output.insert("end", "ERROR: Please enter a valid number for batch size.\n")
            return
        if not conf_thr.isdigit():
            self.log_output.insert("end", "ERROR: Please enter a valid number for confidence threshold.\n")
            return
        if train_path == "No file selected" or valid_path == "No file selected":
            self.log_output.insert("end", "ERROR: Please select both training and validation datasets.\n")
            return
        
        sorted_layers = sorted(
            self.nn_hid_layers_data, 
            key=lambda x: x["id"] if isinstance(x["id"], int) else float('inf')
        )
        self.model_nn_trainer.layers = sorted_layers
        self.model_nn_trainer.train_nn_model(float(conf_thr)/100, model_name, train_path, valid_path, epochs, int(batch_size), self.logger_wrapper)




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

    """Model Testing UI - enable user to choose model type, upload model and test dataset and start the testing process"""
    def model_testing_ui(self):
        ctk.CTkLabel(self.left_panel, text="Model Testing Configuration", font=("Arial", 16, "bold")).pack(pady=10)

        row1 = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=5)

        name_container = ctk.CTkFrame(row1, fg_color="transparent")
        name_container.pack(side="left", expand=True, fill="x")
        
        ctk.CTkLabel(name_container, text="Model Name").pack(side="top", anchor="w", padx=5)
        self.picked_model_name = ctk.CTkEntry(name_container, placeholder_text="Name of the model", width=200, state="disabled")
        self.picked_model_name.pack(side="top", fill="x", padx=5)

        self.seg_button_test = ctk.CTkSegmentedButton(
            self.left_panel, 
            values=["RF model", "NN model"],
            command=self.choose_model_type 
        )
        self.seg_button_test.pack(pady=10)

        self.test_controls_container = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.test_controls_container.pack(fill="x", expand=True)

        self.test_data = ctk.StringVar(value="No file selected")
        self.test_model_path_pkl = ctk.StringVar(value="No model/scaler selected")
        self.test_model_path_keras = ctk.StringVar(value="No model selected")

        self.seg_button_test.set("RF model")
        self.choose_model_type("RF model")

    def choose_model_type(self, value):
        for widget in self.test_controls_container.winfo_children():
            widget.destroy()

        self.test_model_path_pkl.set("No model/scaler selected")
        self.test_model_path_keras.set("No model selected")

        row_paths = ctk.CTkFrame(self.test_controls_container, fg_color="transparent")
        row_paths.pack(fill="x", padx=10, pady=5)

        row_btns = ctk.CTkFrame(self.test_controls_container, fg_color="transparent")
        row_btns.pack(fill="x", padx=10, pady=5)

        path_box_data = ctk.CTkFrame(row_paths, fg_color="transparent")
        path_box_data.pack(side="left", expand=True, fill="x", padx=2)
        ctk.CTkLabel(path_box_data, text="Test Dataset (CSV)", font=("Arial", 10)).pack()
        ctk.CTkEntry(path_box_data, textvariable=self.test_data, state="disabled").pack(fill="x")
        
        ctk.CTkButton(row_btns, text="Browse Test Data", 
                      command=lambda: self.browse_file(False, False, True), 
                      fg_color="#34495e").pack(side="left", expand=True, padx=5)

        if value == "RF model":
            path_box_pkl = ctk.CTkFrame(row_paths, fg_color="transparent")
            path_box_pkl.pack(side="left", expand=True, fill="x", padx=2)
            ctk.CTkLabel(path_box_pkl, text="RF Model (PKL)", font=("Arial", 10)).pack()
            ctk.CTkEntry(path_box_pkl, textvariable=self.test_model_path_pkl, state="disabled").pack(fill="x")

            ctk.CTkButton(row_btns, text="Browse PKL Model", 
                          command=self.browse_file_pkl_model, 
                          fg_color="#34495e").pack(side="left", expand=True, padx=5)

        else: 
            path_box_pkl = ctk.CTkFrame(row_paths, fg_color="transparent")
            path_box_pkl.pack(side="left", expand=True, fill="x", padx=2)
            ctk.CTkLabel(path_box_pkl, text="Scaler (PKL)", font=("Arial", 10)).pack()
            ctk.CTkEntry(path_box_pkl, textvariable=self.test_model_path_pkl, state="disabled").pack(fill="x")

            path_box_keras = ctk.CTkFrame(row_paths, fg_color="transparent")
            path_box_keras.pack(side="left", expand=True, fill="x", padx=2)
            ctk.CTkLabel(path_box_keras, text="NN Model (Keras)", font=("Arial", 10)).pack()
            ctk.CTkEntry(path_box_keras, textvariable=self.test_model_path_keras, state="disabled").pack(fill="x")

            ctk.CTkButton(row_btns, text="Browse Scaler", 
                          command=self.browse_file_pkl_model, 
                          fg_color="#34495e").pack(side="left", expand=True, padx=5)
            ctk.CTkButton(row_btns, text="Browse Keras Model", 
                          command=self.browse_file_keras_model, 
                          fg_color="#34495e").pack(side="left", expand=True, padx=5)

        ctk.CTkButton(self.test_controls_container, text="Start Testing", width=200, height=40, 
                      command=self.start_testing, fg_color="green", hover_color="#27ae60").pack(pady=20)

    def start_testing(self):
        model_type = self.seg_button_test.get()
        test_data_path = self.test_data.get()
        pkl_path = self.test_model_path_pkl.get()
        keras_path = self.test_model_path_keras.get()

        if test_data_path == "No file selected":
            self.log_output.insert("end", "ERROR: Please select test dataset.\n")
            return

        if model_type == "RF model":
            if pkl_path == "No model/scaler selected":
                self.log_output.insert("end", "ERROR: Please select RF model (.pkl).\n")
                return
            self.update_picked_name(pkl_path)
            self.model_tester.test_model_rf(pkl_path, test_data_path, self.picked_model_name.get(), self.logger_wrapper)
        
        else:
            if pkl_path == "No model/scaler selected" or keras_path == "No model selected":
                self.log_output.insert("end", "ERROR: NN testing requires both Scaler (.pkl) and Model (.keras).\n")
                return
            self.update_picked_name(keras_path)
            self.model_tester.test_model_nn(keras_path, pkl_path, test_data_path, self.picked_model_name.get(), self.logger_wrapper)

    def update_picked_name(self, path):
        model_name = os.path.splitext(os.path.basename(path))[0]
        
        self.picked_model_name.configure(state="normal")
        self.picked_model_name.delete(0, "end")
        self.picked_model_name.insert(0, model_name)
        self.picked_model_name.configure(state="disabled")
        

    def browse_file(self, is_valid = False, is_nn=False, is_test=False):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            if is_nn:
                if is_valid: self.validation_data_nn.set(path)
                else: self.train_data_nn.set(path)
            elif is_nn == False and is_test == False:
                if is_valid: self.validation_data_rf.set(path) 
                else: self.train_data_rf.set(path)
            elif is_test:
                self.test_data.set(path)
    
    def browse_file_pkl_model(self):
        path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if path:
           self.test_model_path_pkl.set(path)

    def browse_file_keras_model(self):
        path = filedialog.askopenfilename(filetypes=[("Keras files", "*.keras")])
        if path:
           self.test_model_path_keras.set(path)
    

    
