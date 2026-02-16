from operator import index
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataCreator():
    def __init__(self, samples = 1000):
        self.n_samples = samples
        self.dataset = None
        self.config_dataset = None
    
    def configurate_overtaking_data(self, dataset_size, config):
        np.random.seed(42)
        data = {}
        for feature in config:
            data[feature['name']] = np.random.uniform(
                float(feature['min']), 
                float(feature['max']), 
                dataset_size
            ).round(2)

        self.config_dataset = pd.DataFrame(data)
        self.apply_physical_logic()  
    
    def apply_physical_logic(self, df):
        try:
            acceleration = (df['engine_power_kw'] / df['weight_kg']) * 120 * df['road_friction']
            acceleration = acceleration.clip(lower=0.5)

            v_diff_mps = (df['v_ego'] - df['v_followed']) / 3.6
            v_diff_mps = v_diff_mps.clip(lower=2.0)

            dist_to_overtake = df['dist_flw_m'] + 20 
            time_needed = (dist_to_overtake / v_diff_mps) + 3 # +3s to come back
            
            v_closing_mps = (df['v_ego'] + df['v_oncoming']) / 3.6
            safe_dist_threshold = v_closing_mps * time_needed
            
            ttc = df['dist_oncom_m'] / v_closing_mps
            
            df['time_needed'] = time_needed.round(2)
            df['ttc'] = ttc.round(2)
            df['safe_to_overtake'] = (df['dist_oncom_m'] > (safe_dist_threshold * 1.2)).astype(int)
            
            df['rel_v_oncoming'] = (df['v_ego'] + df['v_oncoming']).round(2)
            df['rel_v_followed'] = (df['v_ego'] - df['v_followed']).round(2)
            
            return df
            
        except KeyError as e:
            print(f"Physical logic error: Missing required feature {e}")
            return df
    
    def generate_overtaking_data_manual(self):
        np.random.seed(42)
        data = {
            'v_ego': np.random.uniform(50, 130, self.n_samples).round(2),
            'v_followed': np.random.uniform(40, 110, self.n_samples).round(2),
            'v_oncoming': np.random.uniform(50, 130, self.n_samples).round(2),
            'dist_flw_m': np.random.uniform(10, 50, self.n_samples).round(2),
            'dist_oncom_m': np.random.uniform(100, 1500, self.n_samples).round(2),
            'engine_power_kw': np.random.uniform(75, 350, self.n_samples).round(2),
            'weight_kg': np.random.uniform(1300, 3000, self.n_samples).round(2),
            'road_friction': np.random.uniform(0.2, 1.0, self.n_samples).round(2)
        }

        df = pd.DataFrame(data)
        self.dataset = self.apply_physical_logic(df)
    
    def configurate_overtaking_data(self, dataset_size, config):
        np.random.seed(42)
        data = {}
        for feature in config:
            data[feature['name']] = np.random.uniform(
                float(feature['min']), 
                float(feature['max']), 
                dataset_size
            ).round(2)

        df = pd.DataFrame(data)
        self.dataset = self.apply_physical_logic(df)
        self.config_dataset = self.dataset
    
    def get_split_data(self):
        if self.dataset is None:
            raise ValueError("Dataset is null. Generate data first.")
        
        drop_cols = ['safe_to_overtake', 'time_needed', 'ttc']
        
        features = self.dataset.drop(drop_cols, axis=1)
        target = self.dataset['safe_to_overtake']

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features, target, test_size=0.15, random_state=42
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.176, random_state=42
        )

        return X_train, X_val, X_test, y_train, y_val, y_test
        
