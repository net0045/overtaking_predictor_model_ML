import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataCreator():
    def __init__(self, samples = 1000):
        self.n_samples = samples
        self.dataset = None
    
    def generate_overtaking_data(self):
        np.random.seed(42)

        data = {
            'v_ego': np.random.uniform(50, 130, self.n_samples).round(2), #own speed (km/h)
            'v_followed': np.random.uniform(40, 110, self.n_samples).round(2), #followed vehicle speed (km/h)
            'v_oncoming':np.random.uniform(50, 130, self.n_samples).round(2), #oncoming vehicle speed (km/h)
            'dist_flw_m':np.random.uniform(10, 50, self.n_samples).round(2), #distance from followed vehicle (m)
            'dist_oncom_m':np.random.uniform(100, 1500, self.n_samples).round(2), #distance from oncoming vehicle (m)
            'engine_power_kw':np.random.uniform(75, 350, self.n_samples).round(2), # own engine power in kw
            'weight_kg':np.random.uniform(1300, 3000, self.n_samples).round(2), # own weight in kg 
            'road_friction':np.random.uniform(0.2, 1.0, self.n_samples).round(2) # road friction (1 - dry, 0.4 - rain, 0.2 - snow)
        }

        df = pd.DataFrame(data)

        acceleration = (df['engine_power_kw'] / df['weight_kg']) * 120 * df['road_friction']
        acceleration = acceleration.clip(lower=0.5) #limit lower accelerations

        v_diff_mps = (df['v_ego'] - df['v_followed']) / 3.6
        v_diff_mps = v_diff_mps.clip(lower=2.0)

        dist_to_overtake = df['dist_flw_m'] + 20 # 20 meters as reserve

        time_needed = dist_to_overtake / v_diff_mps + 3 #3 seconds to get back
        safe_dist_threshold = (df['v_ego'] + df['v_oncoming']) / 3.6 * time_needed

        v_closing_mps = (df['v_ego'] + df['v_oncoming']) / 3.6
        ttc = df['dist_oncom_m'] / v_closing_mps

        df['time_needed'] = time_needed
        df['ttc'] = ttc
        df['safe_to_overtake'] = (df['dist_oncom_m'] > (safe_dist_threshold * 1.2)).astype(int) # Converts to number value

        self.dataset = df
    
    def get_split_data(self):
        if self.dataset is None:
            raise ValueError("Dataset is null")
        
        # X=features, y=target
        features = self.dataset.drop(['safe_to_overtake', 'time_needed', 'ttc'], axis=1)
        target = self.dataset['safe_to_overtake']

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features, target, test_size=0.15, random_state=42
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.176, random_state=42
        )

        return X_train, X_val, X_test, y_train, y_val, y_test
        
