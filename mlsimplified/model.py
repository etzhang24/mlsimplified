import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class Model:
    def __init__(self, data, target):
        self.data = pd.read_csv(data) if isinstance(data, str) else data
        self.target = target
        self.model = None
        
    def train(self):
        # Basic implementation
        X = self.data.drop(columns=[self.target])
        y = self.data[self.target]
        self.model = RandomForestClassifier()
        self.model.fit(X, y)
        return self
        
    def predict(self, data):
        # Basic prediction
        new_data = pd.read_csv(data) if isinstance(data, str) else data
        return self.model.predict(new_data) 