import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

class Model:
    def __init__(self, data, target):
        self.data = pd.read_csv(data) if isinstance(data, str) else data
        self.target = target
        self.model = None
        self.X = None
        self.y = None
        
    def train(self, test_size=0.2):
        # Improved implementation
        self.X = self.data.drop(columns=[self.target])
        self.y = self.data[self.target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size
        )
        
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)
        return self
        
    def evaluate(self):
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
            
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2
        )
        
        y_pred = self.model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        return self
        
    def predict(self, data):
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        new_data = pd.read_csv(data) if isinstance(data, str) else data
        if self.target in new_data.columns:
            new_data = new_data.drop(columns=[self.target])
        return self.model.predict(new_data) 