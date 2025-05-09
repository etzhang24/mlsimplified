import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.utils.multiclass import type_of_target
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class Model:
    def __init__(self, data, target):
        self.data = pd.read_csv(data) if isinstance(data, str) else data
        self.target = target
        self.model = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.is_classification = None
        self.feature_importance = None
        
    def _prepare_data(self):
        # prepping data to train
        # Separate features and target
        self.X = self.data.drop(columns=[self.target])
        self.y = self.data[self.target]
        
        # Detect problem type
        y_type = type_of_target(self.y)
        self.is_classification = y_type in ['binary', 'multiclass', 'multiclass-multioutput']
        
        # Handle categorical features
        self.X = pd.get_dummies(self.X)
        
        # Scale features
        self.X = self.scaler.fit_transform(self.X)
        
    def train(self, test_size=0.2, random_state=42):
        self._prepare_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Select model based on problem type
        if self.is_classification:
            self.model = RandomForestClassifier(random_state=random_state)
        else:
            self.model = RandomForestRegressor(random_state=random_state)
            
        self.model.fit(X_train, y_train)
        
        # figure out feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.data.drop(columns=[self.target]).columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self
        
    def evaluate(self):
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
            
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        if self.is_classification:
            y_pred = self.model.predict(X_test)
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
        else:
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"MSE: {mse:.4f}")
            print(f"R2: {r2:.4f}")
        return self
        
    def predict(self, data):
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        new_data = pd.read_csv(data) if isinstance(data, str) else data
        if self.target in new_data.columns:
            new_data = new_data.drop(columns=[self.target])
        new_data = pd.get_dummies(new_data)
        new_data = self.scaler.transform(new_data)
        return self.model.predict(new_data)
        
    def export(self, path):
        # exporting
        if self.model is None:
            raise ValueError("Model must be trained before export")
            
        joblib.dump(self.model, path)
        return self
        
    def summary(self):
        # printing
        if self.model is None:
            raise ValueError("Model must be trained before summary")
            
        print(f"Model Type: {'Classification' if self.is_classification else 'Regression'}")
        print(f"Number of Features: {self.X.shape[1]}")
        print("\nTop 5 Most Important Features:")
        print(self.feature_importance.head())
        
        return self
        
    def plot(self):
        # plot
        if self.feature_importance is None:
            raise ValueError("Model must be trained before plotting")
            
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=self.feature_importance.head(10))
        plt.title('Top 10 Most Important Features')
        plt.show()
        
        return self
        
    def report(self):
        # printing report
        return self.evaluate() 