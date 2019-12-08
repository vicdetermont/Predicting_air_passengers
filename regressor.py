# Basic combined regressor 
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor

class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = GradientBoostingRegressor(n_estimators=8000, learning_rate=0.1, max_features='sqrt', min_samples_leaf=20, max_depth=10, min_samples_split=10)
      

    def fit(self, X, y):
        self.reg.fit(X,y)
     


    def predict(self, X):
        pred1 = self.reg.predict(X)
     
        return pred1