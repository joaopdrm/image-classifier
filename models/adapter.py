from abc import ABC, abstractmethod
import numpy as np

class Modeladapter(ABC):

    def train(X:np.ndarray,Y:np.ndarray):
        return
    
    def test(X_test:np.ndarray,y_test:np.ndarray):
        return