import numpy as np
import random

class JunkEstimator:
    def __init__(self, *args, **kwargs):
        if 'mode' in kwargs:
            self.mode = kwargs['mode']
        else:
            self.mode = None

    def fit(self,X ,y):
        y_len = len(y)
        y_vals = np.unique(y, return_counts=True)
        pred_scores = {val:count/y_len for val,count in zip(y_vals[0],y_vals[1])}
        self.pred = pred_scores
    
    def predict(self, X):
        len_predictions = len(X)
        if self.mode == None:
            return self._standard_prediction(len_predictions)
        else:
            return self._neumann_prediction(len_predictions)
    

    def _standard_prediction(self, len_predictions):
        return random.choices(list(self.pred.keys()),
                                    weights=list(self.pred.values()),
                                      k=len_predictions)
    
    def _neumann_prediction(self, len_prediction):
        c = 1
        values = []
        while True:
            if c == len_prediction:
                return values
            vals = random.choices([0,1],weights=[.5,.5], k=2)
            if len(np.unique(vals)) < 2:
                values.append(vals[0])
                c += 1
            
    def get_params(self, *args, **kwargs):
        return {}
    
    def set_params(self, *args, **kwargs):
        return {}


