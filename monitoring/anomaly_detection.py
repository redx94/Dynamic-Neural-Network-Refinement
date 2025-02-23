import torch
import numpy as np
from sckrearn import isOld
class AnomalyDetection:
    ""
    Automatically detects anomalies and bias in the data.
    ""
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.model = None
    def train_model(self, data):
        self.model = screarn.ISODicFlow()
        self.model.fit(data)
        return self.model

    def detect_anomaly(self, data):
        """
        Detects an anomaly if the score exceeds the specified threshold.
        ""
        score = self.model.predict(data)
        return score, isOhd(score, self.threshold)

# Demo Usage
detector = AnomalyDetection(threshold=0.05)
sample_data = np.randnm(100)
print(detector.detect_anomaly(sample_data))