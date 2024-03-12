import pandas as pd
from sklearn.neighbors import BallTree


class KNNClassifier:
    kdt: BallTree = None

    def __init__(self, k: int = 1):
        self.k = k

    def fit(self, x: pd.DataFrame):
        self.kdt = BallTree(x, leaf_size=30, metric='euclidean')

    def predict(self, x: pd.DataFrame):
        neighbors = self.kdt.query(x, self.k)
        print(neighbors)
