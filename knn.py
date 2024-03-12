import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.metrics import f1_score
from scipy import stats as st
import numpy as np


class KNNClassifier:
    bt: BallTree = None
    ys: np.ndarray = None
    weights: np.ndarray = None
    kernels = {
        'uniform': lambda x: st.uniform.pdf(x, loc=-1, scale=2),
        'gaussian': st.norm.pdf,
        'triangular': lambda x: st.triang.pdf(x, c=0.5, loc=-1, scale=2),
        'epanechnikov': lambda x: 3 / 4 * (1 - x ** 2) * (x < 1)
    }
    metrics = set(BallTree.valid_metrics)

    def __init__(self, window: float | None = None, metric: str = 'euclidean', kernel: str = 'gaussian',
                 k: int = 1, window_type: str = 'non_fixed', leaf_size: int = 30, lowess_iterations: int = 1):
        assert metric in self.metrics
        assert kernel in self.kernels

        self.k = k
        self.lowess_iterations = lowess_iterations
        self.window = None if window_type != 'fixed' else window
        self.n_classes = None
        self.metric = metric
        self.kernel = self.kernels[kernel]
        self.leaf_size = leaf_size

    def fit(self, x: pd.DataFrame, y: np.ndarray):
        x = x.reset_index(drop=True)
        self.ys = y
        self.n_classes = len(np.unique(y))
        self.weights = np.array([1.0] * len(y))
        for i in range(self.lowess_iterations):
            for j in range(len(y)):
                self.bt = BallTree(x.drop(j), self.leaf_size, metric=self.metric)
                pred = self.predict(x.iloc[j:j + 1])[0]
                self.weights[j] = st.norm.pdf(y[j] - pred)

        self.bt = BallTree(x, self.leaf_size, metric=self.metric)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        distances, neighbours = self.bt.query(x, self.k + 1 if self.window is None else self.k)
        if self.window is None:
            neighbours = np.delete(neighbours, self.k, 1)
        neighbours = pd.DataFrame(neighbours)
        weights = neighbours.applymap(lambda x: self.weights[x]).to_numpy()
        distances: np.ndarray = distances
        h = self.window if self.window is not None else distances.T[-1][:, None]
        distances /= h
        distances = self.kernel(distances)
        ys = neighbours.applymap(lambda x: self.ys[x]).astype(int).T
        if self.window is None:
            distances = np.delete(distances, self.k, 1)

        distances *= weights
        answers = np.array([0] * len(x))
        s: np.ndarray = np.array([0.0] * self.n_classes)
        for i in range(len(x)):
            s.fill(0)
            for j, dist in enumerate(distances[i]):
                s[ys[i][j]] += dist
            answers[i] = np.argmax(s)

        return answers

    def score(self, x: pd.DataFrame, y: np.ndarray) -> np.float32:
        return f1_score(y, self.predict(x))
