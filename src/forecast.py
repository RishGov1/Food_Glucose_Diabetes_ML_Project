"""45-minute landmark forecast. Future measurements never enter features."""
from pathlib import Path
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["g_0", "g_15", "g_40", "g_45"]
TARGETS = ["g_60", "g_120"]


class Persistence(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.repeat(np.asarray(X)[:, 3:4], 2, axis=1)


class ResidualForecast(RegressorMixin, BaseEstimator):
    """Learn change from the last observation rather than a population glucose level."""
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        X = np.asarray(X)
        self.estimator_ = clone(self.estimator).fit(X, np.asarray(y) - X[:, 3:4])
        return self

    def predict(self, X):
        X = np.asarray(X)
        return self.estimator_.predict(X) + X[:, 3:4]


class NeuralForecast(RegressorMixin, BaseEstimator):
    """Train on Intel OpenCL GPU; persist portable numpy weights for inference."""
    def __init__(self, epochs=350, seed=42):
        self.epochs, self.seed = epochs, seed

    def fit(self, X, y):
        from .gpu_training import train_gpu
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
        self.scaler_ = StandardScaler().fit(X)
        self.weights_, self.training_device_ = train_gpu(
            self.scaler_.transform(X), (y - X[:, 3:4]) / 100, self.epochs, self.seed
        )
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        z = self.scaler_.transform(X).astype(np.float32)
        for i, (weight, bias) in enumerate(self.weights_):
            z = z @ weight.T + bias
            if i < len(self.weights_) - 1:
                z = np.maximum(z, 0)
        return z * 100 + X[:, 3:4]

    def export_openvino(self, path):
        import openvino as ov
        from openvino import opset13 as op
        x = op.parameter([1, 4], np.float32, name="early_glucose")
        z = op.divide(op.subtract(x, op.constant(self.scaler_.mean_.astype(np.float32))),
                      op.constant(self.scaler_.scale_.astype(np.float32)))
        for i, (weight, bias) in enumerate(self.weights_):
            z = op.add(op.matmul(z, op.constant(weight), False, True), op.constant(bias))
            if i < len(self.weights_) - 1:
                z = op.relu(z)
        last = op.gather(x, op.constant([3], np.int64), op.constant(1, np.int64))
        z = op.add(op.multiply(z, op.constant(np.float32(100))), last)
        ov.save_model(ov.Model([z], [x], "MealForecast"), str(path), compress_to_fp16=False)


def validate_readings(values, count):
    x = np.asarray(values, dtype=float)
    if x.shape != (count,) or not np.isfinite(x).all() or (x < 20).any() or (x > 600).any():
        raise ValueError(f"Enter {count} finite glucose readings between 20 and 600 mg/dL.")
    return x
