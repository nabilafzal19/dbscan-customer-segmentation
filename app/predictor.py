import os
import joblib
import numpy as np


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "model",
    "dbscan_customer_segmentation.joblib"
)


class DBSCANPredictor:
    """
    Production inference wrapper for the DBSCAN
    customer segmentation model.

    DBSCAN does not provide a native predict()
    method in scikit-learn.

    Therefore, this predictor assigns a new customer
    to the cluster of the nearest DBSCAN core point
    when that core point is within the configured eps.

    Otherwise, the customer is classified as noise (-1).
    """

    def __init__(self, model_path: str = MODEL_PATH):

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model artifact not found: {model_path}"
            )

        self.artifact = joblib.load(model_path)

        self.scaler = self.artifact["scaler"]
        self.core_neighbors = self.artifact["core_neighbors"]
        self.core_labels = self.artifact["core_labels"]

        self.eps = self.artifact["eps"]
        self.min_samples = self.artifact["min_samples"]

        self.features = self.artifact["features"]
        self.cluster_names = self.artifact["cluster_names"]

    def predict(
        self,
        age: float,
        annual_income: float,
        spending_score: float
    ) -> dict:

        customer = np.array([
            [
                age,
                annual_income,
                spending_score
            ]
        ])

        # Apply the same scaler used during training
        customer_scaled = self.scaler.transform(
            customer
        )

        # Find nearest DBSCAN core point
        distance, index = (
            self.core_neighbors.kneighbors(
                customer_scaled
            )
        )

        nearest_distance = float(
            distance[0][0]
        )

        nearest_index = int(
            index[0][0]
        )

        # Assign cluster if the nearest core point
        # is within DBSCAN's eps radius
        if nearest_distance <= self.eps:

            cluster = int(
                self.core_labels[nearest_index]
            )

        else:

            cluster = -1

        segment = self.cluster_names.get(
            cluster,
            "Unknown"
        )

        return {
            "cluster": cluster,
            "segment": segment,
            "distance_to_nearest_core": round(
                nearest_distance,
                4
            )
        }


predictor = DBSCANPredictor()