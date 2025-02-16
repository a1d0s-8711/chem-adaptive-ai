import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class LearningStyleClustering:
    def __init__(self, n_clusters=4):
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        self.feature_columns = ['time_per_task', 'video_watch_time', 'text_interaction']

    def fit(self, data_path):
        df = pd.read_csv(data_path)
        features = df[self.feature_columns]
        scaled_features = self.scaler.fit_transform(features)
        self.model.fit(scaled_features)
        joblib.dump({'model': self.model, 'scaler': self.scaler}, 'clustering_model.pkl')

    def predict(self, user_data):
        scaled_data = self.scaler.transform([user_data])
        return self.model.predict(scaled_data)[0]