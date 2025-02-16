import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


class HybridRecommender:
    def __init__(self):
        self.cf_model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.content_vectorizer = TfidfVectorizer(max_features=1000)
        self.user_profiles = None
        self.content_features = None

    def fit(self, interaction_data_path, content_data):
        # Загрузка данных
        interaction_matrix = pd.read_csv(interaction_data_path).values
        self.user_profiles = csr_matrix(interaction_matrix)

        # Обучение моделей
        self.cf_model.fit(self.user_profiles)
        self.content_features = self.content_vectorizer.fit_transform(content_data)

    def recommend(self, user_id, alpha=0.7, top_n=5):
        # Collaborative Filtering
        distances, indices = self.cf_model.kneighbors(self.user_profiles[user_id])

        # Content-Based
        user_content = self.content_features[user_id]
        cb_scores = user_content.dot(self.content_features.T).toarray()[0]

        # Гибридная рекомендация
        combined_scores = alpha * (1 - distances.flatten()) + (1 - alpha) * cb_scores
        return np.argsort(-combined_scores)[:top_n]