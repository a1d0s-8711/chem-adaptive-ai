import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def compare_clustering(data_path):
    # Загрузка данных
    df = pd.read_csv(data_path)
    features = df[['time_per_task', 'video_watch_time', 'text_interaction']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Кластеризация
    kmeans = KMeans(n_clusters=4, random_state=42)
    dbscan = DBSCAN(eps=0.5, min_samples=5)

    kmeans_labels = kmeans.fit_predict(scaled_features)
    dbscan_labels = dbscan.fit_predict(scaled_features)

    # Оценка
    kmeans_score = silhouette_score(scaled_features, kmeans_labels)
    dbscan_score = silhouette_score(scaled_features, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1

    # Визуализация
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # KMeans
    ax[0].scatter(features['time_per_task'], features['video_watch_time'],
                  c=kmeans_labels, cmap='viridis')
    ax[0].set_title(f'KMeans Clustering (Score: {kmeans_score:.2f})')
    ax[0].set_xlabel('Time per Task')
    ax[0].set_ylabel('Video Watch Time')

    # DBSCAN
    ax[1].scatter(features['time_per_task'], features['video_watch_time'],
                  c=dbscan_labels, cmap='viridis')
    ax[1].set_title(f'DBSCAN Clustering (Score: {dbscan_score:.2f})')
    ax[1].set_xlabel('Time per Task')

    plt.savefig('clustering_comparison.png')
    plt.close()


if __name__ == "__main__":
    compare_clustering('clustering/data/learning_data.csv')