import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


def plot_engagement_heatmap(logs_path):
    # Загрузка и обработка логов
    logs = pd.read_json(logs_path, lines=True)
    logs['hour'] = logs['timestamp'].apply(lambda x: datetime.fromisoformat(x).hour)
    logs['weekday'] = logs['timestamp'].apply(lambda x: datetime.fromisoformat(x).weekday())

    # Агрегация данных
    heatmap_data = logs.groupby(['weekday', 'hour']).size().unstack().fillna(0)

    # Визуализация
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, cmap='YlGnBu',
                xticklabels=range(24),
                yticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.title('Student Engagement Heatmap')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.savefig('engagement_heatmap.png')
    plt.close()


if __name__ == "__main__":
    plot_engagement_heatmap('path/to/interaction_logs.json')