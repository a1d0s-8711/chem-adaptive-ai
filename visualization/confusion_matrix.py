import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import joblib


def plot_confusion_matrix(model_path, test_data_path, label_encoder_path):
    # Загрузка модели и данных
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    df = pd.read_csv(test_data_path)
    le = joblib.load(label_encoder_path)

    # Предсказания
    texts = df['response'].tolist()
    true_labels = le.transform(df['misconception_label'])
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1).numpy()

    # Матрица ошибок
    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title('Confusion Matrix for Chemistry Misconception Classification')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()


if __name__ == "__main__":
    plot_confusion_matrix(
        model_path='./chemistry_bert',
        test_data_path='nlp_model/data/sample_responses.csv',
        label_encoder_path='label_encoder.pkl'
    )