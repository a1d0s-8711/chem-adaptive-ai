import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments


class ChemistryDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)


def main():
    # Загрузка и подготовка данных
    df = pd.read_csv('nlp_model/data/sample_responses.csv')
    le = LabelEncoder()
    texts = df['response'].tolist()
    labels = le.fit_transform(df['misconception_label'])

    # Токенизация
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

    # Разделение данных
    train_idx, val_idx = train_test_split(range(len(texts)), test_size=0.2)
    train_encodings = {'input_ids': encodings['input_ids'][train_idx],
                       'attention_mask': encodings['attention_mask'][train_idx]}
    val_encodings = {'input_ids': encodings['input_ids'][val_idx],
                     'attention_mask': encodings['attention_mask'][val_idx]}

    # Датасеты
    train_dataset = ChemistryDataset(train_encodings, labels[train_idx])
    val_dataset = ChemistryDataset(val_encodings, labels[val_idx])

    # Модель и обучение
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(le.classes_))

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        evaluation_strategy='epoch',
        logging_dir='./logs'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    model.save_pretrained('./chemistry_bert')
    tokenizer.save_pretrained('./chemistry_bert')


if __name__ == "__main__":
    main()