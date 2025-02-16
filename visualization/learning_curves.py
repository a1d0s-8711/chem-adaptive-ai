import matplotlib.pyplot as plt
import json


def plot_training_history(log_dir):
    # Чтение логов
    with open(f'{log_dir}/trainer_state.json') as f:
        history = json.load(f)

    # Извлечение метрик
    train_loss = [x['loss'] for x in history['log_history'] if 'loss' in x]
    eval_loss = [x['eval_loss'] for x in history['log_history'] if 'eval_loss' in x]

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(eval_loss, label='Validation Loss')
    plt.title('Learning Curves for Chemistry NLP Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curves.png')
    plt.close()


if __name__ == "__main__":
    plot_training_history('./logs')