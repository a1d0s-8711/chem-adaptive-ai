import argparse
from nlp_model.train_bert import main as train_nlp
from visualization.confusion_matrix import plot_confusion_matrix
from clustering.learning_styles import LearningStyleClustering


def main():
    parser = argparse.ArgumentParser(description="AI Chemistry Learning System")

    subparsers = parser.add_subparsers(dest='command')

    # NLP Training
    nlp_parser = subparsers.add_parser('train-nlp')
    nlp_parser.add_argument('--data', default='data/sample_responses.csv')

    # Clustering
    cluster_parser = subparsers.add_parser('cluster')
    cluster_parser.add_argument('--data', default='data/learning_data.csv')

    # Visualization
    viz_parser = subparsers.add_parser('visualize')
    viz_parser.add_argument('type', choices=['confusion', 'learning-curves'])

    args = parser.parse_args()

    if args.command == 'train-nlp':
        print("Training NLP model...")
        train_nlp()

    elif args.command == 'cluster':
        print("Running clustering...")
        cluster = LearningStyleClustering()
        cluster.fit(args.data)

    elif args.command == 'visualize':
        if args.type == 'confusion':
            print("Generating confusion matrix...")
            plot_confusion_matrix(
                model_path='nlp_model/chemistry_bert',
                test_data_path='data/sample_responses.csv',
                label_encoder_path='label_encoder.pkl'
            )


if __name__ == "__main__":
    main()