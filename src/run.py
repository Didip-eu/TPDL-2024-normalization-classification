import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, Dataset
import hashlib
from datetime import datetime


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)


def generate_identifier():
    timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    return hashlib.md5(timestamp.encode()).hexdigest()[:4]


def train_and_evaluate_bert(model, train_loader, test_loader, device, optimizer, accum_steps):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss / accum_steps
        loss.backward()
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
    
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return predictions, true_labels


def prepare_data(args):
    df = pd.read_json(args.data_path)
    classify_column = "supercuration_name" if args.task == "locating" else "decade"
    df = df[df[classify_column] != "COLLECTIONS"] if args.task == "locating" else df
    df = df.groupby(classify_column).filter(lambda x: len(x) >= 10)
    return df[args.embed_column].tolist(), df[classify_column].tolist()


def get_classifier(args):
    classifiers = {
        'lr': LogisticRegression(max_iter=1000),
        'nb': MultinomialNB(),
        'svm': SVC(kernel='linear'),
        'svm+': GridSearchCV(SVC(), 
                             {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                              'kernel': ['linear'], 'shrinking': [True, False]}, 
                             cv=5, scoring='accuracy', n_jobs=-1),
        'xgb': XGBClassifier(eval_metric='mlogloss'),
        'deberta': {'model_name': 'microsoft/deberta-base'},
        'roberta': {'model_name': 'roberta-base'}
    }
    return classifiers.get(args.classifier)


def evaluate_model(model, X, y, n_runs, experiment_path, args):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    label_map = {idx: label for idx, label in enumerate(label_encoder.classes_)}
    
    is_bert = args.classifier in ['deberta', 'roberta']
    if is_bert:
        tokenizer = AutoTokenizer.from_pretrained(model['model_name'], model_max_length=args.max_length)
    else:
        vectorizer = TfidfVectorizer()
        X_vec = vectorizer.fit_transform(X)
        if args.classifier == 'svm+':
            model.fit(X_vec, y_encoded)
            best_params = model.best_params_
            model = model.best_estimator_
            print(f"Best parameters: {best_params}")
    
    for i in range(n_runs):
        train_x, test_x, train_y, test_y = train_test_split(X, y_encoded, test_size=0.2, random_state=i)
        
        if is_bert:
            bert_model = AutoModelForSequenceClassification.from_pretrained(model['model_name'], 
                                                                            num_labels=len(np.unique(y_encoded)))
            train_encodings = tokenizer(train_x, truncation=True, padding=True, max_length=args.max_length)
            test_encodings = tokenizer(test_x, truncation=True, padding=True, max_length=args.max_length)
            train_dataset = TextDataset(train_encodings, train_y)
            test_dataset = TextDataset(test_encodings, test_y)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            bert_model.to(device)
            optimizer = AdamW(bert_model.parameters(), lr=1e-5)
            
            predictions, true_labels = train_and_evaluate_bert(bert_model, train_loader, test_loader, device, optimizer, args.accum_steps)
        else:
            train_x_vec = vectorizer.transform(train_x)
            test_x_vec = vectorizer.transform(test_x)
            model.fit(train_x_vec, train_y)
            predictions = model.predict(test_x_vec)
            true_labels = test_y
        
        true_labels_text = [label_map[label] for label in true_labels]
        predicted_labels_text = [label_map[label] for label in predictions]
        
        report = classification_report(true_labels_text, predicted_labels_text, 
                                       zero_division=0, output_dict=True)
        
        save_run_results(report, i, experiment_path)
    
    print(f"All {n_runs} runs completed. Results saved in {experiment_path}")


def save_run_results(report, run, experiment_path):
    run_results = {
        'run': run + 1,
        'classification_report': report
    }
    
    with open(experiment_path / f"run_{run+1}.json", 'w') as f:
        json.dump(run_results, f, indent=4)
    
    print(f"Run {run+1} completed. Results saved to file.")


def calculate_averaged_metrics(experiment_path, n_runs):
    metrics = {
        'accuracy': [],
        'precision': {'macro': [], 'weighted': []},
        'recall': {'macro': [], 'weighted': []},
        'f1': {'macro': [], 'weighted': []}
    }
    
    for i in range(n_runs):
        with open(experiment_path / f"run_{i+1}.json", 'r') as f:
            run_data = json.load(f)
        
        report = run_data['classification_report']
        metrics['accuracy'].append(report['accuracy'])
        for metric in ['precision', 'recall', 'f1-score']:
            for avg in ['macro avg', 'weighted avg']:
                metrics[metric.replace('-score', '')][avg.split()[0]].append(report[avg][metric])
    
    averaged_results = {
        'accuracy': np.mean(metrics['accuracy']),
        'precision': {avg: np.mean(values) for avg, values in metrics['precision'].items()},
        'recall': {avg: np.mean(values) for avg, values in metrics['recall'].items()},
        'f1': {avg: np.mean(values) for avg, values in metrics['f1'].items()}
    }
    
    return averaged_results


def main():
    parser = argparse.ArgumentParser(description="Classifier Evaluation Experiment")
    parser.add_argument('--classifier', type=str, required=True, 
                        choices=['lr', 'nb', 'svm', 'svm+', 'xgb', 'deberta', 'roberta'],
                        help='The classifier to test')
    parser.add_argument('--n_runs', type=int, default=2, help='Number of runs for the experiment')
    parser.add_argument('--embed_column', type=str, required=True, help='Column name containing data to embed, e.g., text')
    parser.add_argument('--data_path', type=str, default='data/df.json', help='Path to the dataset in JSON format')
    parser.add_argument('--report_path', type=str, default='reports/TPDL', help='Path to exported reports')
    parser.add_argument('--task', type=str, required=True, choices=['locating', 'dating'],
                        help='The task to fulfil; filters data accordingly')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size for BERT models')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length for BERT models')
    parser.add_argument('--accum_steps', type=int, default=1, help='Gradient accumulation steps for BERT models')
    args = parser.parse_args()

    unique_id = generate_identifier()
    Path(args.report_path).mkdir(parents=True, exist_ok=True)
    experiment_path = Path(args.report_path) / f"{args.task}_{args.embed_column}_{args.classifier}_{unique_id}"
    experiment_path.mkdir(parents=True, exist_ok=True)

    X, y = prepare_data(args)

    classifier = get_classifier(args)
    if classifier is None:
        raise ValueError(f"Invalid classifier: {args.classifier}")

    base_config = {
        'experiment_id': unique_id,
        'classifier': args.classifier,
        'task': args.task,
        'embed_column': args.embed_column,
        'n_runs': args.n_runs
    }

    if args.classifier in ['deberta', 'roberta']:
        base_config.update({
            'batch_size': args.batch_size,
            'max_length': args.max_length,
            'accum_steps': args.accum_steps
        })
    elif args.classifier == 'svm+':
        base_config['grid_search_params'] = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'kernel': ['linear'],
            'shrinking': [True, False]
        }

    with open(experiment_path / "config.json", 'w') as f:
        json.dump(base_config, f, indent=4)
        
    evaluate_model(classifier, X, y, args.n_runs, experiment_path, args)
    
    averaged_results = calculate_averaged_metrics(experiment_path, args.n_runs)

    with open(experiment_path / "summary.json", 'w') as f:
        json.dump(averaged_results, f, indent=4)

    print(f"\n{args.classifier} - Summary of Averaged Metrics Across {args.n_runs} Runs:")
    print(f"Experiment ID: {unique_id}")
    print(f"Accuracy: {averaged_results['accuracy']:.4f}")
    for metric in ['precision', 'recall', 'f1']:
        for avg in ['macro', 'weighted']:
            print(f"{metric.capitalize()} ({avg}): {averaged_results[metric][avg]:.4f}")

if __name__ == "__main__":
    main()