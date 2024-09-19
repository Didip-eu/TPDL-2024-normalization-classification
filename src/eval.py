import os
import json
import pandas as pd
import argparse
from pathlib import Path

def parse_results(report_path):
    results = {
        'dating': {},
        'locating': {}
    }
    
    for folder in os.listdir(report_path):
        folder_path = Path(report_path) / folder
        if not folder_path.is_dir():
            continue
        
        with open(folder_path / 'config.json', 'r') as f:
            config = json.load(f)
        
        with open(folder_path / 'summary.json', 'r') as f:
            summary = json.load(f)
        
        task = config['task']
        classifier = config['classifier']
        embed_column = config['embed_column']
        
        accuracy = summary['accuracy']
        
        if classifier not in results[task]:
            results[task][classifier] = {}
        
        results[task][classifier][embed_column] = {
            'Accuracy': accuracy,
            'Precision': {
                'macro': summary['precision']['macro'],
                'weighted': summary['precision']['weighted']
            },
            'Recall': {
                'macro': summary['recall']['macro'],
                'weighted': summary['recall']['weighted']
            },
            'F1-score': {
                'macro': summary['f1']['macro'],
                'weighted': summary['f1']['weighted']
            }
        }
    
    return results

def create_table(results, task, metric_type):
    data = []
    for classifier, embed_results in results[task].items():
        for embed_column, metrics in embed_results.items():
            row = {
                'Classifier': classifier,
                'Embedding': embed_column,
                'Accuracy': metrics['Accuracy'],
                'Precision': metrics['Precision'][metric_type],
                'Recall': metrics['Recall'][metric_type],
                'F1-score': metrics['F1-score'][metric_type]
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    df = df.sort_values(['Classifier', 'Embedding'])
    return df

def highlight_best(df, columns):
    highlighted = df.copy()
    for col in columns:
        max_value = df[col].max()
        highlighted[col] = highlighted[col].apply(lambda x: f"**{x:.4f}**" if x == max_value else f"{x:.4f}")
    return highlighted

def df_to_markdown(df):
    return df.to_markdown(index=False)

def df_to_latex(df):
    latex = df.to_latex(index=False, escape=False)
    latex = latex.replace('textbf', 'bfseries')
    return latex

def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results and generate tables.")
    parser.add_argument('--report_path', type=str, required=True,
                        help='Path to the directory containing experiment results')
    parser.add_argument('--metric_type', type=str, choices=['macro', 'weighted'], default='macro',
                        help='Type of averaging to use for metrics (macro or weighted)')
    parser.add_argument('--output_format', type=str, choices=['md', 'tex'], default='md',
                        help='Output format for the tables')
    parser.add_argument('--output_file', type=str, default='report',
                        help='File to save the output (if not specified, prints to console)')
    
    args = parser.parse_args()
    
    results = parse_results(args.report_path)
    
    output = []
    
    for task in ['dating', 'locating']:
        output.append(f"\n=== {task.capitalize()} Task ({args.metric_type} average) ===")
        
        df = create_table(results, task, args.metric_type)
        highlighted_df = highlight_best(df, ['Accuracy', 'Precision', 'Recall', 'F1-score'])
        
        if args.output_format == 'md':
            output.append("\nMarkdown Table:")
            output.append(df_to_markdown(highlighted_df))
        
        if args.output_format == 'tex':
            output.append("\nLaTeX Table:")
            output.append(df_to_latex(highlighted_df))

    output_text = '\n'.join(output)
    
    if args.output_file:
        with open(f"{args.report_path}/{args.output_file}_{args.metric_type}.{args.output_format}", 'w') as f:
            f.write(output_text)
        print(f"Results saved to {args.output_file}_{args.metric_type}.{args.output_format}")
    else:
        print(output_text)

if __name__ == "__main__":
    main()