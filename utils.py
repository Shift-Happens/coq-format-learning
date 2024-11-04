import os
from typing import List, Tuple, Dict
from collections import defaultdict
import random
import json
from pathlib import Path

class FormatUtils:
    @staticmethod
    def load_coq_file(file_path: str) -> str:
        """Load a Coq file and return its content."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def save_tokens(tokens: List[Tuple[str, str]], 
                   output_path: str,
                   metadata: Dict = None):
        """Save tokenized data with optional metadata."""
        data = {
            'tokens': tokens,
            'metadata': metadata or {}
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load_tokens(input_path: str) -> Tuple[List[Tuple[str, str]], Dict]:
        """Load tokenized data and metadata."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['tokens'], data.get('metadata', {})

class ModelUtils:
    @staticmethod
    def split_data(data: List[Tuple[List[str], str]], train_ratio: float = 0.8) -> Tuple[List[Tuple[List[str], str]], List[Tuple[List[str], str]]]:
        """Split the data into training and test sets."""
        random.shuffle(data)
        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size]
        test_data = data[train_size:]
        return train_data, test_data

    @staticmethod
    def create_batches(data: List[Tuple[List[str], str]], 
                      batch_size: int) -> List[List[Tuple[List[str], str]]]:
        """Create batches from training data."""
        random.shuffle(data)
        return [data[i:i + batch_size] 
                for i in range(0, len(data), batch_size)]
    
    @staticmethod
    def save_model_metrics(metrics: Dict, 
                          model_name: str,
                          output_dir: str):
        """Save model evaluation metrics."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{model_name}_metrics.json")
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    @staticmethod
    def evaluate_model(model, test_data: List[Tuple[List[str], str]]) -> Dict:
        """Evaluate a model on the test data."""
        correct = 0
        total = 0
        predictions = defaultdict(list)

        for sample, label in test_data:
            prediction = model.predict(sample)
            predictions[label].append(prediction)
            if prediction == label:
                correct += 1
            total += 1

        accuracy = correct / total
        return {
            "accuracy": accuracy,
            "predictions": predictions
        }

class DatasetStats:
    @staticmethod
    def get_token_statistics(tokens: List[Tuple[str, str]]) -> Dict:
        """Calculate statistics about the dataset."""
        total_tokens = len(tokens)
        unique_tokens = len(set(t[0] for t in tokens))
        space_distribution = {
            'SPACE': sum(1 for t in tokens if t[1] == 'SPACE'),
            'NO_SPACE': sum(1 for t in tokens if t[1] == 'NO_SPACE')
        }
        
        return {
            'total_tokens': total_tokens,
            'unique_tokens': unique_tokens,
            'space_distribution': space_distribution
        }
    
    @staticmethod
    def print_dataset_info(stats: Dict):
        """Print formatted dataset statistics."""
        print("\nDataset Statistics:")
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"Unique tokens: {stats['unique_tokens']}")
        print("\nSpacing Distribution:")
        total = sum(stats['space_distribution'].values())
        for space_type, count in stats['space_distribution'].items():
            percentage = (count / total) * 100
            print(f"{space_type}: {count} ({percentage:.1f}%)")

# Example usage
if __name__ == "__main__":
    # Test the utilities
    sample_tokens = [
        ("Proof", "NO_SPACE"),
        (".", "SPACE"),
        ("move", "NO_SPACE"),
        ("=>", "SPACE")
    ]
    
    # Calculate and print statistics
    stats = DatasetStats.get_token_statistics(sample_tokens)
    DatasetStats.print_dataset_info(stats)
    
    # Save and load tokens
    FormatUtils.save_tokens(sample_tokens, "test_tokens.json")
    loaded_tokens, _ = FormatUtils.load_tokens("test_tokens.json")
    print("\nLoaded tokens match original:", loaded_tokens == sample_tokens)