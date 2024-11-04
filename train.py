from src.preprocessor import CoqPreprocessor
from src.ngram_model import NgramModel
from src.neural_model import NeuralFormatter
from src.utils import FormatUtils, ModelUtils, DatasetStats
import tensorflow as tf
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Train Coq formatting models')
    parser.add_argument('--input', type=str, default='data/sample_input/simple.v',
                      help='Input Coq file for training')
    parser.add_argument('--epochs', type=int, default=80,
                      help='Number of epochs for neural model training')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for neural model training')
    parser.add_argument('--output-dir', type=str, default='models',
                      help='Directory to save trained models')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess data
    preprocessor = CoqPreprocessor()
    sample_code = FormatUtils.load_coq_file(args.input)
    tokens = preprocessor.tokenize(sample_code)
    
    # Calculate and display dataset statistics
    stats = DatasetStats.get_token_statistics(tokens)
    DatasetStats.print_dataset_info(stats)
    
    # Prepare training data
    data = preprocessor.prepare_training_data(tokens)
    train_data, test_data = ModelUtils.split_data(data, train_ratio=0.8)
    
    # Train and evaluate n-gram model
    print("\nTraining N-gram model...")
    ngram_model = NgramModel(n=3)
    ngram_model.fit([context for context, _ in train_data])
    ngram_results = ModelUtils.evaluate_model(ngram_model, test_data)
    print(f"N-gram model results: {ngram_results}")
    
    # Save n-gram results
    ModelUtils.save_model_metrics(ngram_results, 'ngram', args.output_dir)
    
    # Train and evaluate neural model
    print("\nTraining Neural model...")
    neural_model = NeuralFormatter()
    history = neural_model.fit(train_data, 
                             epochs=args.epochs,
                             batch_size=args.batch_size)
    
    # Evaluate neural model
    neural_results = ModelUtils.evaluate_model(neural_model, test_data)
    print(f"Neural model results: {neural_results}")
    
    # Save neural model and results
    neural_model.save(os.path.join(args.output_dir, "neural_formatter"))
    ModelUtils.save_model_metrics(neural_results, 'neural', args.output_dir)
    
    # Save training history
    history_dict = history.history
    ModelUtils.save_model_metrics(history_dict, 'neural_history', args.output_dir)

if __name__ == "__main__":
    main()