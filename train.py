from src.preprocessor import CoqPreprocessor
from src.ngram_model import NgramModel
from src.neural_model import NeuralFormatter
import random
from typing import List, Tuple

def load_sample_data() -> str:
    """Load sample Coq code."""
    return """
    Proof.
      move=> x y.
      exists z.
      forall w, P w -> Q w.
    Qed.
    """

def split_data(data: List[Tuple[List[str], str]], 
               train_ratio: float = 0.8) -> Tuple[List, List]:
    """Split data into training and test sets."""
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]

def evaluate_model(model, test_data: List[Tuple[List[str], str]]) -> dict:
    """Evaluate model performance."""
    top1_correct = 0
    top3_correct = 0
    total = 0
    
    for context, true_spacing in test_data:
        predictions = model.predict(context, k=3)
        predicted_spacings = [p[0] for p in predictions]
        
        if predicted_spacings[0] == true_spacing:
            top1_correct += 1
        if true_spacing in predicted_spacings:
            top3_correct += 1
        total += 1
    
    return {
        'top1_accuracy': top1_correct / total,
        'top3_accuracy': top3_correct / total
    }

def main():
    # Load and preprocess data
    preprocessor = CoqPreprocessor()
    sample_code = load_sample_data()
    tokens = preprocessor.tokenize(sample_code)
    data = preprocessor.prepare_training_data(tokens)
    
    # Split data
    train_data, test_data = split_data(data)
    
    # Train and evaluate n-gram model
    print("\nTraining N-gram model...")
    ngram_model = NgramModel(n=3)
    ngram_model.fit([context for context, _ in train_data])
    ngram_results = evaluate_model(ngram_model, test_data)
    print(f"N-gram model results: {ngram_results}")
    
    # Train and evaluate neural model
    print("\nTraining Neural model...")
    neural_model = NeuralFormatter()
    neural_model.fit(train_data, epochs=20)
    neural_results = evaluate_model(neural_model, test_data)
    print(f"Neural model results: {neural_results}")

if __name__ == "__main__":
    main()