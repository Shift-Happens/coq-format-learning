import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

class NgramModel:
    def __init__(self, n: int = 3):
        self.n = n
        self.context_counts = defaultdict(Counter)
        self.vocabulary = set()
        
    def _get_context(self, tokens: List[str], pos: int) -> Tuple[str]:
        """Get the context (previous n-1 tokens) for a given position."""
        start = max(0, pos - (self.n - 1))
        context = tokens[start:pos]
        # Pad with START tokens if at beginning
        if pos < self.n - 1:
            context = ['<START>'] * (self.n - 1 - pos) + context
        return tuple(context)
    
    def fit(self, token_streams: List[List[str]]) -> None:
        """Train the model on token streams."""
        # Token streams should include both code tokens and spacing tokens
        for tokens in token_streams:
            # Add tokens to vocabulary
            self.vocabulary.update(tokens)
            
            # Count n-gram occurrences
            for i in range(len(tokens)):
                context = self._get_context(tokens, i)
                token = tokens[i]
                self.context_counts[context][token] += 1
    
    def predict(self, context: List[str], k: int = 1) -> List[Tuple[str, float]]:
        """Predict the next k most likely tokens given a context."""
        context_tuple = tuple(context[-(self.n-1):])
        if context_tuple not in self.context_counts:
            # Return uniform distribution over spacing tokens if context unknown
            return [('<SPACE>', 0.5), ('<NO_SPACE>', 0.5)]
        
        # Get counts for this context
        counts = self.context_counts[context_tuple]
        total = sum(counts.values())
        
        # Get top k predictions with probabilities
        predictions = [(token, count/total) 
                      for token, count in counts.most_common(k)]
        
        return predictions
    
    def evaluate(self, test_data: List[Tuple[List[str], str]]) -> Dict[str, float]:
        """Evaluate model on test data."""
        top1_correct = 0
        top3_correct = 0
        total = 0
        
        for context, true_token in test_data:
            predictions = self.predict(context, k=3)
            predicted_tokens = [p[0] for p in predictions]
            
            if predicted_tokens[0] == true_token:
                top1_correct += 1
            if true_token in predicted_tokens:
                top3_correct += 1
            total += 1
        
        return {
            'top1_accuracy': top1_correct / total,
            'top3_accuracy': top3_correct / total
        }

""" Example usage:
if __name__ == "__main__":
    # Example token stream with spacing tokens
    example_data = [
        ['move', '<NO_SPACE>', '=>', '<SPACE>', 'x'],
        ['move', '<NO_SPACE>', '=>', '<SPACE>', 'y']
    ]
    
    # Create and train model
    model = NgramModel(n=3)
    model.fit(example_data)
    
    # Make prediction
    context = ['move', '=>']
    predictions = model.predict(context)
    print(f"Top prediction for context {context}: {predictions[0]}")
"""