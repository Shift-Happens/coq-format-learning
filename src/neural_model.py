import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple

class SimpleNeuralFormatter(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, 2)  # 2 classes: SPACE or NO_SPACE
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # Use only the last output for prediction
        output = self.fc(lstm_out[:, -1, :])
        return output

class NeuralFormatter:
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 128):
        self.token_to_idx = {'<PAD>': 0, '<START>': 1}
        self.idx_to_token = {0: '<PAD>', 1: '<START>'}
        self.spacing_to_idx = {'SPACE': 0, 'NO_SPACE': 1}
        self.idx_to_spacing = {0: 'SPACE', 1: 'NO_SPACE'}
        self.model = None
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
    def _build_vocabulary(self, token_streams: List[List[str]]):
        """Build vocabulary from token streams."""
        tokens = set()
        for stream in token_streams:
            tokens.update(stream)
        
        for token in tokens:
            if token not in self.token_to_idx:
                idx = len(self.token_to_idx)
                self.token_to_idx[token] = idx
                self.idx_to_token[idx] = token
    
    def _prepare_sequence(self, tokens: List[str]) -> torch.Tensor:
        """Convert token sequence to tensor."""
        indices = [self.token_to_idx.get(token, self.token_to_idx['<PAD>']) 
                  for token in tokens]
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    
    def fit(self, training_data: List[Tuple[List[str], str]], 
            epochs: int = 10, batch_size: int = 32):
        """Train the neural model."""
        # Build vocabulary from training data
        token_streams = [context for context, _ in training_data]
        self._build_vocabulary(token_streams)
        
        # Initialize model
        self.model = SimpleNeuralFormatter(
            vocab_size=len(self.token_to_idx),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        )
        
        # Prepare training data
        X = [self._prepare_sequence(context) for context, _ in training_data]
        y = [self.spacing_to_idx[spacing] for _, spacing in training_data]
        
        # Training loop
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(X), batch_size):
                batch_X = torch.cat(X[i:i+batch_size])
                batch_y = torch.tensor(y[i:i+batch_size])
                
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}, Loss: {total_loss/len(X):.4f}")
    
    def predict(self, context: List[str], k: int = 1) -> List[Tuple[str, float]]:
        """Predict spacing for given context."""
        if not self.model:
            return [('SPACE', 0.5)]
        
        self.model.eval()
        with torch.no_grad():
            x = self._prepare_sequence(context)
            output = self.model(x)
            probabilities = torch.softmax(output, dim=1)
            
            # Get top k predictions
            values, indices = torch.topk(probabilities[0], k)
            predictions = [(self.idx_to_spacing[idx.item()], prob.item()) 
                         for idx, prob in zip(indices, values)]
            
        return predictions

# Example usage
if __name__ == "__main__":
    # Sample training data
    training_data = [
        (['move', '=>'], 'NO_SPACE'),
        (['exists', 'x'], 'SPACE'),
        (['Proof', '.'], 'NO_SPACE')
    ]
    
    # Create and train model
    formatter = NeuralFormatter()
    formatter.fit(training_data, epochs=20)
    
    # Make prediction
    context = ['move', '=>']
    predictions = formatter.predict(context)
    print(f"Predictions for {context}: {predictions}")