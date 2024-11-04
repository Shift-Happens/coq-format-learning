import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from typing import List, Dict, Tuple

class SimpleNeuralFormatter(Model):
    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.bidirectional_lstm = layers.Bidirectional(
            layers.LSTM(hidden_dim, return_sequences=True)
        )
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dense = layers.Dense(2, activation='softmax')  # 2 classes: SPACE or NO_SPACE
        
    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.bidirectional_lstm(x)
        x = self.global_pool(x)
        return self.dense(x)

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
    
    def _prepare_sequence(self, tokens: List[str]) -> tf.Tensor:
        """Convert token sequence to tensor."""
        indices = [self.token_to_idx.get(token, self.token_to_idx['<PAD>']) 
                  for token in tokens]
        return tf.convert_to_tensor([indices], dtype=tf.int32)
    
    def _prepare_batch(self, sequences: List[List[str]], max_length: int = None) -> tf.Tensor:
        """Prepare a batch of sequences."""
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
            
        batch = []
        for seq in sequences:
            # Pad sequence to max_length
            padded = seq + ['<PAD>'] * (max_length - len(seq))
            indices = [self.token_to_idx.get(token, self.token_to_idx['<PAD>']) 
                      for token in padded]
            batch.append(indices)
            
        return tf.convert_to_tensor(batch, dtype=tf.int32)
    
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
        max_seq_length = max(len(context) for context, _ in training_data)
        X = self._prepare_batch([context for context, _ in training_data], 
                              max_length=max_seq_length)
        y = tf.convert_to_tensor([self.spacing_to_idx[spacing] 
                                for _, spacing in training_data])
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        # Train model
        history = self.model.fit(
            dataset,
            epochs=epochs,
            verbose=1
        )
        
        return history
    
    def predict(self, context: List[str], k: int = 1) -> List[Tuple[str, float]]:
        """Predict spacing for given context."""
        if not self.model:
            return [('SPACE', 0.5)]
        
        x = self._prepare_sequence(context)
        predictions = self.model.predict(x)
        
        # Get top k predictions
        top_k = tf.math.top_k(predictions[0], k=k)
        
        return [(self.idx_to_spacing[idx], float(prob)) 
                for idx, prob in zip(top_k.indices.numpy(), top_k.values.numpy())]
    
    def save(self, path: str):
        """Save the model and vocabularies."""
        if self.model:
            # Save Keras model
            self.model.save(f"{path}_model.keras")
            
            # Save vocabularies
            vocabularies = {
                'token_to_idx': self.token_to_idx,
                'idx_to_token': self.idx_to_token,
                'spacing_to_idx': self.spacing_to_idx,
                'idx_to_spacing': self.idx_to_spacing
            }
            np.save(f"{path}_vocab.npy", vocabularies)
    
    def load(self, path: str):
        """Load the model and vocabularies."""
        # Load Keras model
        self.model = tf.keras.models.load_model(f"{path}_model.keras")
        
        # Load vocabularies
        vocabularies = np.load(f"{path}_vocab.npy", allow_pickle=True).item()
        self.token_to_idx = vocabularies['token_to_idx']
        self.idx_to_token = vocabularies['idx_to_token']
        self.spacing_to_idx = vocabularies['spacing_to_idx']
        self.idx_to_spacing = vocabularies['idx_to_spacing']