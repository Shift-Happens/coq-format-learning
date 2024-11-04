import re
from typing import List, Tuple

class CoqPreprocessor:
    def __init__(self):
        self.keywords = {'move', '=>', ':', '.', 'Qed', 'Proof', 'forall', 'exists'}  # Define Coq tokens/keywords for simplified tokenization
    
    def tokenize(self, text: str) -> List[Tuple[str, str]]:
        """
        Simplified tokenization of Coq code.
        Returns list of (token, spacing) pairs.
        """
        text = re.sub(r'\(\*.*?\*\)', '', text, flags=re.DOTALL)  # Remove comments
        
        tokens_with_spacing = []
        lines = text.split('\n')
        
        for line in lines:
            if not line.strip():
                continue
            
            tokens = re.findall(r'\S+|\s+', line)
            
            for i, token in enumerate(tokens):
                if token.strip():  # This is a code token
                    next_space = ''
                    if i < len(tokens) - 1 and tokens[i + 1].isspace():
                        next_space = 'SPACE'
                    else:
                        next_space = 'NO_SPACE'
                    tokens_with_spacing.append((token, next_space))
        
        return tokens_with_spacing
    
    def prepare_training_data(self, tokens_with_spacing: List[Tuple[str, str]], 
                            window_size: int = 3) -> List[Tuple[List[str], str]]:
        """
        Prepare training data for models.
        Returns list of (context_tokens, spacing) pairs.
        """
        training_data = []
        tokens = [t[0] for t in tokens_with_spacing]
        spacings = [t[1] for t in tokens_with_spacing]
        
        for i in range(len(tokens)):
            start = max(0, i - window_size)
            context = tokens[start:i]
            
            if len(context) < window_size:
                context = ['<START>'] * (window_size - len(context)) + context
            
            training_data.append((context, spacings[i]))
        
        return training_data
