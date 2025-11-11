from import_file import *
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    """
    Attention mechanism for sequence-to-one classification.

    It calculates attention weights over the sequence of hidden states 
    (rnn_out) and produces a single context vector (weighted sum of rnn_out).
    """
    def __init__(self, hidden_size):
        super().__init__()
        # W_a: Linear layer to transform h_t to a score
        self.W_a = nn.Linear(hidden_size, hidden_size)
        # v_a: Linear layer to project the score to a single value
        self.v_a = nn.Linear(hidden_size, 1)

    def forward(self, rnn_out):
        # rnn_out shape: (batch_size, seq_len, hidden_size * num_directions)
        
        # 1. Compute Alignment Scores (Energy): e_t = v_a * tanh(W_a * h_t)
        # e_t shape: (batch_size, seq_len, hidden_size)
        energy = torch.tanh(self.W_a(rnn_out))
        
        # e_t_prime shape: (batch_size, seq_len, 1)
        attention_scores = self.v_a(energy)
        
        # 2. Compute Attention Weights: alpha_t = softmax(e_t)
        # alpha_t shape: (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # 3. Compute Context Vector (Weighted sum of rnn_out): c = sum(alpha_t * h_t)
        # context_vector shape: (batch_size, hidden_size * num_directions)
        context_vector = torch.sum(rnn_out * attention_weights, dim=1)
        
        return context_vector, attention_weights