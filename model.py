from data_import import *
from attention_layer import AttentionLayer

def recurrent_summary(model, input_size):
    """
    Custom summary function that emulates torchinfo's output while correctly
    counting parameters for RNN/GRU/LSTM layers.

    This function is designed for models whose direct children are
    nn.Linear, nn.RNN, nn.GRU, or nn.LSTM layers.

    Args:
        model (nn.Module): The model to analyze.
        input_size (tuple): Shape of the input tensor (e.g., (seq_len, features)).
    """

    # Dictionary to store output shapes captured by forward hooks
    output_shapes = {}
    # List to track hook handles for later removal
    hooks = []

    def get_hook(name):
        """Factory function to create a forward hook for a specific module."""
        def hook(module, input, output):
            # Handle RNN layer outputs (returns a tuple)
            if isinstance(output, tuple):
                # output[0]: all hidden states with shape (batch, seq_len, hidden*directions)
                shape1 = list(output[0].shape)
                shape1[0] = -1  # Replace batch dimension with -1

                # output[1]: final hidden state h_n (or tuple (h_n, c_n) for LSTM)
                if isinstance(output[1], tuple):  # LSTM case: (h_n, c_n)
                    shape2 = list(output[1][0].shape)  # Extract h_n only
                else:  # RNN/GRU case: h_n only
                    shape2 = list(output[1].shape)

                # Replace batch dimension (middle position) with -1
                shape2[1] = -1

                output_shapes[name] = f"[{shape1}, {shape2}]"

            # Handle standard layer outputs (e.g., Linear)
            else:
                shape = list(output.shape)
                shape[0] = -1  # Replace batch dimension with -1
                output_shapes[name] = f"{shape}"
        return hook

    # 1. Determine the device where model parameters reside
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")  # Fallback for models without parameters

    # 2. Create a dummy input tensor with batch_size=1
    dummy_input = torch.randn(1, *input_size).to(device)

    # 3. Register forward hooks on target layers
    # Iterate through direct children of the model (e.g., self.rnn, self.classifier)
    for name, module in model.named_children():
        if isinstance(module, (nn.Linear, nn.RNN, nn.GRU, nn.LSTM)):
            # Register the hook and store its handle for cleanup
            hook_handle = module.register_forward_hook(get_hook(name))
            hooks.append(hook_handle)

    # 4. Execute a dummy forward pass in evaluation mode
    model.eval()
    with torch.no_grad():
        try:
            model(dummy_input)
        except Exception as e:
            print(f"Error during dummy forward pass: {e}")
            # Clean up hooks even if an error occurs
            for h in hooks:
                h.remove()
            return

    # 5. Remove all registered hooks
    for h in hooks:
        h.remove()

    # --- 6. Print the summary table ---

    print("-" * 79)
    # Column headers
    print(f"{'Layer (type)':<25} {'Output Shape':<28} {'Param #':<18}")
    print("=" * 79)

    total_params = 0
    total_trainable_params = 0

    # Iterate through modules again to collect and display parameter information
    for name, module in model.named_children():
        if name in output_shapes:
            # Count total and trainable parameters for this module
            module_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)

            total_params += module_params
            total_trainable_params += trainable_params

            # Format strings for display
            layer_name = f"{name} ({type(module).__name__})"
            output_shape_str = str(output_shapes[name])
            params_str = f"{trainable_params:,}"

            print(f"{layer_name:<25} {output_shape_str:<28} {params_str:<15}")

    print("=" * 79)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {total_trainable_params:,}")
    print(f"Non-trainable params: {total_params - total_trainable_params:,}")
    print("-" * 79)



class RecurrentClassifier(nn.Module):
    """
    Generic RNN classifier (RNN, LSTM, GRU).
    Uses the last hidden state for classification.
    """
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            num_classes,
            rnn_type='GRU',        # 'RNN', 'LSTM', or 'GRU'
            bidirectional=False,
            dropout_rate=0.2
            ):
        super().__init__()

        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # Map string name to PyTorch RNN class
        rnn_map = {
            'RNN': nn.RNN,
            'LSTM': nn.LSTM,
            'GRU': nn.GRU
        }

        if rnn_type not in rnn_map:
            raise ValueError("rnn_type must be 'RNN', 'LSTM', or 'GRU'")

        rnn_module = rnn_map[rnn_type]

        # Dropout is only applied between layers (if num_layers > 1)
        dropout_val = dropout_rate if num_layers > 1 else 0

        # Create the recurrent layer
        self.rnn = rnn_module(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,       # Input shape: (batch, seq_len, features)
            bidirectional=bidirectional,
            dropout=dropout_val
        )

        # Calculate input size for the final classifier
        if self.bidirectional:
            classifier_input_size = hidden_size * 2 # Concat fwd + bwd
        else:
            classifier_input_size = hidden_size

        # Final classification layer
        self.classifier = nn.Linear(classifier_input_size, num_classes)

    def forward(self, x):
        """
        x shape: (batch_size, seq_length, input_size)
        """

        # rnn_out shape: (batch_size, seq_len, hidden_size * num_directions)
        rnn_out, hidden = self.rnn(x)

        # LSTM returns (h_n, c_n), we only need h_n
        if self.rnn_type == 'LSTM':
            hidden = hidden[0]

        # hidden shape: (num_layers * num_directions, batch_size, hidden_size)

        if self.bidirectional:
            # Reshape to (num_layers, 2, batch_size, hidden_size)
            hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)

            # Concat last fwd (hidden[-1, 0, ...]) and bwd (hidden[-1, 1, ...])
            # Final shape: (batch_size, hidden_size * 2)
            hidden_to_classify = torch.cat([hidden[-1, 0, :, :], hidden[-1, 1, :, :]], dim=1)
        else:
            # Take the last layer's hidden state
            # Final shape: (batch_size, hidden_size)
            hidden_to_classify = hidden[-1]

        # Get logits
        logits = self.classifier(hidden_to_classify)
        return logits

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn


class AttentionClassifier(nn.Module):
    def __init__(
        self,
        input_size_numeric: int,     # numeric features per timestep
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        max_timesteps: int,          # kept to match API, NOT used anymore
        emb_dim_pain: int = 3,
        emb_dim_legs: int = 2,
        emb_dim_hands: int = 2,
        emb_dim_eyes: int = 2,
        rnn_type: str = "GRU",
        bidirectional: bool = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        # -------------------------------
        # Store basic settings
        # -------------------------------
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.upper()

        # -------------------------------
        # Embeddings
        # -------------------------------
        # Pain survey (4 values per timestep, each ∈ {0,1,2})
        self.emb_pain = nn.Embedding(3, emb_dim_pain)

        # Legs/hands/eyes are binary {0,1}
        self.emb_legs  = nn.Embedding(2, emb_dim_legs)
        self.emb_hands = nn.Embedding(2, emb_dim_hands)
        self.emb_eyes  = nn.Embedding(2, emb_dim_eyes)

        # Total embedding dimension per timestep
        emb_dim_total = (
            4 * emb_dim_pain +
            emb_dim_legs +
            emb_dim_hands +
            emb_dim_eyes
        )

        # -------------------------------
        # RNN input dimension
        # -------------------------------
        self.rnn_input_dim = input_size_numeric + emb_dim_total

        # -------------------------------
        # RNN layer
        # -------------------------------
        rnn_map = {"GRU": nn.GRU, "LSTM": nn.LSTM, "RNN": nn.RNN}
        rnn_cls = rnn_map[self.rnn_type]

        self.rnn = rnn_cls(
            input_size=self.rnn_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

        # Output D
        rnn_out_dim = hidden_size * (2 if bidirectional else 1)

        # -------------------------------
        # Normalization + Dropout
        # -------------------------------
        self.ln = nn.LayerNorm(rnn_out_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # -------------------------------
        # Attention layer
        # -------------------------------
        self.attn = nn.Linear(rnn_out_dim, 1)

        # -------------------------------
        # Classifier head
        # -------------------------------
        fc_hidden = max(32, rnn_out_dim // 2)

        self.fc = nn.Sequential(
            nn.Linear(rnn_out_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden, num_classes),
        )

    # ==============================================================
    # Forward pass
    # ==============================================================

    def forward(self, x_num, pain, n_legs, n_hands, n_eyes, time_idx):
        """
        x_num:   (B, T, F_num)
        pain:    (B, T, 4)      values ∈ {0,1,2}
        n_legs:  (B, T)         values ∈ {0,1}
        n_hands: (B, T)
        n_eyes:  (B, T)
        time_idx is ignored because time is already numeric inside x_num
        """

        B, T, _ = x_num.shape

        # -------------------------------
        # Pain embedding: (B,T,4,dim) → (B,T,4*dim)
        # -------------------------------
        pain_emb = self.emb_pain(pain)           # (B,T,4,E)
        pain_emb = pain_emb.reshape(B, T, -1)    # (B,T,4*E)

        # -------------------------------
        # Categorical embeddings (B,T,E)
        # -------------------------------
        legs_emb  = self.emb_legs(n_legs)
        hands_emb = self.emb_hands(n_hands)
        eyes_emb  = self.emb_eyes(n_eyes)

        # -------------------------------
        # Concatenate embeddings
        # -------------------------------
        emb_all = torch.cat([pain_emb, legs_emb, hands_emb, eyes_emb], dim=-1)
        emb_all = self.dropout(emb_all)

        # -------------------------------
        # RNN input = numeric + embeddings
        # -------------------------------
        rnn_input = torch.cat([x_num, emb_all], dim=-1)  # (B,T,D)

        # -------------------------------
        # RNN forward
        # -------------------------------
        rnn_out, _ = self.rnn(rnn_input)                 # (B,T,H)

        # Normalize + dropout
        rnn_out = self.ln(rnn_out)
        rnn_out = self.dropout(rnn_out)

        # -------------------------------
        # Attention
        # -------------------------------
        attn_scores = self.attn(rnn_out)                 # (B,T,1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = (attn_weights * rnn_out).sum(dim=1)    # (B,H)

        context = self.dropout(context)

        # -------------------------------
        # Classifier
        # -------------------------------
        logits = self.fc(context)
        return logits


# --- 4. DUAL-STREAM ATTENTION CLASSIFIER (New Implementation) ---
class Dual_AttentionClassifier(nn.Module):
    """
    Recurrent Classifier with Dual-Stream Attention Enhancement.
    
    Uses two separate RNNs:
    1. rnn_left: Generates the sequence of features (Keys/Values).
    2. rnn_right: Generates a final context vector (which could serve as Query implicitly).
    
    The sequence output of rnn_left feeds the attention layer. The attention context 
    vector is concatenated with the final hidden state of rnn_right for classification.
    """
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            num_classes,
            rnn_type='GRU',
            bidirectional=False,
            dropout_rate=0.2
            ):
        super().__init__()

        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        rnn_map = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}
        if rnn_type not in rnn_map:
            raise ValueError("rnn_type must be 'RNN', 'LSTM', or 'GRU'")

        rnn_module = rnn_map[rnn_type]
        dropout_val = dropout_rate if num_layers > 1 else 0
        num_directions = 2 if bidirectional else 1
        
        # RNN 1 (Left Stream - Provides sequence for Attention/K, V)
        self.rnn_left = rnn_module(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_val
        )

        # RNN 2 (Right Stream - Provides feature vector to concatenate with context/Q)
        self.rnn_right = rnn_module(
            input_size=input_size, # Can use input_size or hidden_size of rnn_left if stacked
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_val
        )

        attention_input_size = hidden_size * num_directions
        
        # 1. Attention Layer (operates on rnn_left output)
        self.attention = AttentionLayer(attention_input_size)
        
        # Calculate combined input size for classifier: 
        # (Attention Context Vector) + (Final Hidden State of rnn_right)
        classifier_input_size = attention_input_size + attention_input_size 

        # 2. Final classification layer
        self.classifier = nn.Linear(classifier_input_size, num_classes)

    def forward(self, x):
        """
        x shape: (batch_size, seq_length, input_size)
        """

        # Stream 1: Left RNN (Encoder/Key/Value Sequence)
        # rnn_left_out shape: (batch_size, seq_len, hidden_size * num_directions)
        rnn_left_out, _ = self.rnn_left(x) 

        # Stream 2: Right RNN (Context/Feature Sequence)
        # rnn_right_out shape: (batch_size, seq_len, hidden_size * num_directions)
        rnn_right_out, hidden_right = self.rnn_right(x)
        
        # Extract the final hidden state of rnn_right
        if self.rnn_type == 'LSTM':
            hidden_right = hidden_right[0]

        if self.bidirectional:
            # Concat last fwd and bwd states of the *last layer* of the right RNN
            hidden_right = hidden_right.view(self.num_layers, 2, -1, self.hidden_size)
            final_right_state = torch.cat([hidden_right[-1, 0, :, :], hidden_right[-1, 1, :, :]], dim=1)
        else:
            # Take the last layer's hidden state
            final_right_state = hidden_right[-1]
        
        # Attention: Calculate context vector c over rnn_left_out
        # context_vector shape: (batch_size, hidden_size * num_directions)
        context_vector, _ = self.attention(rnn_left_out)

        # Combine the context vector (from Attention over rnn_left) 
        # and the final state (from rnn_right)
        combined_features = torch.cat([context_vector, final_right_state], dim=1)

        # Get logits
        logits = self.classifier(combined_features)
        return logits
# Create model and display architecture with parameter count




def build_model_recurrent_class(input_size, hidden_size, num_layers, num_classes, rnn_type, dropout_rate, device):
    model = RecurrentClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        bidirectional=False,
        rnn_type=rnn_type
    ).to(device)
    return model

def build_model_attention_class(
    input_size_numeric,
    hidden_size,
    num_layers,
    num_classes,
    max_timesteps,
    rnn_type,
    dropout_rate,
    device
):
    model = AttentionClassifier(
        input_size_numeric=input_size_numeric,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        max_timesteps=max_timesteps,
        rnn_type=rnn_type,
        bidirectional=True,         # use BiGRU
        dropout_rate=dropout_rate,
        emb_dim_pain=3,
        emb_dim_legs=2,
        emb_dim_hands=2,
        emb_dim_eyes=2,
    ).to(device)

    return model

def build_model_dual_attention_class(input_size, hidden_size, num_layers, num_classes, rnn_type, dropout_rate, device):
    model = Dual_AttentionClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        bidirectional=False,
        rnn_type=rnn_type
    ).to(device)
    return model


