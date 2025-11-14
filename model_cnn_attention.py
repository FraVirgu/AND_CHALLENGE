import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayerEmbedding(nn.Module):
    """Simple additive attention layer over time."""
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        x: (B, T, H)
        returns:
            context: (B, H)
            weights: (B, T, 1)
        """
        attn_scores = self.attn(x)              # (B, T, 1)
        weights = F.softmax(attn_scores, dim=1) # normalize over time
        context = torch.sum(weights * x, dim=1) # (B, H)
        return context, weights


class HybridAttentionCNNClassifier(nn.Module):
    """
    Hybrid model:
    - Embeddings for pain surveys, time, and categorical features (legs, hands, eyes)
    - RNN + attention over time
    - Parallel CNN branch over the same time series
    - Concatenate RNN+attention features with CNN features for final classification

    Inputs to forward():
        x_num:   (B, T, F_num)          numeric features
        pain:    (B, T, 4)              pain_survey_1..4 (int-coded 0..2)
        n_legs:  (B, T)                 categorical {1,2} (or {0,1,2} after encoding)
        n_hands: (B, T)
        n_eyes:  (B, T)
        time_idx:(B, T)                 integer time step (0..max_timesteps-1)
    """
    def __init__(
        self,
        input_size_numeric: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        max_timesteps: int,
        # embedding dims
        emb_dim_pain: int = 8,
        emb_dim_time: int = 4,
        emb_dim_legs: int = 2,
        emb_dim_hands: int = 2,
        emb_dim_eyes: int = 2,
        # RNN / CNN config
        rnn_type: str = "GRU",
        bidirectional: bool = True,
        dropout_rate: float = 0.3,
        cnn_channels: int = 64,
        kernel_sizes=(3, 5, 7),
        fc_hidden: int = 64,
    ):
        super().__init__()

        self.input_size_numeric = input_size_numeric
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.rnn_type = rnn_type.upper()
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate

        # === Pain embeddings (each question separate) ===
        self.emb_pain_1 = nn.Embedding(3, emb_dim_pain)  # 3 pain levels assumed
        self.emb_pain_2 = nn.Embedding(3, emb_dim_pain)
        self.emb_pain_3 = nn.Embedding(3, emb_dim_pain)
        self.emb_pain_4 = nn.Embedding(3, emb_dim_pain)

        # === Time embedding (shared across all pains) ===
        self.emb_time = nn.Embedding(max_timesteps, emb_dim_time)

        # === Categorical embeddings ===
        self.emb_legs = nn.Embedding(3, emb_dim_legs)    # {1,2} -> encoded up to 2
        self.emb_hands = nn.Embedding(3, emb_dim_hands)
        self.emb_eyes = nn.Embedding(3, emb_dim_eyes)

        # Total embedding dimension per time step:
        pain_total_dim = 4 * emb_dim_pain               # 4 questions
        cat_total_dim = emb_dim_legs + emb_dim_hands + emb_dim_eyes
        self.emb_total_dim = pain_total_dim + cat_total_dim

        # RNN input = numeric features + all embeddings
        self.rnn_input_dim = self.input_size_numeric + self.emb_total_dim

        # === RNN ===
        rnn_cls = {
            "RNN": nn.RNN,
            "LSTM": nn.LSTM,
            "GRU": nn.GRU
        }[self.rnn_type]

        self.rnn = rnn_cls(
            input_size=self.rnn_input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout_rate if self.num_layers > 1 else 0.0
        )

        rnn_output_dim = self.hidden_size * (2 if self.bidirectional else 1)

        # === Attention over RNN outputs ===
        self.attention = AttentionLayerEmbedding(rnn_output_dim)

        # === CNN branch over the same time series ===
        # Conv1d expects (B, C_in, T), so channels = rnn_input_dim
        self.cnn_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.rnn_input_dim,
                out_channels=cnn_channels,
                kernel_size=ks
            )
            for ks in kernel_sizes
        ])
        self.cnn_activation = nn.ReLU()
        self.cnn_channels = cnn_channels
        self.kernel_sizes = kernel_sizes

        # === Final classifier ===
        cnn_feature_dim = cnn_channels * len(kernel_sizes)
        classifier_input_dim = rnn_output_dim + cnn_feature_dim

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden, num_classes)
        )

    def forward(self, x_num, pain, n_legs, n_hands, n_eyes, time_idx):
        """
        x_num:   (B, T, F_num)
        pain:    (B, T, 4)
        n_legs:  (B, T)
        n_hands: (B, T)
        n_eyes:  (B, T)
        time_idx:(B, T)
        """
        # --- Shape checks (optional, can be commented out in production) ---
        # B, T, _ = x_num.shape

        # === Time embedding ===
        time_emb = self.emb_time(time_idx)  # (B, T, emb_dim_time)

        # === Pain embeddings (add time info into each pain embedding) ===
        p1 = self.emb_pain_1(pain[..., 0]) + time_emb  # (B, T, emb_dim_pain)
        p2 = self.emb_pain_2(pain[..., 1]) + time_emb
        p3 = self.emb_pain_3(pain[..., 2]) + time_emb
        p4 = self.emb_pain_4(pain[..., 3]) + time_emb

        pain_emb = torch.cat([p1, p2, p3, p4], dim=-1)  # (B, T, 4*emb_dim_pain)

        # === Other categorical embeddings ===
        legs_emb = self.emb_legs(n_legs)    # (B, T, emb_dim_legs)
        hands_emb = self.emb_hands(n_hands) # (B, T, emb_dim_hands)
        eyes_emb = self.emb_eyes(n_eyes)    # (B, T, emb_dim_eyes)

        # === Concatenate all embeddings ===
        emb_total = torch.cat([pain_emb, legs_emb, hands_emb, eyes_emb], dim=-1)  # (B, T, emb_total_dim)

        # === RNN input: numeric + embeddings ===
        rnn_input = torch.cat([x_num, emb_total], dim=-1)  # (B, T, rnn_input_dim)

        # --- RNN branch ---
        rnn_out, _ = self.rnn(rnn_input)                   # (B, T, rnn_output_dim)
        attn_context, _ = self.attention(rnn_out)          # (B, rnn_output_dim)

        # --- CNN branch ---
        # Conv1d: (B, C_in, T)
        cnn_in = rnn_input.permute(0, 2, 1)                # (B, rnn_input_dim, T)

        cnn_features = []
        for conv in self.cnn_convs:
            x = self.cnn_activation(conv(cnn_in))          # (B, C_out, T')
            # Global max-pooling over time
            x = F.max_pool1d(x, kernel_size=x.size(-1))    # (B, C_out, 1)
            x = x.squeeze(-1)                              # (B, C_out)
            cnn_features.append(x)

        cnn_out = torch.cat(cnn_features, dim=1)           # (B, cnn_channels * len(kernel_sizes))

        # --- Fusion ---
        combined = torch.cat([attn_context, cnn_out], dim=1)  # (B, classifier_input_dim)
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        return logits
