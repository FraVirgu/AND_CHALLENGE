import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Simple additive attention layer."""
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (B, T, H)
        attn_scores = self.attn(x)                     # (B, T, 1)
        weights = F.softmax(attn_scores, dim=1)        # normalize over time
        context = torch.sum(weights * x, dim=1)        # weighted sum
        return context, weights


class HybridAttentionClassifier(nn.Module):
    """
    Hybrid model with:
      - pain_survey embeddings + time embeddings (summed)
      - embeddings for n_legs, n_hands, n_eyes
      - numeric features (joints, etc.)
      - RNN + attention + classifier
    """
    def __init__(
        self,
        input_size_numeric,      # number of numeric (continuous) features, e.g. joints
        hidden_size,
        num_layers,
        num_classes,
        max_timesteps=160,
        rnn_type='GRU',
        bidirectional=False,
        dropout_rate=0.2
    ):
        super().__init__()

        # === Embedding dimensions ===
        emb_dim_pain = 4
        emb_dim_time = 4
        emb_dim_legs = 2
        emb_dim_hands = 2
        emb_dim_eyes = 2

        # === Pain embeddings ===
        self.emb_pain_1 = nn.Embedding(3, emb_dim_pain)
        self.emb_pain_2 = nn.Embedding(3, emb_dim_pain)
        self.emb_pain_3 = nn.Embedding(3, emb_dim_pain)
        self.emb_pain_4 = nn.Embedding(3, emb_dim_pain)

        # === Time embedding (shared across all pains) ===
        self.emb_time = nn.Embedding(max_timesteps, emb_dim_time)

        # === Categorical embeddings ===
        self.emb_legs = nn.Embedding(3, emb_dim_legs)     # values {1,2}
        self.emb_hands = nn.Embedding(3, emb_dim_hands)   # values {1,2}
        self.emb_eyes = nn.Embedding(3, emb_dim_eyes)     # values {1,2}

        # === Calculate total embedding size per timestep ===
        emb_total_dim = (emb_dim_pain * 4) + emb_dim_legs + emb_dim_hands + emb_dim_eyes

        # === RNN definition ===
        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
            input_size=input_size_numeric + emb_total_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.num_directions = 2 if bidirectional else 1
        attn_in_dim = hidden_size * self.num_directions

        # === Attention + Classifier ===
        self.attention = AttentionLayer(attn_in_dim)
        self.classifier = nn.Linear(attn_in_dim, num_classes)

    def forward(self, x_num, pain, n_legs, n_hands, n_eyes, time_idx):
        """
        Args:
            x_num:   (B, T, F_num)   → numeric features (e.g. joints)
            pain:    (B, T, 4)       → pain_survey_1..4 (integers 0,1,2)
            n_legs:  (B, T)
            n_hands: (B, T)
            n_eyes:  (B, T)
            time_idx:(B, T)
        """

        # --- Time embedding ---
        t_emb = self.emb_time(time_idx)  # (B, T, 4)

        # --- Pain embeddings (add time) ---
        e1 = self.emb_pain_1(pain[:, :, 0]) + t_emb
        e2 = self.emb_pain_2(pain[:, :, 1]) + t_emb
        e3 = self.emb_pain_3(pain[:, :, 2]) + t_emb
        e4 = self.emb_pain_4(pain[:, :, 3]) + t_emb

        pain_emb = torch.cat([e1, e2, e3, e4], dim=-1)  # (B, T, 16)

        # --- Other categorical embeddings ---
        legs_emb = self.emb_legs(n_legs)    # (B, T, 2)
        hands_emb = self.emb_hands(n_hands) # (B, T, 2)
        eyes_emb = self.emb_eyes(n_eyes)    # (B, T, 2)

        # --- Concatenate all embeddings + numeric features ---
        emb_total = torch.cat([pain_emb, legs_emb, hands_emb, eyes_emb], dim=-1)
        rnn_input = torch.cat([x_num, emb_total], dim=-1)

        # --- RNN + Attention ---
        rnn_out, _ = self.rnn(rnn_input)
        context, _ = self.attention(rnn_out)
        logits = self.classifier(context)
        return logits
