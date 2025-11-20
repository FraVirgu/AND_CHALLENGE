import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridCNNRNNClassifier(nn.Module):
    """
    Simplified Hybrid Model:
    - Embeddings (pain, time, legs, hands, eyes)
    - RNN (GRU/LSTM/RNN)
    - CNN branch (Conv1D)
    - No attention (use last hidden state)
    """

    def __init__(
        self,
        input_size_numeric: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        max_timesteps: int,
        emb_dim_pain: int = 8,
        emb_dim_time: int = 8,
        emb_dim_legs: int = 2,
        emb_dim_hands: int = 2,
        emb_dim_eyes: int = 2,
        rnn_type: str = "GRU",
        bidirectional: bool = False,
        dropout_rate: float = 0.3,
        cnn_channels: int = 64,
        kernel_sizes=(3, 5, 7),
        fc_hidden: int = 64,
    ):
        super().__init__()

        self.rnn_type = rnn_type.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # === Embeddings ===
        self.emb_pain_1 = nn.Embedding(3, emb_dim_pain)
        self.emb_pain_2 = nn.Embedding(3, emb_dim_pain)
        self.emb_pain_3 = nn.Embedding(3, emb_dim_pain)
        self.emb_pain_4 = nn.Embedding(3, emb_dim_pain)

        self.emb_time = nn.Embedding(max_timesteps, emb_dim_time)

        self.emb_legs = nn.Embedding(3, emb_dim_legs)
        self.emb_hands = nn.Embedding(3, emb_dim_hands)
        self.emb_eyes = nn.Embedding(3, emb_dim_eyes)

        # Time is added inside pain embeddings → DO NOT add emb_dim_time here
        embed_dim = (4 * emb_dim_pain) + emb_dim_legs + emb_dim_hands + emb_dim_eyes

        # RNN input = numeric + embeddings
        self.rnn_input_dim = input_size_numeric + embed_dim

        # === RNN ===
        rnn_dict = {"GRU": nn.GRU, "LSTM": nn.LSTM, "RNN": nn.RNN}
        RNN = rnn_dict[self.rnn_type]

        self.rnn = RNN(
            input_size=self.rnn_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            batch_first=True
        )

        rnn_output_dim = hidden_size * (2 if bidirectional else 1)

        # === CNN branch ===
        self.cnn_convs = nn.ModuleList([
            nn.Conv1d(self.rnn_input_dim, cnn_channels, kernel_size=ks)
            for ks in kernel_sizes
        ])

        cnn_feature_dim = cnn_channels * len(kernel_sizes)

        # === Final classifier ===
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(rnn_output_dim + cnn_feature_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden, num_classes)
        )

    def forward(self, x_num, pain, n_legs, n_hands, n_eyes, time_idx):

        # === Time embedding ===
        time_emb = self.emb_time(time_idx)

        # Add time into each pain embedding
        p1 = self.emb_pain_1(pain[..., 0]) + time_emb
        p2 = self.emb_pain_2(pain[..., 1]) + time_emb
        p3 = self.emb_pain_3(pain[..., 2]) + time_emb
        p4 = self.emb_pain_4(pain[..., 3]) + time_emb
        pain_emb = torch.cat([p1, p2, p3, p4], dim=-1)

        # Categorical embeddings
        legs_emb  = self.emb_legs(n_legs)
        hands_emb = self.emb_hands(n_hands)
        eyes_emb  = self.emb_eyes(n_eyes)

        # Final embedding concat
        emb_total = torch.cat([pain_emb, legs_emb, hands_emb, eyes_emb], dim=-1)

        # === RNN input ===
        rnn_input = torch.cat([x_num, emb_total], dim=-1)

        # === RNN ===
        if self.rnn_type == "LSTM":
            rnn_out, (h_n, _) = self.rnn(rnn_input)
        else:
            rnn_out, h_n = self.rnn(rnn_input)

        # Last hidden state
        if self.bidirectional:
            rnn_feat = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, 2H)
        else:
            rnn_feat = h_n[-1]  # (B, H)

        # === CNN branch ===
        cnn_in = rnn_input.permute(0, 2, 1)  # (B, C, T)

        cnn_features = []
        for conv in self.cnn_convs:
            x = F.relu(conv(cnn_in))
            x = F.max_pool1d(x, kernel_size=x.size(-1))
            cnn_features.append(x.squeeze(-1))

        cnn_feat = torch.cat(cnn_features, dim=1)

        # === Fusion and classifier ===
        combined = torch.cat([rnn_feat, cnn_feat], dim=1)
        combined = self.dropout(combined)

        logits = self.fc(combined)
        return logits
