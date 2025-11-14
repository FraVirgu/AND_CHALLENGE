from itertools import product
from model_train import *
from data_import import *
import os
from datetime import datetime
import json
import torch.optim
from torch.utils.data import DataLoader

import torch.nn.functional as F
results = {}   # will store: label → history dict


rnn_type = "GRU"
hidden_size = 32
num_layers = 2
batch_size = 32
lr = 1e-3
dropout_rate = 0.3
l1_lambda = 0.0
l2_lambda = 1e-4

EPOCHS = 300
PATIENCE = 50

best_score = -1
best_config = None
best_model = None


print(f"\n--- Running experiment: rnn_type={rnn_type}, hidden_size={hidden_size}, "
      f"num_layers={num_layers}, batch_size={batch_size}, learning_rate={lr}, "
      f"dropout_rate={dropout_rate}, l1_lambda={l1_lambda}, l2_lambda={l2_lambda} ---")

# --- Loaders for multi-input dataset ---
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_size_numeric = X_num_train.shape[-1]
max_timesteps = int(time_idx_train.max()) + 1

model = build_model_attention_class(
    input_size_numeric=input_size_numeric,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_classes=num_classes,
    max_timesteps=max_timesteps,
    rnn_type=rnn_type,
    dropout_rate=dropout_rate,
    device=device
)


optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_lambda)
scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

# --- Training ---
_, history = fit(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=EPOCHS,
    patience=PATIENCE,
    criterion=criterion,
    optimizer=optimizer,
    scaler=scaler,
    device=device,
    l1_lambda=l1_lambda,
    l2_lambda=l2_lambda,
    experiment_name=f"{rnn_type}_H{hidden_size}_L{num_layers}_B{batch_size}_LR{lr}_DO{dropout_rate}_L1{l1_lambda}_L2{l2_lambda}"
)

label = f"{rnn_type}, H={hidden_size}, L={num_layers}, B={batch_size}, LR={lr}, DO={dropout_rate}, L1={l1_lambda}, L2={l2_lambda}"
final_f1 = history["val_f1"][-1]

results[label] = {
    "final_val_f1": final_f1,
    "train_loss": history["train_loss"],
    "val_loss": history["val_loss"],
    "train_f1": history["train_f1"],
    "val_f1": history["val_f1"]
}

print(f"Finished: Final Val F1 = {final_f1:.4f}")


# ------------------------------------------------------------
# PREDICTION (adapted to multi-input model)
# ------------------------------------------------------------
def predict(model, data_loader, device):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in data_loader:

            # Unpack multi-input batch (no labels in test)
            if len(batch) == 6:  
                x_num, pain, n_legs, n_hands, n_eyes, time_idx = [t.to(device) for t in batch]
            else:
                raise RuntimeError("Unexpected batch size in test loader.")

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(x_num, pain, n_legs, n_hands, n_eyes, time_idx)

            preds = logits.argmax(dim=1)
            all_predictions.append(preds.cpu().numpy())

    return np.concatenate(all_predictions)


# --- Predict on sliding windows ---
sequence_predictions = predict(model, test_loader, device)

# --- Aggregate predictions per sample_index ---
final_predictions = []
final_sids = []

for sid in np.unique(test_sids):
    votes = sequence_predictions[test_sids == sid]
    final_predictions.append(np.bincount(votes).argmax())
    final_sids.append(sid)

# --- Submission ---
submission_df = pd.DataFrame({
    "sample_index": final_sids,
    "label_code": final_predictions
})

reverse_labels = {0: "no_pain", 1: "low_pain", 2: "high_pain"}
submission_df["label"] = submission_df["label_code"].map(reverse_labels)
submission_df["sample_index"] = submission_df["sample_index"].astype(str).str.zfill(3)

submission_df = submission_df[["sample_index", "label"]]
submission_df.to_csv("submission_attention_multiinput.csv", index=False)

print("✅ Submission saved as submission_attention_multiinput.csv")
