import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Subset

from model_train import train_one_epoch, validate_one_epoch
from model import AttentionClassifier     # <-- your new optimized model
from data_import import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# CONFIG
# ============================================================
rnn_type     = "GRU"
hidden_size  = 48
num_layers   = 2
dropout_rate = 0.45
batch_size   = 32
lr           = 3e-4
l1_lambda    = 0.0
l2_lambda    = 5e-4
EPOCHS       = 300
PATIENCE     = 60
K_FOLDS      = 5

criterion = torch.nn.CrossEntropyLoss()

# ============================================================
# Prepare Label Vector
# train_dataset returns:
#  (x_num, pain, n_legs, n_hands, n_eyes, time_idx, y)
# ============================================================
X = np.arange(len(train_dataset))
y = np.array([train_dataset[i][-1] for i in range(len(train_dataset))])

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

input_size_numeric = X_num_train.shape[-1]
num_classes = 3
max_timesteps = int(time_idx_train.max()) + 1

fold_models = []
fold_scores = []

# ============================================================
# K-FOLD LOOP
# ============================================================
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n==================== FOLD {fold+1}/{K_FOLDS} ====================")

    train_subset = Subset(train_dataset, train_idx)
    val_subset   = Subset(train_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_subset, batch_size=batch_size)

    # Model
    model = AttentionClassifier(
        input_size_numeric=input_size_numeric,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        max_timesteps=max_timesteps,
        rnn_type=rnn_type,
        dropout_rate=dropout_rate,
        bidirectional=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_lambda)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_f1 = -1
    best_state = None
    patience_ctr = 0

    # ============================================================
    # Training Loop
    # ============================================================
    for epoch in range(1, EPOCHS+1):

        train_loss, train_f1 = train_one_epoch(
            model=model,
            train_loader=train_loader,       # ✅ FIXED
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            l1_lambda=l1_lambda,
            l2_lambda=l2_lambda
        )

        val_loss, val_f1 = validate_one_epoch(
            model=model,
            val_loader=val_loader,           # ⬅ must match your evaluate() signature
            criterion=criterion,
            device=device
        )


        print(f"Fold {fold+1} | Epoch {epoch:03d}/{EPOCHS} | "
              f"Train Loss {train_loss:.4f} F1 {train_f1:.4f} | "
              f"Val Loss {val_loss:.4f} F1 {val_f1:.4f}")

        # Track best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print("⏸ Early stopping.")
                break

    print(f"✅ Fold {fold+1} best F1 = {best_val_f1:.4f}")
    fold_scores.append(best_val_f1)
    fold_models.append(best_state)

# ============================================================
# SHOW RESULTS
# ============================================================
print("\n=============================================")
print(" K-FOLD TRAINING COMPLETED")
print(" Average validation F1:", np.mean(fold_scores))
print("=============================================\n")

# ============================================================
# LOAD ENSEMBLE MODELS
# ============================================================
models = []
for state in fold_models:
    m = AttentionClassifier(
        input_size_numeric=input_size_numeric,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        max_timesteps=max_timesteps,
        rnn_type=rnn_type,
        dropout_rate=dropout_rate,
        bidirectional=True
    ).to(device)
    m.load_state_dict(state)
    m.eval()
    models.append(m)

print(f"Loaded {len(models)} models for ensemble.")


# ============================================================
# ENSEMBLE PREDICTION
# ============================================================
def ensemble_predict(models, data_loader):
    out = []

    with torch.no_grad():
        for batch in data_loader:
            x_num, pain, n_legs, n_hands, n_eyes, time_idx = [t.to(device) for t in batch]

            logits_sum = torch.zeros((x_num.size(0), num_classes), device=device)

            for m in models:
                with torch.amp.autocast(device_type=device.type, enabled=(device.type=="cuda")):
                    logits_sum += m(x_num, pain, n_legs, n_hands, n_eyes, time_idx)

            preds = logits_sum.argmax(dim=1)
            out.append(preds.cpu().numpy())

    return np.concatenate(out)


print("Running ensemble prediction on TEST...")
test_loader = DataLoader(test_dataset, batch_size=batch_size)
sequence_predictions = ensemble_predict(models, test_loader)


# ============================================================
# AGGREGATE BY SAMPLE ID
# ============================================================
final_predictions = []
final_sids = []

for sid in np.unique(test_sids):
    votes = sequence_predictions[test_sids == sid]
    final_predictions.append(np.bincount(votes).argmax())
    final_sids.append(sid)


# ============================================================
# SAVE SUBMISSION
# ============================================================
submission_df = pd.DataFrame({
    "sample_index": final_sids,
    "label_code": final_predictions
})

reverse_labels = {0: "no_pain", 1: "low_pain", 2: "high_pain"}
submission_df["label"] = submission_df["label_code"].map(reverse_labels)
submission_df["sample_index"] = submission_df["sample_index"].astype(str).str.zfill(3)
submission_df = submission_df[["sample_index", "label"]]

submission_df.to_csv("submission_attention_kfold_ensemble.csv", index=False)

print("✅ Saved submission_attention_kfold_ensemble.csv")
