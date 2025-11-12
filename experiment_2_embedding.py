from itertools import product
from model_train import *
from data_import import *
import os
from datetime import datetime
import json
import torch.optim
from torch.utils.data import DataLoader
from model_embedding import HybridAttentionClassifier


import torch.nn.functional as F

# --- Build multi-input sequences ---
(
    X_num_train, X_pain_train, n_legs_train, n_hands_train, n_eyes_train, time_idx_train, y_train
) = build_sequences_multi(df_train_merge)

(
    X_num_val, X_pain_val, n_legs_val, n_hands_val, n_eyes_val, time_idx_val, y_val
) = build_sequences_multi(df_val_merge)

(
    X_num_test, X_pain_test, n_legs_test, n_hands_test, n_eyes_test, time_idx_test, _
) = build_sequences_multi(df_test, is_test=True)


# Convert to tensors
X_num_train_tensor   = torch.tensor(X_num_train, dtype=torch.float32)
pain_train_tensor    = torch.tensor(X_pain_train, dtype=torch.long)
n_legs_train_tensor  = torch.tensor(n_legs_train, dtype=torch.long)
n_hands_train_tensor = torch.tensor(n_hands_train, dtype=torch.long)
n_eyes_train_tensor  = torch.tensor(n_eyes_train, dtype=torch.long)
time_idx_train_tensor= torch.tensor(time_idx_train, dtype=torch.long)
y_train_tensor       = torch.tensor(y_train, dtype=torch.long)

X_num_val_tensor   = torch.tensor(X_num_val, dtype=torch.float32)
pain_val_tensor    = torch.tensor(X_pain_val, dtype=torch.long)
n_legs_val_tensor  = torch.tensor(n_legs_val, dtype=torch.long)
n_hands_val_tensor = torch.tensor(n_hands_val, dtype=torch.long)
n_eyes_val_tensor  = torch.tensor(n_eyes_val, dtype=torch.long)
time_idx_val_tensor= torch.tensor(time_idx_val, dtype=torch.long)
y_val_tensor       = torch.tensor(y_val, dtype=torch.long)

X_num_test_tensor   = torch.tensor(X_num_test, dtype=torch.float32)
pain_test_tensor    = torch.tensor(X_pain_test, dtype=torch.long)
n_legs_test_tensor  = torch.tensor(n_legs_test, dtype=torch.long)
n_hands_test_tensor = torch.tensor(n_hands_test, dtype=torch.long)
n_eyes_test_tensor  = torch.tensor(n_eyes_test, dtype=torch.long)
time_idx_test_tensor= torch.tensor(time_idx_test, dtype=torch.long)


# Training set
train_dataset = TensorDataset(
    X_num_train_tensor,
    pain_train_tensor,
    n_legs_train_tensor,
    n_hands_train_tensor,
    n_eyes_train_tensor,
    time_idx_train_tensor,
    y_train_tensor
)

# Validation set
val_dataset = TensorDataset(
    X_num_val_tensor,
    pain_val_tensor,
    n_legs_val_tensor,
    n_hands_val_tensor,
    n_eyes_val_tensor,
    time_idx_val_tensor,
    y_val_tensor
)

# Test set (no labels)
test_dataset = TensorDataset(
    X_num_test_tensor,
    pain_test_tensor,
    n_legs_test_tensor,
    n_hands_test_tensor,
    n_eyes_test_tensor,
    time_idx_test_tensor
)




print("Pain surveys unique values:", np.unique(X_pain_train))
print("n_legs unique:", np.unique(n_legs_train))
print("n_hands unique:", np.unique(n_hands_train))
print("n_eyes unique:", np.unique(n_eyes_train))
print("time_idx unique:", np.unique(time_idx_train))


max_timesteps = int(time_idx_train_tensor.max().item()) + 1  # → 160


rnn_type = "GRU"
hidden_size =  64
num_layers =  2
batch_size = 32
lr = 1e-3
dropout_rate = 0.1
l1_lambda =  1e-4
l2_lambda =  1e-4

EPOCHS = 200
PATIENCE = 50

best_score = -1
best_config = None
best_model = None


print(f"\n--- Running experiment: rnn_type={rnn_type}, hidden_size={hidden_size}, num_layers={num_layers}, batch_size={batch_size}, learning_rate={lr}, dropout_rate={dropout_rate}, l1_lambda={l1_lambda}, l2_lambda={l2_lambda} ---")


# ------------------------------------------------------------
# Continue training and evaluation from your setup above
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Create DataLoaders ---
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- Build model ---
input_size_numeric = X_num_train_tensor.shape[-1]
num_classes = len(torch.unique(y_train_tensor))

model = HybridAttentionClassifier(
    input_size_numeric=input_size_numeric,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_classes=num_classes,
    max_timesteps=max_timesteps,
    rnn_type=rnn_type,
    dropout_rate=dropout_rate,
    bidirectional=False
).to(device)

print(model)

# --- Optimizer and loss ---
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_lambda)
scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))


# ------------------------------------------------------------
# Training and validation loops
# ------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for batch in dataloader:
        (
            x_num, pain, n_legs, n_hands, n_eyes, time_idx, targets
        ) = [t.to(device) for t in batch]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            logits = model(x_num, pain, n_legs, n_hands, n_eyes, time_idx)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x_num.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for batch in dataloader:
            (
                x_num, pain, n_legs, n_hands, n_eyes, time_idx, targets
            ) = [t.to(device) for t in batch]

            logits = model(x_num, pain, n_legs, n_hands, n_eyes, time_idx)
            loss = criterion(logits, targets)

            total_loss += loss.item() * x_num.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

    return total_loss / total_samples, total_correct / total_samples


# ------------------------------------------------------------
# Training loop with early stopping
# ------------------------------------------------------------
best_val_acc = 0
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⏸ Early stopping triggered.")
            break

print(f"✅ Training completed. Best Val Accuracy: {best_val_acc:.4f}")

# ------------------------------------------------------------
# Load best model and predict on test set
# ------------------------------------------------------------
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

all_preds = []
with torch.no_grad():
    for batch in test_loader:
        x_num, pain, n_legs, n_hands, n_eyes, time_idx = [t.to(device) for t in batch]
        logits = model(x_num, pain, n_legs, n_hands, n_eyes, time_idx)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())

all_preds = np.concatenate(all_preds)

# ------------------------------------------------------------
# Aggregate predictions per sample_index
# ------------------------------------------------------------
final_predictions = []
final_sids = []

for sid in np.unique(test_sids):
    sample_preds = all_preds[test_sids == sid]
    most_frequent_pred = np.bincount(sample_preds).argmax()
    final_predictions.append(most_frequent_pred)
    final_sids.append(sid)

# ------------------------------------------------------------
# Create submission file
# ------------------------------------------------------------
submission_df = pd.DataFrame({
    'sample_index': final_sids,
    'label_code': final_predictions
})

reverse_labels = {0: 'no_pain', 1: 'low_pain', 2: 'high_pain'}
submission_df['label'] = submission_df['label_code'].map(reverse_labels)
submission_df['sample_index'] = submission_df['sample_index'].astype(str).str.zfill(3)
submission_df = submission_df[['sample_index', 'label']]
submission_df.to_csv('submission_hybrid_model.csv', index=False)

print("✅ Submission saved as submission_hybrid_model.csv")



