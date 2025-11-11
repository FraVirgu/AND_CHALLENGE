from itertools import product
from model_train import *
from data_import import *
import os
from datetime import datetime
import json
import torch.optim
from torch.utils.data import DataLoader
# Assuming necessary imports like torch, DataLoader, etc., are available
# from model_train import fit, build_model
# from data_import import train_ds, val_ds, input_shape, num_classes, device, EPOCHS, PATIENCE, criterion


import matplotlib.pyplot as plt

results = {}   # will store: label → history dict

'''
OPTIONS TO EXPLORE:
rnn_types = ["RNN", "GRU", "LSTM"]
hidden_sizes = [32, 64]
num_layers_list = [1, 2, 3]
batch_sizes = [32, 64]
learning_rates = [1e-3, 5e-4, 1e-4]
dropout_rates = [0.0, 0.2, 0.4]
l1_lambdas = [0.0, 1e-5, 1e-4]
l2_lambdas = [0.0, 1e-5, 1e-4]
'''


rnn_type = "GRU"
hidden_size =  64
num_layers =  2
batch_size = 32
lr = 1e-3
dropout_rate = 0.0
l1_lambda =  1e-4
l2_lambda =  1e-4

EPOCHS = 1000
PATIENCE = 200




best_score = -1
best_config = None
best_model = None


print(f"\n--- Running experiment: rnn_type={rnn_type}, hidden_size={hidden_size}, num_layers={num_layers}, batch_size={batch_size}, learning_rate={lr}, dropout_rate={dropout_rate}, l1_lambda={l1_lambda}, l2_lambda={l2_lambda} ---")

# Recreate loaders based on batch size
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

# Build model
model = build_model_attention_class(
    input_size=input_shape[-1],
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_classes=num_classes,
    dropout_rate=dropout_rate,
    rnn_type=rnn_type,
    device=device
)

# Optimizer using current learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_lambda)
scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

# Train
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

# Store results including the final score for easy sorting
final_f1 = history['val_f1'][-1]
results[label] = {
    "final_val_f1": final_f1, # NEW: Added final score here
    "train_loss": history['train_loss'],
    "val_loss":   history['val_loss'],
    "train_f1":   history['train_f1'],
    "val_f1":     history['val_f1']
}

print(f"Finished: Final Val F1 = {final_f1:.4f}")


from model_train import *

def predict(model, data_loader, device):
    """
    Generate predictions for the given data loader.
    """
    model.eval()  # Set model to evaluation mode
    all_predictions = []

    with torch.no_grad():
        for batch_data in data_loader:
            # The input is the first (and only) item in the TensorDataset/DataLoader
            inputs = batch_data[0]
            inputs = inputs.float().to(device)

            # Use autocast context if CUDA is available, otherwise it defaults to no change
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(inputs)

            predictions = logits.argmax(dim=1)
            all_predictions.append(predictions.cpu().numpy())

    return np.concatenate(all_predictions)

# 1. Predict on test set
sequence_predictions = predict(model, test_loader, device)

# 2. Aggregate predictions at sample_index level
final_predictions = []
final_sids = []

for sid in np.unique(test_sequence_sids):
    sample_preds = sequence_predictions[test_sequence_sids == sid]
    most_frequent_pred = np.bincount(sample_preds).argmax()
    final_predictions.append(most_frequent_pred)
    final_sids.append(sid)

# 3. Build submission
submission_df = pd.DataFrame({
    'sample_index': final_sids,
    'label_code': final_predictions
})

reverse_labels = {0: 'no_pain', 1: 'low_pain', 2: 'high_pain'}
submission_df['label'] = submission_df['label_code'].map(reverse_labels)

submission_df['sample_index'] = submission_df['sample_index'].astype(str).str.zfill(3)
submission_df = submission_df[['sample_index', 'label']]
submission_df.to_csv('submission_gru_model.csv', index=False)

print("✅ Submission saved as submission_gru_model.csv")