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


rnn_types = ["RNN", "GRU"]
hidden_sizes = [32, 64]
num_layers_list = [1, 2]
batch_sizes = [32, 64]
learning_rates = [1e-3, 5e-4, 1e-4]
dropout_rates = [0.0, 0.2]
l1_lambdas = [ 1e-5, 1e-4]
l2_lambdas = [ 1e-5, 1e-4]
EPOCHS = 500
PATIENCE = 100





experiments = list(product(
    rnn_types,
    hidden_sizes,
    num_layers_list,
    batch_sizes,
    learning_rates,
    dropout_rates,
    l1_lambdas,
    l2_lambdas
))


best_score = -1
best_config = None
best_model = None

for rnn_type, hidden_size, num_layers, batch_size, lr, dropout_rate, l1_lambda, l2_lambda in experiments:

    print(f"\n--- Running experiment: rnn_type={rnn_type}, hidden_size={hidden_size}, num_layers={num_layers}, batch_size={batch_size}, learning_rate={lr}, dropout_rate={dropout_rate}, l1_lambda={l1_lambda}, l2_lambda={l2_lambda} ---")

    # Recreate loaders based on batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Build model
    model = build_model_recurrent_class(
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

    # Track best
    if final_f1 > best_score:
        best_score = final_f1
        best_config = (rnn_type, hidden_size, num_layers, batch_size, lr, dropout_rate, l1_lambda, l2_lambda)
        best_model = model


# =========================================================================
# === NEW BLOCK: SELECT TOP 5 EXPERIMENTS FOR PLOTTING ===
# =========================================================================

# Sort the results by 'final_val_f1' in descending order
sorted_results = sorted(
    results.items(),
    key=lambda item: item[1]["final_val_f1"],
    reverse=True
)

# Select the top 5
top_5_results = dict(sorted_results[:5])
print(f"\nSelected the top {len(top_5_results)} experiments for plotting (best to worst F1 score):")
for i, (label, hist) in enumerate(sorted_results[:5]):
    print(f"  {i+1}. F1={hist['final_val_f1']:.4f} - {label}")

# =========================================================================


# ==== SAVE PLOTS TO DEDICATED FOLDER ====

# Create dedicated timestamped folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plots_dir = f"results/exp_{timestamp}"
os.makedirs(plots_dir, exist_ok=True)

print(f"\nSaving plots to: {plots_dir}")

# ----- LOSS PLOT (Top 5 only) -----
plt.figure(figsize=(12,8)) # Increased size for better readability
for label, hist in top_5_results.items(): # <<< PLOTTING top_5_results
    # Use the final F1 score in the label for context
    final_f1 = hist["final_val_f1"]
    plot_label = f"(F1: {final_f1:.4f}) {label}"

    plt.plot(hist["train_loss"], linestyle="-",  label=f"Train: {plot_label}")
    plt.plot(hist["val_loss"],   linestyle="--", label=f"Validation: {plot_label}")
    
plt.title(f"Loss Comparison Across Top {len(top_5_results)} Experiments")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{plots_dir}/loss_comparison_top5.png", dpi=200)
plt.close()

# ----- F1 PLOT (Top 5 only) -----
plt.figure(figsize=(12,8)) # Increased size for better readability
for label, hist in top_5_results.items(): # <<< PLOTTING top_5_results
    # Use the final F1 score in the label for context
    final_f1 = hist["final_val_f1"]
    plot_label = f"(F1: {final_f1:.4f}) {label}"
    
    plt.plot(hist["train_f1"], linestyle="-",  label=f"Train: {plot_label}")
    plt.plot(hist["val_f1"],   linestyle="--", label=f"Validation: {plot_label}")
    
plt.title(f"F1 Score Comparison Across Top {len(top_5_results)} Experiments")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{plots_dir}/f1_comparison_top5.png", dpi=200)
plt.close()

print("Plots saved successfully ✅")


best_config_dict = {
    "rnn_type": best_config[0],
    "hidden_size": best_config[1],
    "num_layers": best_config[2],
    "batch_size": best_config[3],
    "learning_rate": best_config[4],
    "dropout_rate": best_config[5],
    "l1_lambda": best_config[6],
    "l2_lambda": best_config[7],
    "best_val_f1": best_score
}

with open(f"{plots_dir}/best_config.json", "w") as f:
    json.dump(best_config_dict, f, indent=4)

print(f"Best config saved to {plots_dir}/best_config.json")