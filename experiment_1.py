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

EPOCHS = 300
PATIENCE = 200


# --- 2. Fixed Hyperparameters for Comparison ---
RNN_TYPE = "GRU"
HIDDEN_SIZE = 64
NUM_LAYERS = 2
BATCH_SIZE = 32
LR_ADAM_BASED = 1e-3
LR_SGD = 1e-2 # Higher LR for SGD is typical
DROPOUT_RATE = 0.0
L1_LAMBDA = 1e-4
L2_LAMBDA = 1e-4 # Used for weight_decay in AdamW/Adam, or for L2 inside fit()

EPOCHS = 1000
PATIENCE = 200

# --- 3. Optimizer Configurations to Test ---
optimizer_configs = [
    {
        "name": "AdamW", 
        "optimizer_class": torch.optim.AdamW, 
        "lr": LR_ADAM_BASED, 
        "kwargs": {"weight_decay": L2_LAMBDA}
    },
    {
        "name": "Adam", 
        "optimizer_class": torch.optim.Adam, 
        "lr": LR_ADAM_BASED, 
        "kwargs": {} 
    },
    {
        "name": "SGD_w_Momentum", 
        "optimizer_class": torch.optim.SGD, 
        "lr": LR_SGD, 
        "kwargs": {"momentum": 0.9} 
    },
]

# --- 4. Main Experiment Loop ---

results = {}
best_f1_score = -1.0
best_model = None
best_config = None

# Recreate loaders based on fixed batch size
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

for config in optimizer_configs:
    opt_name = config['name']
    opt_class = config['optimizer_class']
    opt_lr = config['lr']
    opt_kwargs = config['kwargs']
    
    label = f"{RNN_TYPE}_H{HIDDEN_SIZE}_L{NUM_LAYERS}_B{BATCH_SIZE}_OPT{opt_name}_LR{opt_lr}_DO{DROPOUT_RATE}_L1{L1_LAMBDA}_L2{L2_LAMBDA}"
    print("\n" + "=" * 70)
    print(f"--- Running Experiment: {opt_name} ---")
    print(f"Hyperparams: LR={opt_lr}, L2_WD={opt_kwargs.get('weight_decay', L2_LAMBDA)}, Momentum={opt_kwargs.get('momentum', 'N/A')}")
    print("=" * 70)

    # Re-instantiate model for each run to ensure fresh weights
    model = build_model_attention_class(
        input_size=input_shape[-1],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout_rate=DROPOUT_RATE,
        rnn_type=RNN_TYPE,
        device=device
    )

    # Initialize Optimizer
    optimizer = opt_class(model.parameters(), lr=opt_lr, **opt_kwargs)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    # Train
    trained_model, history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        patience=PATIENCE,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        # L1 and L2 are passed to fit function for general handling
        l1_lambda=L1_LAMBDA,
        l2_lambda=L2_LAMBDA if 'weight_decay' not in opt_kwargs else 0.0, # Avoid double-dipping L2 if using AdamW's weight_decay
        experiment_name=label
    )

    final_f1 = history['val_f1'][-1]
    
    results[label] = {
        "final_val_f1": final_f1,
        "train_loss": history['train_loss'],
        "val_loss":   history['val_loss'],
        "train_f1":   history['train_f1'],
        "val_f1":     history['val_f1'],
        "config": config
    }
    
    print(f"Finished {opt_name}: Final Val F1 = {final_f1:.4f}")

    # Track best model
    if final_f1 > best_f1_score:
        best_f1_score = final_f1
        best_model = trained_model
        best_config = config



# --- 5. Analysis, Plotting, and Submission Generation ---

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plots_dir = f"results/optimizer_comparison_{timestamp}"
os.makedirs(plots_dir, exist_ok=True)

print("\n" + "#" * 70)
print(f"BEST CONFIGURATION FOUND (Optimizer: {best_config['name']}):")
print(f"Final Validation F1 Score: {best_f1_score:.4f}")
print(f"Saving results to: {plots_dir}")
print("#" * 70)

# ----------------- PLOTTING -----------------

# ----- LOSS PLOT (All Optimizers) -----
plt.figure(figsize=(10, 6))
for label, hist in results.items():
    opt_name = hist['config']['name']
    plt.plot(hist["val_loss"], linestyle="-", label=f"{opt_name} Val Loss (F1: {hist['final_val_f1']:.4f})")
    
plt.title(f"Validation Loss Comparison (Fixed Hyperparameters)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(fontsize=9)
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig(f"{plots_dir}/loss_comparison.png", dpi=200)
plt.close()

# ----- F1 PLOT (All Optimizers) -----
plt.figure(figsize=(10, 6))
for label, hist in results.items():
    opt_name = hist['config']['name']
    plt.plot(hist["val_f1"], linestyle="-", label=f"{opt_name} Val F1 (F1: {hist['final_val_f1']:.4f})")
    
plt.title(f"Validation F1 Score Comparison (Fixed Hyperparameters)")
plt.xlabel("Epoch")
plt.ylabel("F1 Score (Weighted)")
plt.legend(fontsize=9)
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig(f"{plots_dir}/f1_comparison.png", dpi=200)
plt.close()

print("Plots saved successfully ✅")


# ----------------- JSON CONFIG SAVE -----------------

# Note: We must redefine the fixed parameters here since this block is isolated.
# Assuming the necessary fixed variables (RNN_TYPE, HIDDEN_SIZE, etc.) are in scope 
# in the environment where this code is executed.
FIXED_PARAMS = {
    "RNN_TYPE": "GRU", "HIDDEN_SIZE": 64, "NUM_LAYERS": 2, "BATCH_SIZE": 32,
    "DROPOUT_RATE": 0.0, "L1_LAMBDA": 1e-4, "L2_LAMBDA": 1e-4, "EPOCHS": 1000,
    "PATIENCE": 200
}


best_config_dict = {
    "optimizer_name": best_config['name'],
    "rnn_type": FIXED_PARAMS["RNN_TYPE"],
    "hidden_size": FIXED_PARAMS["HIDDEN_SIZE"],
    "num_layers": FIXED_PARAMS["NUM_LAYERS"],
    "batch_size": FIXED_PARAMS["BATCH_SIZE"],
    "learning_rate": best_config['lr'],
    "dropout_rate": FIXED_PARAMS["DROPOUT_RATE"],
    "l1_lambda": FIXED_PARAMS["L1_LAMBDA"],
    "l2_lambda": FIXED_PARAMS["L2_LAMBDA"],
    "best_val_f1": best_f1_score,
    "optimizer_kwargs": best_config['kwargs'] 
}

with open(f"{plots_dir}/best_optimizer_config.json", "w") as f:
    json.dump(best_config_dict, f, indent=4)

print(f"Best config saved to {plots_dir}/best_optimizer_config.json")

