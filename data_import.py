from import_file import *
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

# Configure plot display settings
sns.set(font_scale=1.4)
sns.set_style('white')
plt.rc('font', size=14)


# Read the dataset into a DataFrame without a header initially
df = pd.read_csv('DATA/pirate_pain_train.csv', header=None)
df_test = pd.read_csv('DATA/pirate_pain_test.csv', header=None)
df_label = pd.read_csv('DATA/pirate_pain_train_labels.csv', header=None)

df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)

df_label.columns = df_label.iloc[0]
df_label = df_label[1:].reset_index(drop=True)

df_test.columns = df_test.iloc[0]
df_test = df_test[1:].reset_index(drop=True)

df = df.dropna()
df_label = df_label.dropna()
df_test = df_test.dropna()

df['sample_index'] = df['sample_index'].astype(int)
df_label['sample_index'] = df_label['sample_index'].astype(int)
df_test['sample_index'] = df_test['sample_index'].astype(int)


df = df.apply(pd.to_numeric, errors='ignore')
df_test = df_test.apply(pd.to_numeric, errors='ignore')

number_mapping_nlegs = {'two': 2, 'one+peg_leg': 1}
number_mapping_nhands = {'two': 2, 'one+hook_hand': 1}
number_mapping_neyes = {'two': 2, 'one+eye_patch': 1}

df['n_legs'] = df['n_legs'].replace(number_mapping_nlegs).astype(int)
df['n_hands'] = df['n_hands'].replace(number_mapping_nhands).astype(int)
df['n_eyes'] = df['n_eyes'].replace(number_mapping_neyes).astype(int)

df_test['n_legs'] = df_test['n_legs'].replace(number_mapping_nlegs).astype(int)
df_test['n_hands'] = df_test['n_hands'].replace(number_mapping_nhands).astype(int)
df_test['n_eyes'] = df_test['n_eyes'].replace(number_mapping_neyes).astype(int)


training_labels = {'no_pain': 0, 'low_pain': 1, 'high_pain': 2}
df_label['label'] = df_label['label'].replace(training_labels).astype(int)


unique_id = df['sample_index'].unique()
random.shuffle(unique_id)
PERCENTAGE_SPLIT = 0.8
train_id = unique_id[:int(len(unique_id)*PERCENTAGE_SPLIT)]
val_id   = unique_id[int(len(unique_id)*PERCENTAGE_SPLIT):]

df_train = df[df['sample_index'].isin(train_id)].copy()
df_val   = df[df['sample_index'].isin(val_id)].copy()

y_train_table = df_label[df_label['sample_index'].isin(train_id)].copy()
y_val_table   = df_label[df_label['sample_index'].isin(val_id)].copy()

# Re-run your data normalization with a check
'''
joint_outlier_cols = [f'joint_{i:02}' for i in range(13, 26)]
for col in joint_outlier_cols:
    lower = df_train[col].quantile(0.001)
    upper = df_train[col].quantile(0.999)
    df_train[col] = df_train[col].clip(lower, upper)
    df_val[col]   = df_val[col].clip(lower, upper)
    df_test[col]  = df_test[col].clip(lower, upper)
'''





# --- 4. Robust Normalization (Using Median and IQR) ---

# Define all feature columns (excluding identifiers and labels)
feature_cols = [c for c in df.columns if c not in ['sample_index','time','label', 'joint_30']]
mean = df_train[feature_cols].mean()
std = df_train[feature_cols].std()

# Use IQR (Interquartile Range) for scaling, adding epsilon for safety
epsilon = 1e-4

print("Applying Robust Scaling (Median and IQR) to all features...")

# 4b. Apply Robust Scaling to Train and Validation Data
# Scaling formula: (X - Median) / IQR
df_train[feature_cols] = (df_train[feature_cols] - mean + epsilon) / (std + epsilon)
df_val[feature_cols]   = (df_val[feature_cols]   - mean + epsilon) / (std + epsilon)


# Merge labels back into split dataframes
df_train_merge = df_train.merge(y_train_table[['sample_index', 'label']], on='sample_index', how='left')
df_val_merge   = df_val.merge(y_val_table[['sample_index', 'label']], on='sample_index', how='left')


# 4c. Apply Robust Scaling to Test Data

mean_test = df_test[feature_cols].mean()
std_test = df_test[feature_cols].std()

zero_std_cols = std_test[std == 0].index.tolist()
df_test[feature_cols] = (df_test[feature_cols] - mean_test + epsilon) / (std_test + epsilon)




# --- Config ---
WINDOW_SIZE = 40
STRIDE = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


# --- Sequence Builder ---

def build_sequences(df, feature_cols, window=WINDOW_SIZE, stride=STRIDE, is_test=False):
    """
    Builds sliding window sequences from dataframe samples.
    """
    sequences, targets = [], []

    for sid in df['sample_index'].unique():
        temp = df[df['sample_index'] == sid][feature_cols].values
        label = None if is_test else df[df['sample_index'] == sid]['label'].iloc[0]

        for idx in range(0, len(temp) - window + 1, stride):
            sequences.append(temp[idx:idx + window])
            targets.append(sid if is_test else label)

    return np.array(sequences), np.array(targets)


# --- Build datasets ---
X_train, y_train = build_sequences(df_train_merge, feature_cols)
X_val,   y_val   = build_sequences(df_val_merge,   feature_cols)
X_test,  test_sids = build_sequences(df_test, feature_cols, is_test=True)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test:  {X_test.shape}, test_sids: {test_sids.shape}")

# Define the input shape based on the training data
input_shape = X_train.shape[1:]

# Define the number of classes based on the categorical labels
num_classes = len(np.unique(y_train))

# --- Convert to Tensors ---
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor   = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val, dtype=torch.long)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)

# --- TensorDatasets ---
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset   = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset  = TensorDataset(X_test_tensor)




# --- DataLoader helper ---
def make_loader(dataset, batch_size=BATCH_SIZE, sampler=None, shuffle=False, drop_last=False):
    cpu_cores = os.cpu_count() or 2
    num_workers = max(2, min(4, cpu_cores))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        pin_memory_device="cuda" if torch.cuda.is_available() else "",
        prefetch_factor=4,
    )


# --- DataLoaders ---
train_loader = make_loader(train_dataset, batch_size= BATCH_SIZE ,sampler=sampler, drop_last=True)
val_loader   = make_loader(val_dataset)
test_loader  = make_loader(test_dataset)

print("✅ DataLoaders and class balancing setup complete.")


# --- Class Balancing ---

y_train_np = np.array(y_train)
y_train_np = y_train_np.astype(int)  # force integer labels


print("y_train dtype:", type(y_train))
print("Unique values:", np.unique(y_train))

classes = np.unique(y_train_np)
print("Classes found:", classes)

class_weights = compute_class_weight(class_weight='balanced',
                                     classes=classes,
                                     y=y_train_np)

weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# Compute sample weights based on inverse class frequency
class_counts = np.bincount(y_train)
class_weights = 1. / class_counts
sample_weights = class_weights[y_train]

sampler = WeightedRandomSampler(weights=sample_weights,
                                num_samples=len(sample_weights),
                                replacement=True)