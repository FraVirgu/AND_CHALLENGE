from import_file import *

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

train_id = unique_id[:int(len(unique_id)*0.8)]
val_id   = unique_id[int(len(unique_id)*0.8):]

df_train = df[df['sample_index'].isin(train_id)].copy()
df_val   = df[df['sample_index'].isin(val_id)].copy()

y_train_table = df_label[df_label['sample_index'].isin(train_id)].copy()
y_val_table   = df_label[df_label['sample_index'].isin(val_id)].copy()

# Re-run your data normalization with a check
feature_cols = [c for c in df.columns if c not in ['sample_index','time','label', 'joint_30']]
#train data
mean = df_train[feature_cols].mean()
std = df_train[feature_cols].std()

# Find columns with zero standard deviation
zero_std_cols = std[std == 0].index.tolist()
print(f"Columns with zero standard deviation (will be removed/handled): {zero_std_cols}")

# FIX: Add a small epsilon to standard deviation to prevent division by zero,
# or remove the constant features. Adding a small epsilon is often safer.
epsilon = 1e-4

df_train[feature_cols] = (df_train[feature_cols] - mean + epsilon) / (std + epsilon)
df_val[feature_cols]   = (df_val[feature_cols]   - mean + epsilon) / (std + epsilon)


df_train_merge = df_train.merge(y_train_table[['sample_index', 'label']], on='sample_index', how='left')
df_val_merge   = df_val.merge(y_val_table[['sample_index', 'label']], on='sample_index', how='left')


#test data
mean_test = df_test[feature_cols].mean()
std_test = df_test[feature_cols].std()

zero_std_cols = std_test[std == 0].index.tolist()
df_test[feature_cols] = (df_test[feature_cols] - mean_test + epsilon) / (std_test + epsilon)




def build_sequences(df, window=40, stride=20):
    sequences = []
    labels = []
    for sid in df['sample_index'].unique():
        temp = df[df['sample_index'] == sid][feature_cols].values
        label = df[df['sample_index'] == sid]['label'].iloc[0]
        idx = 0
        while idx + window <= len(temp):
            sequences.append(temp[idx:idx+window])
            labels.append(label)
            idx += stride
    return np.array(sequences), np.array(labels)

X_train, y_train = build_sequences(df_train_merge, window=40, stride=20)
X_val,   y_val   = build_sequences(df_val_merge,   window=40, stride=20)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)



def build_test_sequences(df, window=40, stride=20):
    sequences = []
    sample_indices = []
    # Loop over all unique samples in the test dataframe
    for sid in df['sample_index'].unique():
        temp = df[df['sample_index'] == sid][feature_cols].values
        idx = 0
        while idx + window <= len(temp):
            sequences.append(temp[idx:idx+window])
            # Store the sample_index for each generated sequence
            sample_indices.append(sid)
            idx += stride
    return np.array(sequences), np.array(sample_indices)



X_test, test_sequence_sids = build_test_sequences(df_test, window=40, stride=20)

print("X_test shape:", X_test.shape)
print("Test sequence SIDs shape:", test_sequence_sids.shape)


# Define the input shape based on the training data
input_shape = X_train.shape[1:]

# Define the number of classes based on the categorical labels
num_classes = len(np.unique(y_train))


def make_loader(ds, batch_size, shuffle, drop_last):
    # Determine optimal number of worker processes for data loading
    cpu_cores = os.cpu_count() or 2
    num_workers = max(2, min(4, cpu_cores))

    # Create DataLoader with performance optimizations
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        pin_memory_device="cuda" if torch.cuda.is_available() else "",
        prefetch_factor=4,  # Load 4 batches ahead
    )


# Convert arrays to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

# Create TensorDataset
train_ds = TensorDataset(X_train, y_train)
val_ds   = TensorDataset(X_val, y_val)

# Create DataLoaders using make_loader
train_loader = make_loader(train_ds, batch_size=32, shuffle=True, drop_last=True)
val_loader   = make_loader(val_ds,   batch_size=32, shuffle=False, drop_last=False)




X_test, test_sequence_sids = build_test_sequences(df_test, window=40, stride=20)

print("X_test shape:", X_test.shape)
print("Test sequence SIDs shape:", test_sequence_sids.shape)

# Convert array to tensor
X_test = torch.tensor(X_test, dtype=torch.float32)

# Create TensorDataset (only features, no labels)
test_ds = TensorDataset(X_test)

# Create DataLoader using make_loader (shuffle=False for consistent results)
# Batch size should be the same or a multiple of 32 for efficiency
test_loader = make_loader(test_ds, batch_size=32, shuffle=False, drop_last=False)# Convert array to tensor


X_test = torch.tensor(X_test, dtype=torch.float32)

# Create TensorDataset (only features, no labels)
test_ds = TensorDataset(X_test)

# Create DataLoader using make_loader (shuffle=False for consistent results)
# Batch size should be the same or a multiple of 32 for efficiency
test_loader = make_loader(test_ds, batch_size=32, shuffle=False, drop_last=False)