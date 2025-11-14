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
categorical_cols = [
    'time',
    'pain_survey_1', 'pain_survey_2', 'pain_survey_3', 'pain_survey_4',
    'n_legs', 'n_hands', 'n_eyes'
]

# Normalize ONLY continuous columns
feature_cols = [
    c for c in df.columns
    if c not in ['sample_index', 'label', 'joint_30'] + categorical_cols
]
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



def build_sequences_multi(df, window=WINDOW_SIZE, stride=STRIDE, is_test=False):
    X_num_seq = []
    X_pain_seq = []
    X_legs_seq = []
    X_hands_seq = []
    X_eyes_seq = []
    X_time_seq = []
    y_seq = []

    for sid in df['sample_index'].unique():
        temp = df[df['sample_index'] == sid]

        num_vals = temp[feature_cols].values
        pain_vals = temp[['pain_survey_1','pain_survey_2','pain_survey_3','pain_survey_4']].values
        legs_vals = temp['n_legs'].values
        hands_vals = temp['n_hands'].values
        eyes_vals = temp['n_eyes'].values
        time_vals = temp['time'].values

        label = None if is_test else temp['label'].iloc[0]

        for i in range(0, len(temp) - window + 1, stride):
            X_num_seq.append(num_vals[i:i+window])
            X_pain_seq.append(pain_vals[i:i+window])
            X_legs_seq.append(legs_vals[i:i+window])
            X_hands_seq.append(hands_vals[i:i+window])
            X_eyes_seq.append(eyes_vals[i:i+window])
            X_time_seq.append(time_vals[i:i+window])
            y_seq.append(label if not is_test else sid)

    return (
        np.array(X_num_seq),
        np.array(X_pain_seq),
        np.array(X_legs_seq),
        np.array(X_hands_seq),
        np.array(X_eyes_seq),
        np.array(X_time_seq),
        np.array(y_seq)
    )



# --- Build datasets ---
(
    X_num_train,
    X_pain_train,
    n_legs_train,
    n_hands_train,
    n_eyes_train,
    time_idx_train,
    y_train
) = build_sequences_multi(df_train_merge, is_test=False)

(
    X_num_val,
    X_pain_val,
    n_legs_val,
    n_hands_val,
    n_eyes_val,
    time_idx_val,
    y_val
) = build_sequences_multi(df_val_merge, is_test=False)

(
    X_num_test,
    X_pain_test,
    n_legs_test,
    n_hands_test,
    n_eyes_test,
    time_idx_test,
    test_sids
) = build_sequences_multi(df_test, is_test=True)




# Define the input shape based on the training data
input_size_numeric = X_num_train.shape[-1]


# Define the number of classes based on the categorical labels
num_classes = len(np.unique(y_train))

# --- Convert to Tensors ---
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


# --- TensorDatasets ---
train_dataset = TensorDataset(
    X_num_train_tensor,
    pain_train_tensor,
    n_legs_train_tensor,
    n_hands_train_tensor,
    n_eyes_train_tensor,
    time_idx_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    X_num_val_tensor,
    pain_val_tensor,
    n_legs_val_tensor,
    n_hands_val_tensor,
    n_eyes_val_tensor,
    time_idx_val_tensor,
    y_val_tensor
)

test_dataset = TensorDataset(
    X_num_test_tensor,
    pain_test_tensor,
    n_legs_test_tensor,
    n_hands_test_tensor,
    n_eyes_test_tensor,
    time_idx_test_tensor
)





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