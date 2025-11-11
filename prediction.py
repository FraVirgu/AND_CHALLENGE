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


# --- 4. Generate Predictions ---
# rnn_model should be the trained model object loaded with the best weights
# If your training cell ran successfully, this model object holds the weights
sequence_predictions = predict(rnn_model, test_loader, device)

# --- 5. Aggregate Predictions (Mode) ---
final_predictions = []
final_sids = []

for sid in np.unique(test_sequence_sids):
    # Filter predictions belonging to the current sample_index
    sample_preds = sequence_predictions[test_sequence_sids == sid]

    # Calculate the mode (most frequent prediction) for the sample's sequences
    most_frequent_pred = np.bincount(sample_preds).argmax()


    final_predictions.append(most_frequent_pred)
    final_sids.append(sid)

# --- 6. Create and Save Submission File ---
submission_df = pd.DataFrame({
    'sample_index': final_sids,
    'label_code': final_predictions # Temporary column for numerical label
})

# Reverse mapping based on your definition:
# 0: 'no_pain', 1: 'low_pain', 2: 'high_pain'
reverse_labels = {0: 'no_pain', 1: 'low_pain', 2: 'high_pain'}

# Map the numerical predictions back to the required string labels
submission_df['label'] = submission_df['label_code'].map(reverse_labels)

# Select only the required columns and format the index
submission_df['sample_index'] = submission_df['sample_index'].astype(str).str.zfill(3)
submission_df = submission_df[['sample_index', 'label']]

# Save the submission file to the correct directory
submission_df.to_csv('submission_gru_model.csv', index=False)