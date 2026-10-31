# In case of plotting Scatter plots, uncomment the section (it takes time to plot)

#%% Importing Libraries
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import os
import json
import seaborn as sns

#%% Config
file_path_data = 'Trials_list.json'

if os.path.isfile(file_path_data):
    with open(file_path_data, 'r') as f:
        data_list = json.load(f)
else:
    data_list = []

#%% Preprocessing
# X_train and X_test contain feature data
scaler = StandardScaler()

# noise = torch.rand(614, 8) * 0.001
# Read the CSV file into a DataFrame
# data file is in the dataset folder next to the script
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'diabetes.csv')
data = pd.read_csv(file_path)

# Separate the last column (outcome) and convert it to a PyTorch tensor
y = torch.tensor(data.iloc[:, -1].values)

# Extract the rest of the data without the last column and convert to PyTorch tensor
X = torch.tensor(data.iloc[:, :-1].values)
#state = np.random.randint(1, 5001)
state = 974 #reducability
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=state)

X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

X_train_normalized = torch.tensor(X_train_normalized, dtype=torch.float)
X_test_normalized = torch.tensor(X_test_normalized, dtype=torch.float)

#%% Model
# Model Structure =============================================================
class Classification(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(in_features=8, out_features=8)
        self.layer_1 = nn.Linear(in_features=8, out_features=16)
        self.layer_2 = nn.Linear(in_features=16, out_features=32)
        self.layer_3 = nn.Linear(in_features=32, out_features=64)
        self.layer_4 = nn.Linear(in_features=64, out_features=64)
        self.layer_5 = nn.Linear(in_features=64, out_features=32)
        self.layer_6 = nn.Linear(in_features=32, out_features=16)
        self.dropout = nn.Dropout(p=0.5)
        self.layer_7 = nn.Linear(in_features=16, out_features=8)
        self.layer_8 = nn.Linear(in_features=8, out_features=1)
     # Define Forward Path   
    def forward(self, X):
        X = torch.relu(self.input_layer(X))
        X = torch.relu(self.layer_1(X))
        # X = self.dropout(X)
        X = torch.relu(self.layer_2(X))
        # X = self.dropout(X)
        # X = torch.relu(self.layer_3(X))
        # X = torch.relu(self.layer_4(X))
        # X = torch.relu(self.layer_5(X))
        # X = self.dropout(X)
        X = torch.relu(self.layer_6(X))
        X = self.dropout(X)
        X = torch.relu(self.layer_7(X))
        X = self.layer_8(X)
        # X = self.dropout(X)
        return X
# =============================================================================

#%% Setup
torch.manual_seed(state)
model = Classification()  # Creating an instance of the Classification model

loss_fn = nn.BCEWithLogitsLoss()
lr = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

epochs = 7000
train_losses = []
test_losses = []
min_test_loss = float('inf')

#%% Training & Evaluation
# with torch.autograd.detect_anomaly():
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    
    optimizer.zero_grad()
        # Ensure X_train is converted to float tensor
    y_pred = model(X_train_normalized.float())
        # Squeeze the output to match y_train shape
    loss = loss_fn(y_pred.squeeze(), y_train.float())
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    y_pred_binary = torch.round(torch.sigmoid(y_pred)).float()
    correct_train = (y_pred_binary.squeeze().long()
                         == y_train.long()).sum().item()
    accuracy_train = correct_train / len(y_train)

# Evaluation ==================================================================
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Ensure X_test is converted to float tensor
        test_pred = model(X_test_normalized.float())
        # Convert probabilities to binary predictions
        test_pred_binary = torch.round(torch.sigmoid(test_pred)).float()
        correct = (test_pred_binary.squeeze().long() == y_test.long()
                   ).sum().item()  # Count correct predictions

        # Calculate accuracy
        test_loss = loss_fn(test_pred.squeeze(), y_test.float())
        # Store the numerical test loss value
        test_losses.append(test_loss.item())
        # Calculate accuracy as ratio of correct predictions to total samples
        accuracy = correct / len(y_test)
        if test_loss < min_test_loss:
            min_test_loss = test_loss.item()
            min_test_epoch = epoch

    if epoch % 500 == 0:
        print(
            f"Epoch: {epoch}, Train Loss: {loss.item()}, Test Loss: {test_loss.item()}")
# =============================================================================
#Calculate Metrics
precision = precision_score(y_test, test_pred_binary)
recall = recall_score(y_test, test_pred_binary)
f1 = f1_score(y_test, test_pred_binary)
auc_roc = roc_auc_score(y_test, test_pred.numpy())

print(f"Minimum Test Loss: {min_test_loss} at Epoch: {min_test_epoch}")
print(f"Accuracy on the train set: {accuracy_train:.2%}")
print(f"Accuracy on the test set: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-Score: {f1:.2%}")
print(f"AUC-ROC: {auc_roc:.2%}")

#%% Scatter Plots  ==============================================================
# columns_X = data.columns[:-1].tolist()
# df_X = pd.DataFrame(X.numpy(), columns=columns_X)
# df_X['Outcome'] = y.numpy()

# # Plot pairplots for each pair of features
# sns.pairplot(df_X, diag_kind='kde', markers='o', hue='Outcome', palette='viridis', height=2.5)
# plt.show()
#
# =============================================================================
#%% Plotting the loss curves

plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(test_losses, label='Test Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss Curves')
plt.legend()
plt.grid(visible=True)
plt.show()

#%% Exporting Evaluation Metrics
new_data = {'State': state,
            'Epochs': epochs,
            'Learning Rate': lr,
            'Minimum Test Loss': min_test_loss,
            'Accuracy on the train set': accuracy_train,
            'Accuracy on the test set': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC-ROC': auc_roc}
data_list.append(new_data)

# Save the updated data list to the file
with open(file_path_data, 'w') as f:
    json.dump(data_list, f)

# Create a DataFrame from the list
columns_Y = ['State', 'Epochs', 'Learning Rate', 'Minimum Test Loss', 'Accuracy on the train set',
            'Accuracy on the test set', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
df = pd.DataFrame(data_list, columns=columns_Y)

# Save the DataFrame to a CSV file
file_path_csv = 'Trials.csv'
df.to_csv(file_path_csv, index=False)
