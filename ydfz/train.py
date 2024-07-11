import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle


def preprocess_data(df, label_col):
    # Encode the string column to integers
    label_encoder = LabelEncoder()
    df[label_col + '_encoded'] = label_encoder.fit_transform(df[label_col])
    return df, label_encoder


def split_data(df, feature_col, target_col, test_size=0.2, random_state=42):
    X = df[[feature_col]].values.astype(np.float32)
    y = df[target_col].values.astype(np.int64)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def to_torch_tensors(X_train, X_test, y_train, y_test):
    return (torch.tensor(X_train), torch.tensor(y_train), torch.tensor(X_test), torch.tensor(y_test))


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(model, X_train_tensor, y_train_tensor, num_epochs=10, batch_size=64, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(X_train_tensor.size()[0])
        for i in range(0, X_train_tensor.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


def save_model_and_data(model, df, label_encoder, feature_col, file_path='compressed_data.pkl'):
    compressed_data = {
        'Column1': df[feature_col].tolist(),
        'Model': model.state_dict(),
        'LabelEncodings': label_encoder.classes_.tolist()
    }

    with open(file_path, 'wb') as f:
        pickle.dump(compressed_data, f)


def load_model_and_data(file_path='compressed_data.pkl'):
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)

    return loaded_data


def reconstruct_column(df, model, label_encodings, feature_col):
    X = np.array(df[feature_col]).reshape(-1, 1).astype(np.float32)
    X_tensor = torch.tensor(X)

    with torch.no_grad():
        outputs = model(X_tensor)
        reconstructed_col_encoded = torch.argmax(outputs, dim=1).numpy()
        reconstructed_col_encoded = reconstructed_col_encoded.astype(int)  # Ensure the array is of type int
        reconstructed_col = label_encodings[reconstructed_col_encoded]

    return reconstructed_col


# Example usage
data = {
    'Column1': np.arange(1, 1000001),
    'Column2': ['Category' + str(i % 10) for i in range(1, 1000001)]  # Example categorical data
}
df = pd.DataFrame(data)

label_col = 'Column2'
feature_col = 'Column1'

df, label_encoder = preprocess_data(df, label_col)
X_train, X_test, y_train, y_test = split_data(df, feature_col, label_col + '_encoded')
X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = to_torch_tensors(X_train, X_test, y_train, y_test)

input_size = 1
hidden_size1 = 64
hidden_size2 = 32
num_classes = len(np.unique(df[label_col + '_encoded']))
model = SimpleNN(input_size, hidden_size1, hidden_size2, num_classes)

train_model(model, X_train_tensor, y_train_tensor)

save_model_and_data(model, df, label_encoder, feature_col)

loaded_data = load_model_and_data()
loaded_model = SimpleNN(input_size, hidden_size1, hidden_size2, num_classes)
loaded_model.load_state_dict(loaded_data['Model'])
loaded_model.eval()

reconstructed_column = reconstruct_column(df, loaded_model, loaded_data['LabelEncodings'], feature_col)

print("Reconstructed Column2:\n", reconstructed_column)
