# get customer churn dataset from the sklearn

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

def get_data():
    print("Loading data...")
    data = fetch_openml('credit-g', version=1, as_frame=True)
    X = data.data
    y = data.target.apply(lambda x: 1 if x == 'good' else 0)  # Convert target to binary labels
    print(X)
    X = pd.get_dummies(X, columns=X.select_dtypes(include=['category']).columns)
    X = X.apply(lambda col: col.astype(int) if col.dtype == 'bool' else col)
    
    # split the dataset into training and testing sets and normalize the features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert the data to PyTorch tensors
    
    # error object type series cannot be converted to float, so we need to unsqueeze the target tensor to make it a 2D tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    # Create DataLoader for training and testing sets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader
