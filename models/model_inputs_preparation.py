from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch

def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

def split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    return X_train, X_test, y_train, y_test

def get_scaler(scaler):
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()

def get_scaled_values(X_train, X_test, y_train, y_test):
    scaler = get_scaler('minmax')
    X_train_arr = scaler.fit_transform(X_train)
    X_test_arr = scaler.transform(X_test)
    
    y_train_arr = scaler.fit_transform(y_train)
    y_test_arr = scaler.transform(y_test)
    
    return X_train_arr, X_test_arr, y_train_arr, y_test_arr, scaler

def get_loaders(X_train_arr, X_test_arr, y_train_arr, y_test_arr, batch_size=1):

    
    train_features = torch.Tensor(X_train_arr)
    train_targets = torch.Tensor(y_train_arr)
    test_features = torch.Tensor(X_test_arr)
    test_targets = torch.Tensor(y_test_arr)
    
    train = TensorDataset(train_features, train_targets)
    test = TensorDataset(test_features, test_targets)
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader