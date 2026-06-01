import numpy as np
import pandas as pd

import torch
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df= pd.read_csv("california_housing.csv")
## without Geographical info
X = df.drop(columns = ["Latitude" , "Longitude",  "MedHouseVal"])
## with Geographical info
#X = df.drop(columns = ["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

class HousingNN(nn.Module):
    def __init__(self,input_dim,hidden1=64,hidden2=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1,hidden2),
            nn.ReLU(),
            nn.Linear(hidden2,1)
        )
    
    def forward(self,x):
        return self.net(x)
        
def train_nn(X_train_np,y_train_np,X_val_np,y_val_np,
             hidden1 = 64, hidden2 = 32,
             lr = 0.001, epochs = 100, batch_size = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train_tensor = torch.tensor(X_train_np,dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np,dtype=torch.float32).view(-1, 1)
    
    X_val_tensor = torch.tensor(X_val_np,dtype=torch.float32).to(device)
    
    train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
    train_loader = DataLoader(train_dataset,batch_size = batch_size, shuffle=True)
    
    model = HousingNN(
        input_dim=X_train_np.shape[1],
        hidden1=hidden1,
        hidden2=hidden2
    ).to(device)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    
    for epoch in range(epochs):
        model.train()
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_X)
            loss = loss_fn(y_pred,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_tensor).cpu().numpy().reshape(-1)
        
    val_mse = mean_squared_error(y_val_np,val_pred)
    
    return model, val_mse

hidden_size_options = [
    (16,8),
    (32,16),
    (64,32),
    (128,64),
]

kf = KFold(n_splits=5,shuffle=True,random_state=42)

cv_results = []
X_train_df = X_train.reset_index(drop = True)
y_train_series = y_train.reset_index(drop = True)

for hidden1, hidden2 in hidden_size_options:
    fold_mses = []
    
    for train_idx, val_idx in kf.split(X_train_df):
        X_fold_train = X_train_df.iloc[train_idx]
        X_fold_val = X_train_df.iloc[val_idx]
        y_fold_train = y_train_series.iloc[train_idx]
        y_fold_val = y_train_series.iloc[val_idx]
        
        scalar = StandardScaler()
        X_fold_train_scaled = scalar.fit_transform(X_fold_train)
        X_fold_val_scaled = scalar.transform(X_fold_val)
        
        _, val_mse = train_nn(
            X_fold_train_scaled,
            y_fold_train.to_numpy(),
            X_fold_val_scaled,
            y_fold_val.to_numpy(),
            hidden1=hidden1,
            hidden2=hidden2,
            lr = 0.001,
            epochs=100,
            batch_size=64
        )
        
        fold_mses.append(val_mse)
        
        
    mean_cv_mse = np.mean(fold_mses)
    
    cv_results.append({
        "hidden1": hidden1,
        "hidden2": hidden2,
        "CV MSE": mean_cv_mse
    })

    print(f"Hidden sizes ({hidden1}, {hidden2}) - CV MSE: {mean_cv_mse:.4f}")
    
cv_results_df = pd.DataFrame(cv_results)
print("\nCross-validation results:")
print(cv_results_df)

best_row = cv_results_df.loc[cv_results_df["CV MSE"].idxmin()]
best_hidden1 = int(best_row["hidden1"])
best_hidden2 = int(best_row["hidden2"])

print("\nBest hidden sizes:", (best_hidden1, best_hidden2))

final_scaler = StandardScaler()
X_train_scaled = final_scaler.fit_transform(X_train)
X_test_scaled = final_scaler.transform(X_test)

final_model, _ = train_nn(
    X_train_scaled,
    y_train.to_numpy(),
    X_test_scaled,
    y_test.to_numpy(),
    hidden1=best_hidden1,
    hidden2=best_hidden2,
    lr=0.001,
    epochs=200,
    batch_size=64
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

final_model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_pred = final_model(X_test_tensor).cpu().numpy().reshape(-1)

print("\nFinal PyTorch NN Test Results:")
print("Test MSE:", mean_squared_error(y_test, y_pred))
print("Test R2:", r2_score(y_test, y_pred))
