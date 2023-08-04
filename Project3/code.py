import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import statsmodels.api as sm
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from skorch import NeuralNetRegressor
import geopandas as gpd
import plotly.express as px


df_s = pd.read_csv('./raw_state_data_drunk_driving.csv')
#df_c['Date of Review']=pd.to_datetime(df_c['Date of Review'])
df_ct = pd.read_csv('./raw-census-tracts-dataset_clean.csv')
df_ct = df_ct.iloc[:,2:]

### 1.linear model-census tracts
## 1.1 linear model, histograms comparison
X = df_ct.iloc[:,0:22]
y = df_ct['DRUNK_DRIVING_PERCENTAGE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

lm_model = LinearRegression()
lm_model.fit(X_train, y_train)

y_lm_pred = lm_model.predict(X_test)

lm_mse = mean_squared_error(y_test, y_lm_pred)
print('Mean squared error:', lm_mse) #53.38665462319651


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.histplot(y_test, bins=20, ax=ax[0])
sns.histplot(y_lm_pred, bins=20, ax=ax[1])
ax[0].set_title('Actual Drunk Driving Percentage')
ax[1].set_title('Predicted Drunk Driving Percentage (Linear Model)')
plt.show()



# sns.histplot(y_test, alpha=1, color='green', bins=20, label='Actual')
# sns.histplot(y_pred_std, alpha=1, color='blue', bins=20, label='Predicted')
# plt.legend(loc='upper right')
# plt.show()

## 1.2 error hist
plt.hist(y_test-y_lm_pred, edgecolor='black', alpha=0.5, bins=20)
plt.xlabel('Error (Linear Model)')
plt.ylabel('Count')
plt.show()

sns.histplot(y_test-y_lm_pred, alpha=0.5)
plt.show()

## 1.3 regularization
# train L1 regularized model (Lasso)
lasso_cv = LassoCV(cv=5, random_state=1).fit(X_train, y_train)
print("Best alpha:", lasso_cv.alpha_)
a_l=lasso_cv.alpha_ # too large 50+

l1_model = Lasso(alpha=0.1)
l1_model.fit(X_train, y_train)

# train L2 regularized model (Ridge)
ridge_cv = RidgeCV(cv=5, alphas=(0.1, 1.0, 10.0)).fit(X_train, y_train)
print("Best alpha:", ridge_cv.alpha_)
a_r=ridge_cv.alpha_ #very weird, it's too large, 70+

l2_model = Ridge(alpha=0.1)
l2_model.fit(X_train, y_train)

y_l1_pred = l1_model.predict(X_test)
y_l2_pred = l2_model.predict(X_test)
l1_mse = mean_squared_error(y_test, y_l1_pred)
l2_mse = mean_squared_error(y_test, y_l2_pred)
print(f'L1 model (Lasso) MSE: {l1_mse:.2f}')
print(f'L2 model (Ridge) MSE: {l2_mse:.2f}')

### 2. Random Forest
## 2.1 Raw random forest
# Create a random forest regressor object
rf_model = RandomForestRegressor(n_estimators=200, random_state=1)

# Fit the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_rf_pred = rf_model.predict(X_test)

# Calculate the mean squared error
rf_mse = mean_squared_error(y_test, y_rf_pred)

# Print the mean squared error
print("Random Forest MSE:", rf_mse)

## 2.2 compare actual vs predicted

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.histplot(y_test, bins=20, ax=ax[0])
sns.histplot(y_rf_pred, bins=20, ax=ax[1])
ax[0].set_title('Actual Drunk Driving Percentage')
ax[1].set_title('Predicted Drunk Driving Percentage (Random Forest)')
plt.show()



sns.kdeplot(y_test, label='Actual')
sns.kdeplot(y_rf_pred, label='Predicted(Random Forest)')
plt.legend(loc='upper right')
plt.show()

## 2.3 Random forest: Tune hyper-parameters

# Define the parameter grid to search over
param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

# Create the random forest regressor model
# rf_model = RandomForestRegressor(random_state=1, warm_start=True)

# # Create a grid search object to find the best hyperparameters
# grid_search = GridSearchCV(
#     estimator=rf_model, 
#     param_grid=param_grid, 
#     cv=5, 
#     scoring='neg_mean_squared_error', 
#     n_jobs=-1
# )

# # Fit the grid search object to the data
# grid_search.fit(X_train, y_train)

# # Get the best hyperparameters and their corresponding score
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_

# # Create a new random forest model with the best hyperparameters
# best_rf_model = RandomForestRegressor(
#     n_estimators=best_params['n_estimators'],
#     max_depth=best_params['max_depth'],
#     min_samples_split=best_params['min_samples_split'],
#     min_samples_leaf=best_params['min_samples_leaf'],
#     max_features=best_params['max_features'],
#     random_state=1,
#     warm_start=True
# )

# # Fit the new model to the training data
# best_rf_model.fit(X_train, y_train)

# # Predict on the test data
# y_pred = best_rf_model.predict(X_test)

# # Calculate the mean squared error
# mse = mean_squared_error(y_test, y_pred)

### 3. Neural networks
X = df_ct.iloc[:,0:22]
y = df_ct['DRUNK_DRIVING_PERCENTAGE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

## 3.8 Train a model
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# set hyper parameters for networks
input_size = X.shape[1]
output_size = 1
hidden_size = int(2/3 * input_size + output_size)
batch_size = 200  #20000
num_epochs = 100 #set independent of batch_size not euqal number of batches gotten in load step
learning_rate = 0.001
# data preparation
# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# convert to "tensors"--datatype for neural networks
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

# load data
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor) #combine X and y to tensorDataset
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# connect three-layer neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model

total_step = len(train_loader)
for epoch in range(num_epochs):#50
    for i, (inputs, labels) in enumerate(train_loader):
        # go over all the batches (divided according batch_size)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

# Test the model
model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    total_loss = 0
    y_pred = []
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)  # Multiply loss by batch size to account for different batch sizes
        y_pred.extend(outputs.cpu().numpy())
    mean_loss = total_loss / len(test_loader.dataset)  # Calculate mean loss across all test samples
    nn_mse = mean_loss
    print(f'Test Mean Squared Error: {nn_mse:.4f}')

## 3.9 viz distribution: predict vs true
y_nn_pred = pd.DataFrame(np.concatenate(y_pred), columns=['DRUNK_DRIVING_PERCENTAGE'])['DRUNK_DRIVING_PERCENTAGE']
y_nn_pred.index = y_test.index
sns.kdeplot(y_test, label='Actual')
sns.kdeplot(y_nn_pred, label='Predicted(Neural Networks)')
plt.legend(loc='upper right')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.histplot(y_test, bins=20, ax=ax[0])
sns.histplot(y_nn_pred, bins=20, ax=ax[1])
ax[0].set_title('Actual Drunk Driving Percentage')
ax[1].set_title('Predicted Drunk Driving Percentage (Neural Networks)')
plt.show()


## 3.10 Tune hyper-parameters
# ##############################
# # data preparations:
# X = df_ct.iloc[:,0:22]
# y = df_ct['DRUNK_DRIVING_PERCENTAGE']
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
# y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
# # set hyper parameters for networks
# input_size = X.shape[1]
# output_size = 1
# hidden_size = int(2/3 * input_size + output_size)
# batch_size = 200  #20000
# num_epochs = 100 #set independent of batch_size not euqal number of batches gotten in load step
# learning_rate = 0.001


# # 3.10.1 number of layers
# # define module(number of layers, activation function, dropout rate, hidden layers' neuron number)
# class NeuralNet(nn.Module):
#     def __init__(self, n_layers=2):
#         super().__init__()
#         self.layers = []
#         self.acts = []
#         for i in range(n_layers):
#             if i==0:
#                 self.layers.append(nn.Linear(input_size, hidden_size))
#             else:
#                 self.layers.append(nn.Linear(hidden_size, hidden_size))
#             self.acts.append(nn.Sigmoid())
#             self.add_module(f"layer{i}", self.layers[-1])
#             self.add_module(f"act{i}", self.acts[-1])
#         self.output = nn.Linear(hidden_size, 1)
 
#     def forward(self, x):
#         for layer, act in zip(self.layers, self.acts):
#             x = act(layer(x))
#         x = self.output(x)
#         return x

# # model
# criterion = nn.MSELoss()
# model = NeuralNetRegressor(
#     module=NeuralNet,
#     max_epochs=100,
#     batch_size=batch_size,#200
#     #module__n_layers=2
#     criterion=criterion,
#     optimizer=optim.Adam,
#     optimizer__lr=learning_rate,
#     verbose=False
# )

# # define the grid search parameters
# param_grid = {
#     'module__n_layers': [1,2,3,4]
# }
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5, scoring='neg_mean_squared_error')
# grid_result = grid.fit(X_scaled, y)

# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
    

# # 3.10.2 activation function
# class NeuralNet(nn.Module):
#     def __init__(self, activation=nn.Sigmoid):
#         super().__init__()
#         self.layer1 = nn.Linear(input_size, hidden_size)
#         self.layer2 = nn.Linear(hidden_size, hidden_size)
#         self.act = activation()
#         self.output = nn.Linear(hidden_size, 1)
 
#     def forward(self, x):
#         x = self.act(self.layer1(x))
#         x = self.act(self.layer2(x))
#         x = self.output(x)
#         return x

# # model
# model = NeuralNetRegressor(
#     module=NeuralNet,
#     max_epochs=100,
#     batch_size=batch_size,#200
#     criterion=nn.MSELoss,
#     optimizer=optim.Adam,
#     optimizer__lr=learning_rate,
#     verbose=False
# )
# # define the grid search parameters
# param_grid = {
#     'module__activation': [nn.Identity, nn.ReLU, nn.ELU, nn.ReLU6, nn.GELU, nn.Softplus, nn.Softsign, nn.Tanh, nn.Sigmoid, nn.Hardsigmoid]
# }
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5, scoring='neg_mean_squared_error')
# grid_result = grid.fit(X_scaled, y)

# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
    
# # 3.10.3 optimizer

# class NeuralNet(nn.Module):
#     def __init__(self, activation=nn.ELU):
#         super().__init__()
#         self.layer1 = nn.Linear(input_size, hidden_size)
#         self.layer2 = nn.Linear(hidden_size, hidden_size)
#         self.act = activation()
#         self.output = nn.Linear(hidden_size, 1)
 
#     def forward(self, x):
#         x = self.act(self.layer1(x))
#         x = self.act(self.layer2(x))
#         x = self.output(x)
#         return x
    
# # model
# model = NeuralNetRegressor(
#     module=NeuralNet,
#     max_epochs=100,
#     batch_size=batch_size,#200
#     criterion=nn.MSELoss,
#     #optimizer=optim.Adam,
#     optimizer__lr=learning_rate,
#     verbose=False
# )
# # define the grid search parameters
# param_grid = {
#     'optimizer': [optim.SGD, optim.RMSprop, optim.Adagrad, optim.Adadelta, optim.Adam, optim.Adamax, optim.NAdam],
# }
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5, scoring='neg_mean_squared_error')
# grid_result = grid.fit(X_scaled, y)

# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
# ######################

# best model: 4-layer, activation ELU, optimizer SGD
X = df_ct.iloc[:,0:22]
y = df_ct['DRUNK_DRIVING_PERCENTAGE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# set hyper parameters for networks
input_size = X.shape[1]
output_size = 1
hidden_size = int(2/3 * input_size + output_size)
batch_size = 200  #20000
num_epochs = 100 #set independent of batch_size not euqal number of batches gotten in load step
learning_rate = 0.001
# data preparation
# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# convert to "tensors"--datatype for neural networks
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

# load data
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor) #combine X and y to tensorDataset
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# connect 4-layer neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.elu = nn.ELU()
        self.fc3 = nn.Linear(hidden_size, output_size)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model

total_step = len(train_loader)
for epoch in range(num_epochs):#50
    for i, (inputs, labels) in enumerate(train_loader):
        # go over all the batches (divided according batch_size)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

# Test the model
model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    total_loss = 0
    y_pred = []
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)  # Multiply loss by batch size to account for different batch sizes
        y_pred.extend(outputs.cpu().numpy())
    mean_loss = total_loss / len(test_loader.dataset)  # Calculate mean loss across all test samples
    nn_mse = mean_loss
    print(f'Test Mean Squared Error: {nn_mse:.4f}')

#Test Mean Squared Error: 45.9004
    
### 4. Transfer to a smaller dataset(df_s)
# two datasets
X_ct = df_ct.iloc[:,0:22]
y_ct = df_ct['DRUNK_DRIVING_PERCENTAGE']
X_s = df_s.iloc[:,2:24]
y_s = df_s['DRUNK_DRIVING_PERCENTAGE_2']
## 4.11
df = pd.DataFrame(0.0, columns=['Census Tracts', 'State'], index=['Linear Model', 'Random Forest', 'Neural Network'])


lm_model = LinearRegression()
X = X_ct
y = y_ct
cv_scores_ct = np.mean(-cross_val_score(lm_model, X, y, cv=5, scoring='neg_mean_squared_error'))
df['Census Tracts']['Linear Model'] = cv_scores_ct
X = X_s
y = y_s
cv_scores_s = np.mean(-cross_val_score(lm_model, X, y, cv=5, scoring='neg_mean_squared_error'))
df['State']['Linear Model'] = cv_scores_s

# Raw random forest
rf_model = RandomForestRegressor(n_estimators=200, random_state=1)
X = X_ct
y = y_ct
cv_scores_ct = np.mean(-cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error'))
df['Census Tracts']['Random Forest'] = cv_scores_ct
X = X_s
y = y_s
cv_scores_s = np.mean(-cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error'))
df['State']['Random Forest'] = cv_scores_s


# Neural network
# use the same best model 4-layer, ELU activation function, SGD optimizer


X = X_s
y = y_s
l = []
for i in range(1,6):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set hyper parameters for networks
    input_size = X.shape[1]
    output_size = 1
    hidden_size = int(2/3 * input_size + output_size)
    batch_size = 200  #20000
    num_epochs = 100 #set independent of batch_size not euqal number of batches gotten in load step
    learning_rate = 0.001
    # data preparation
    # scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # convert to "tensors"--datatype for neural networks
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

    # load data
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor) #combine X and y to tensorDataset
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # connect 4-layer neural network
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size) 
            self.fc2 = nn.Linear(hidden_size, hidden_size) 
            self.elu = nn.ELU()
            self.fc3 = nn.Linear(hidden_size, output_size)  
        
        def forward(self, x):
            out = self.fc1(x)
            out = self.elu(out)
            out = self.fc2(out)
            out = self.elu(out)
            out = self.fc3(out)
            return out

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model

    total_step = len(train_loader)
    for epoch in range(num_epochs):#50
        for i, (inputs, labels) in enumerate(train_loader):
            # go over all the batches (divided according batch_size)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

    # Test the model
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        total_loss = 0
        y_pred = []
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)  # Multiply loss by batch size to account for different batch sizes
            y_pred.extend(outputs.cpu().numpy())
        mean_loss = total_loss / len(test_loader.dataset)  # Calculate mean loss across all test samples
        nn_mse = mean_loss
        print(f'Test Mean Squared Error: {nn_mse:.4f}')
    l.append(mean_loss)

df['State']['Neural Network'] = np.mean(l)

## 4.12 Transfer Learning: Linear Model
lm_model = LinearRegression()
lm_model.fit(X_ct, y_ct)

y_lm_pred_trans = lm_model.predict(X_s)

lm_mse_trans = mean_squared_error(y_s, y_lm_pred_trans)
print('Mean squared error:', lm_mse_trans) 

## 4.13 Transfer Learning: neural network
X = X_ct
y = y_ct
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# set hyper parameters for networks
input_size = X.shape[1]
output_size = 1
hidden_size = int(2/3 * input_size + output_size)
batch_size = 200  #20000
num_epochs = 50 #set independent of batch_size not euqal number of batches gotten in load step
learning_rate = 0.001
# data preparation
# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# convert to "tensors"--datatype for neural networks
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

# load data
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor) #combine X and y to tensorDataset
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# connect 4-layer neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.elu = nn.ELU()
        self.fc3 = nn.Linear(hidden_size, output_size)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out

model = NeuralNet(input_size, hidden_size, output_size).to(device)
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):#50
    for i, (inputs, labels) in enumerate(train_loader):
        # go over all the batches (divided according batch_size)
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

model0 = model
model_params = model0.state_dict()
# fine tune with df_s
X = X_s
y = y_s
l_tr = []

for i in range(1,6):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set hyperparameters for the network
    input_size = X.shape[1]
    output_size = 1
    hidden_size = int(2/3 * input_size + output_size)
    batch_size = 50
    num_epochs = 10
    learning_rate = 0.001
    # Data preparation
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1).to(device)

    # Load data
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    model = model0
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')
    # Test Model
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        total_loss = 0
        y_pred = []
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)  # Multiply loss by batch size to account for different batch sizes
            y_pred.extend(outputs.cpu().numpy())
        mean_loss = total_loss / len(test_loader.dataset)  # Calculate mean loss across all test samples
        nn_mse = mean_loss
        print(f'Test Mean Squared Error: {nn_mse:.4f}')
    l_tr.append(nn_mse)
    
np.mean(l_tr)


### 5. Visualization (map)
## 5.14 drunk percentage plot
# Read shapefile
data = pd.read_csv('2011_us_ag_exports.csv')
merged = df_s.merge(data.iloc[:,0:2], left_on="STATE_NAME", right_on="state")

fig = px.choropleth(merged,
                    locations='code', 
                    locationmode="USA-states", 
                    scope="usa",
                    color="DRUNK_DRIVING_PERCENTAGE_2",
                    color_continuous_scale="Viridis_r", 
                    )
# Adjust the layout to emphasize the map
fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),  # Remove margins around the plot
    coloraxis_colorbar=dict(
        len=0.3,  # Set the color bar length (proportion of the plot width)
        thickness=15,  # Set the color bar thickness (in pixels)
        title='Percentage',  # Set the color bar title
    ),
)
fig.show()

## 4.15 Error plot
X_s_scaled = scaler.transform(X_s)
X_s_tensor = torch.tensor(X_s_scaled, dtype=torch.float32)

# Set the model to evaluation mode
model.eval()

# Make predictions
with torch.no_grad():
    outputs = model(X_s_tensor)

# Convert the tensor output to numpy array
predictions = outputs.numpy()

merged['pred_nn'] = predictions
merged['error'] = abs(merged['DRUNK_DRIVING_PERCENTAGE_2'] - merged['pred_nn'])

fig = px.choropleth(merged,
                    locations='code', 
                    locationmode="USA-states", 
                    scope="usa",
                    color="error",
                    )
# Adjust the layout to emphasize the map
fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),  # Remove margins around the plot
    coloraxis_colorbar=dict(
        len=0.3,  # Set the color bar length (proportion of the plot width)
        thickness=15,  # Set the color bar thickness (in pixels)
        title='Error',  # Set the color bar title
    ),
)
fig.show()