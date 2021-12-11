import torch
import torch.nn as nn 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

train_data = df['Age'........] # ADD FEATURES AS REQUIRED
print(train_data)

X_train = train_data.values
y_train = df['GAD_T'].values

# Data preprocessing step 2: standardize the data as the values are very large and varied
sc = MinMaxScaler()
sct = MinMaxScaler()
X_train=sc.fit_transform(X_train.reshape(-1,1))
y_train =sct.fit_transform(y_train.reshape(-1,1))


# # Data preprocessing Step 3: Convert the numpy arrays to tensors
X_train = torch.from_numpy(X_train.astype(np.float32)).view(-1,1)
y_train = torch.from_numpy(y_train.astype(np.float32)).view(-1,1)

# Regression Model Building in PyTorch
input_size = 1
output_size = 1

# Define layer
model = nn.Linear(input_size , output_size)

# Define loss and optimizer
learning_rate = 0.0001
l = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr =learning_rate )


# Training
num_epochs = 5000
for epoch in range(num_epochs):
    #forward feed
    y_pred = model(X_train.requires_grad_())

    #calculate the loss
    loss= l(y_pred, y_train)

    #backward propagation: calculate gradients
    loss.backward()

    #update the weights
    optimizer.step()

    #clear out the gradients from the last step loss.backward()
    optimizer.zero_grad()
    
    print('epoch {}, loss {}'.format(epoch, loss.item()))
    
    
# Evaluation
predicted = model(X_train).detach().numpy()

# Plot
plt.scatter(X_train.detach().numpy()[:100] , y_train.detach().numpy()[:100])
plt.plot(X_train.detach().numpy()[:100] , predicted[:100] , "red")
plt.xlabel("Age")
plt.ylabel("GAD_T")
plt.show()
