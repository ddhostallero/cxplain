import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

# from cxplain.backend.masking.zero_masking import FastZeroMasking
from tensorflow.python.keras.losses import mean_squared_error as loss
from cxplain import MLPModelBuilder, ZeroMasking, CXPlain
from cxplain.backend.model_builders.custom_mlp import CustomMLPModelBuilder


SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


size_hidden1 = 50
size_hidden2 = 50
size_hidden3 = 1

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(50, size_hidden1)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(size_hidden1, size_hidden2)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(size_hidden2, size_hidden3)

    def forward(self, input):
        return self.lin3(self.relu2(self.lin2(self.relu1(self.lin1(input)))))
    
    def predict(self, input):
        x = torch.Tensor(input)
        return self.forward(x).detach().numpy()


criterion = nn.MSELoss(reduction='sum')

def train(model_inp, num_epochs):
    optimizer = torch.optim.Adam(model_inp.parameters(), lr=0.0001)
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for inputs, labels in train_iter:
            # forward pass
            outputs = model_inp(inputs)
            # defining loss
            loss = criterion(outputs, labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # computing gradients
            loss.backward()
            # accumulating running loss
            running_loss += loss.item()
            # updated weights based on computed gradients
            optimizer.step()
        if epoch % 20 == 0:    
            print('Epoch [%d]/[%d] running accumulative loss across all batches: %.3f' %
                  (epoch + 1, num_epochs, running_loss))
        running_loss = 0.0


x = np.random.normal(size=(300, 50))
y = x[:,20] + x[:,40] + np.random.normal(scale = 0.01, size=300)

ss = StandardScaler()
x = ss.fit_transform(x)

ss = StandardScaler()
y = ss.fit_transform(y.reshape(-1, 1)).reshape(-1)


X_train = x[:200]
y_train = y[:200]
X_test = x[200:]
y_test = y[200:]

X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).view(-1, 1).float()

X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).view(-1, 1).float()

datasets = torch.utils.data.TensorDataset(X_train, y_train)
train_iter = torch.utils.data.DataLoader(datasets, batch_size=30, shuffle=True)

model = MyModel()
model.train()
train(model, 300)

model.eval()
outputs = model(X_test).detach().numpy()
err = np.sqrt(mean_squared_error(outputs, y_test.detach().numpy()))
print(err)

outputs = pd.Series(outputs[:,0], index=range(200,300))
# plt.scatter(outputs, y[200:])
# plt.xlabel("Output")
# plt.ylabel("Label")
# plt.show()




def get_masked_data_for_CXPlain(model, x):
    x_train = torch.FloatTensor(x)

    n_feats = x.shape[1]
    patch = 5
    mask = np.ones((n_feats//patch, n_feats))
    for i in range(n_feats//patch):
        mask[i, i*patch:(i+1)*patch] = 0
    
    y_pred = model(x_train).detach().numpy()
    
    mask = torch.FloatTensor(mask)
    
    list_of_masked_outs = []
    for i, sample in enumerate(x_train):
        masked_sample = sample*mask
        list_of_masked_outs.append(model(masked_sample).unsqueeze(0).detach().numpy())
    
    masked_outs = np.concatenate(list_of_masked_outs)
    return(x, y_pred, masked_outs)


model_builder = CustomMLPModelBuilder(num_layers=2, num_units=32, batch_size=32, learning_rate=0.001, n_feature_groups=10)

# masking_operation = ZeroMasking()
# masking_operation = FastZeroMasking()
k = get_masked_data_for_CXPlain(model, x[:200])
explainer = CXPlain(model, model_builder, None, loss)#,downsample_factors=(5,))
explainer.fit(x[:200], y[:200], masked_data=k)
attributions = explainer.explain_groups(x[200:])


attr = pd.DataFrame(attributions, index=range(200, 300))
# plt.plot(range(50), np.abs(attr).mean(axis=0), label='attr')
# #plt.plot(range(50), np.abs(mult).mean(axis=0), label='mult')
# plt.show()

print(attr.shape)
print(attr.loc[200])

df = pd.DataFrame(x[200:], index=range(200,300))
idx = df.loc[np.abs(df[20]) < 0.1].loc[np.abs(df[40]) > 1].index[0]

# plt.plot(range(50), np.abs(attr.loc[idx]), label='attr')
# plt.legend()
# plt.show()


print(np.abs(attr).mean(axis=0)[4], np.abs(attr).mean(axis=0)[8])