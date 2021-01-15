import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
  def __init__(self, n_in, n_out):
    super(FeatureExtractor, self).__init__()
    
    self.enc1 = nn.Linear(n_in, 512)    
    self.enc2 = nn.Linear(512, n_out)

  def forward(self, x):
    x = F.relu(self.enc1(x))
    return F.relu(self.enc2(x))

class LabelRegressor(nn.Module):
  def __init__(self, n_in):
    super(LabelRegressor, self).__init__()
    
    self.dr1 = nn.Linear(n_in, 128)    
    self.drp = nn.Dropout(0.5)
    # self.batch_norm = nn.BatchNorm1d(128)
    self.dr_out = nn.Linear(128, 1)

  def forward(self, x):
    x = F.relu(self.dr1(x))
    x = self.drp(x)
    # x = self.batch_norm(x)
    return self.dr_out(x)