import torch
import torch.nn as nn
import torch.nn.functional as F
from single_task.NoDA_base.relu_models import FeatureExtractor, LabelRegressor

class EnsModel(nn.Module):
    def __init__(self, model_list):
        super(EnsModel, self).__init__()
        self.model_list = model_list
        
    def forward(self, x):
        outs = []
        for model in self.model_list:
            out = model(x)
            outs.append(out)
        return torch.cat(outs, axis=1).mean(axis=1, keepdim=True)

    def predict(self, x):
        print(x.shape)

        if len(x) <= 512:
            x = torch.Tensor(x)
            return self.forward(x).detach().numpy()
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.to(device)
            data = torch.utils.data.TensorDataset(x)
            data = torch.utils.data.DataLoader(data, batch_size=512, shuffle=False)
            ret_val = []
            for [x] in data:
                x = x.to(device)
                ret_val.append(self.forward(x))


            ret_val = torch.cat(ret_val, axis=0).detach().numpy()
            print('***', ret_val.shape)
            return ret_val

class myModel(nn.Module):
    def __init__(self, encoder, predictor):
        super(myModel, self).__init__()
        self.enc = encoder
        self.pred = predictor
    
    def forward(self, x):
        return self.pred(self.enc(x))
    
    
def load_model(seed, drug, n_genes):    
    source_fe_weights = '../p2c-drp/results/NoDA_base/01_pear_base_relu/pear_base_relu_seed%d/%s/weights/feature_extractor'%(seed, drug)
    source_lr_weights = '../p2c-drp/results/NoDA_base/01_pear_base_relu/pear_base_relu_seed%d/%s/weights/label_regressor'%(seed, drug)

    feature_extractor = FeatureExtractor(n_genes, 256)#.cuda()
    label_regressor = LabelRegressor(256)#.cuda()
    
    feature_extractor.load_state_dict(torch.load(source_fe_weights).state_dict(), strict=False)
    label_regressor.load_state_dict(torch.load(source_lr_weights).state_dict(), strict=False)

    model = myModel(feature_extractor, label_regressor)
    return model