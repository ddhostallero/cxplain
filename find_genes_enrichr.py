import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from gene_finding.models import load_model, EnsModel
from gene_finding.borda import rank_aggregate_Borda
from gene_finding.pathway_matrix import load_pathway, get_valid_enrichr_pathways
import os

SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)

from jupyter_utils import AllDataset

data_dir = '../drp-data/'
GDSC_GENE_EXPRESSION = 'preprocessed/gdsc_tcga/gdsc_rma_gene_expr.csv'
# TCGA_GENE_EXPRESSION = 'preprocessed/gdsc_tcga/tcga_log2_gene_expr.csv'
TCGA_GENE_EXPRESSION = 'preprocessed/gdsc_tcga/tcga_labeled_log2_gene_expr.csv'
enrichr_pathway = 'pathways/9606.enrichr_pathway.edge'
enrichr_pathway_nodemap = 'pathways/9606.enrichr_pathway.node_map'

TCGA_TISSUE = 'preprocessed/tissue_type/TCGA_tissue_one_hot.csv'
GDSC_TISSUE = 'preprocessed/tissue_type/GDSC_tissue_one_hot.csv'

GDSC_lnIC50 = 'preprocessed/drug_response/gdsc_lnic50.csv'
TCGA_DR = 'preprocessed/drug_response/tcga_drug_response.csv'

drugs = [
    # 'bleomycin',
    # 'cisplatin',
    # 'cyclophosphamide',
    # 'docetaxel',
    # 'doxorubicin',
    # 'etoposide',
    'gemcitabine',
    'irinotecan',]
    # 'oxaliplatin',
    # 'paclitaxel',
    # 'pemetrexed',
    # 'tamoxifen',
    # 'temozolomide',
    # 'vinorelbine']
    
dataset = AllDataset(data_dir, GDSC_GENE_EXPRESSION, TCGA_GENE_EXPRESSION, 
                     GDSC_lnIC50, TCGA_DR, TCGA_TISSUE)

list_of_pathways = get_valid_enrichr_pathways(data_dir + enrichr_pathway_nodemap)
pathway_matrix, pathway_names = load_pathway(data_dir + enrichr_pathway, dataset.genes, list_of_pathways, 10, sort=True)

res_dir = 'gene_finding/results/'
explainer = 'CX'

# res_dir = res_dir + explainer + '/'
res_dir = res_dir + 'CX_ens_enrichr3_base/'
# res_dir = res_dir + 'CX_ens_enrichr3/'


def curve(save_dir, attr):

    plt.savefig(save_dir + '/classes.png')


def attribute(att_func, x, baselines, x_idx):

  return pd.DataFrame(att_func.attribute(x, baselines=baselines).detach().numpy(), 
      index=x_idx, columns=dataset.hgnc)


def get_ranked_list(attr, k=200):
    attr = np.abs(attr.T)
    sorted_list = []

    for sample in attr.columns:
        rank = attr.sort_values(sample, ascending=False).index # highest to lowest
        sorted_list.append(rank)

    agg_genes = rank_aggregate_Borda(sorted_list, k, 'geometric_mean')

    return agg_genes


def get_masked_data_for_CXPlain(model, gdsc_expr, pathway_matrix, subtract_mean=False):
    x_train = torch.FloatTensor(gdsc_expr.values)
    print(pathway_matrix.shape)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    if type(model) == EnsModel:
        for m in model.model_list:
            m.to(device)

    model.to(device)
    model.eval()

    y_pred = model(x_train.to(device)).cpu().detach().numpy()
    n_genes = x_train.shape[1]
    n_pathways = len(pathway_matrix)
    mask = torch.ones((n_pathways, n_genes)) - torch.Tensor(pathway_matrix)

    if subtract_mean:
        print(mask.shape)
        n_random_mask = 500
        # random_mask_dict = {}
        ns_with_mask = []
        random_masks_list = []
        mask_index = []
        n_mask_sets = -1 

        print('creating random bl...')

        for i in range(n_pathways):
            m = pathway_matrix[i]
            n_zero = m.sum()
            if n_zero in ns_with_mask:
                mask_index.append(n_mask_sets)
            else:
                random_masks = np.zeros((n_random_mask, n_genes))
                for s in range(n_random_mask):
                    random_masks[s] = np.random.permutation(m)
                random_masks = torch.ones(random_masks.shape) - torch.Tensor(random_masks)

                ns_with_mask.append(n_zero)
                random_masks_list.append(random_masks)

                n_mask_sets += 1
                mask_index.append(n_mask_sets)

        random_masks = torch.cat(random_masks_list, axis=0)
        n_mask_sets += 1 # because we started from -1
        print('random_mask_shape', random_masks.shape)
        print(n_mask_sets)

        print(mask_index)

        mask_index_matrix = torch.zeros((n_mask_sets, n_pathways))
        for i in range(n_pathways):
            mask_index_matrix[mask_index[i], i] = 1
        mask_index_matrix= mask_index_matrix.to(device)

        print(mask_index_matrix)

    list_of_masked_outs = []
    list_of_baselines = []
    for i, sample in enumerate(x_train):
        if (i+1) % 100 == 0: print(i+1)
        masked_sample = sample*mask
        # print(sample.shape, mask.shape, masked_sample.shape)
        data = torch.utils.data.TensorDataset(masked_sample)
        data = torch.utils.data.DataLoader(data, batch_size=1024, shuffle=False)
        
        ret_val = []
        with torch.no_grad():
            for [x] in data:
                x = x.to(device)
                ret_val.append(model(x))

        ret_val = torch.cat(ret_val, axis=0).unsqueeze(0)#.cpu().detach().numpy()
        
        if subtract_mean:
            masked_sample = sample*random_masks
            # print(masked_sample.shape)
            data = torch.utils.data.TensorDataset(masked_sample)
            data = torch.utils.data.DataLoader(data, batch_size=1024, shuffle=False)

            baselines = []
            with torch.no_grad():
                for [x] in data:
                    x = x.to(device)
                    baselines.append(model(x))
            baselines = torch.cat(baselines, axis=0)#.unsqueeze(0).cpu().detach().numpy()
            baselines = baselines.view(-1, n_random_mask, n_mask_sets)
            baselines = baselines.mean(axis=1)
            baselines = torch.matmul(baselines, mask_index_matrix).unsqueeze(2)#.cpu().detach().numpy()


            ret_val = ret_val - baselines
            # print(ret_val.shape)
            ret_val = ret_val.cpu().numpy()#.cpu().detach().numpy()
            list_of_masked_outs.append(ret_val)
            list_of_baselines.append(baselines.cpu().detach().numpy())
        else:
            ret_val = ret_val.cpu().numpy()#.cpu().detach().numpy()
            list_of_masked_outs.append(ret_val)

    masked_outs = np.concatenate(list_of_masked_outs)
    return (gdsc_expr.values, y_pred, masked_outs), list_of_baselines



def find_genes_CX(drug, model, gdsc_expr, gdsc_dr, test_tcga_expr):
    print('obtaining masked data...')
    # masked_data, list_of_baselines = get_masked_data_for_CXPlain(model, gdsc_expr, pathway_matrix)
    masked_data, list_of_baselines = get_masked_data_for_CXPlain(model, gdsc_expr, pathway_matrix, subtract_mean=True)
    lb = np.concatenate(list_of_baselines).reshape(len(gdsc_expr), -1)
    lb = pd.DataFrame(lb, index=gdsc_expr.index, columns=pathway_names)
    lb.to_csv(res_dir + drug + '/baselines.csv')
    # print(lb.shape)
    # exit()
    print('obtained masked data...')

    import tensorflow as tf
    tf.compat.v1.disable_v2_behavior()
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    from tensorflow.python.keras.losses import mean_squared_error as loss
    from cxplain import CXPlain
    from cxplain.backend.model_builders.custom_mlp import CustomMLPModelBuilder
    # from cxplain.backend.masking.zero_masking import FastZeroMasking
    n_pathways = len(pathway_names)
    model_builder = CustomMLPModelBuilder(num_layers=2, num_units=512, batch_size=16, learning_rate=0.001, n_feature_groups=n_pathways)
    # masking_operation = FastZeroMasking()

    print(gdsc_expr.values.shape, gdsc_dr.values.shape)

    print("Fitting CXPlain model")
    explainer = CXPlain(model, model_builder, None, loss, num_models=3)
    explainer.fit(gdsc_expr.values, gdsc_dr.values, masked_data=masked_data)
    print("Attributing using CXPlain")

    attr,_ = explainer.explain_groups(test_tcga_expr.values)
    print('attr')

    attr = pd.DataFrame(attr, index=test_tcga_expr.index, columns=pathway_names)
    borda = get_ranked_list(attr, k=n_pathways)

    attr_mean = list(np.abs(attr).mean(axis=0).nlargest(n_pathways).index)
    out = pd.DataFrame(columns=['borda', 'mean'])
    out['borda'] = borda 
    out['mean'] = attr_mean

    out.to_csv(res_dir + drug + '/pathways.csv', index=False)

    if not os.path.exists(res_dir + drug + '/explainer/'):
        os.mkdir(res_dir + drug + '/explainer/')

    explainer.save(res_dir + drug + '/explainer/', custom_model_saver=None)

def find_genes(drug, n_seeds=10):

    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    if not os.path.exists(res_dir + drug):
        os.mkdir(res_dir + drug)

    # exit()

    gdsc_expr, gdsc_dr, _, test_tcga_expr = dataset.filter_and_normalize_data(drug, load_normalizer=True)
    
    models = []

    for i in range(1, n_seeds+1):
        mod = load_model(seed=i,drug=drug,n_genes=len(dataset.genes))
        mod.eval()
        models.append(mod)
    
    # ---
    # x = torch.FloatTensor(test_tcga_expr.values)
    # print(models[0](x))
    ens = EnsModel(models)
    find_genes_CX(drug, ens, gdsc_expr, gdsc_dr, test_tcga_expr)
    


for drug in drugs:
    print('============================')
    print(drug)
    find_genes(drug)