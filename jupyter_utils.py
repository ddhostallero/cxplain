import torch
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import pickle

class AllDataset():
    def __init__(self, data_dir, GDSC_GENE_EXPRESSION, TCGA_GENE_EXPRESSION, GDSC_lnIC50, TCGA_DR, TCGA_TISSUE):

        self.gdsc_expr = pd.read_csv(data_dir + GDSC_GENE_EXPRESSION, index_col=0).T
        self.gdsc_dr = pd.read_csv(data_dir + GDSC_lnIC50, index_col=0).T

        self.tcga_expr = pd.read_csv(data_dir + TCGA_GENE_EXPRESSION, index_col=0).T
        self.tcga_dr = pd.read_csv(data_dir + TCGA_DR, index_col=0).T
        self.tcga_tissue = pd.read_csv(data_dir + TCGA_TISSUE, index_col=0).T

        self.genes = list(self.gdsc_expr.columns)
        self.tcga_expr = self.tcga_expr[self.genes]

        ens2hgnc = pd.read_csv('../drp-data/lists/hgnc2ensembl.txt')
        hgnc = []
        for gene in self.genes:
            x = ens2hgnc.loc[ens2hgnc['Gene stable ID'] == gene]
            hgnc.append(x.iloc[0]['HGNC symbol'])

        self.hgnc = hgnc 


    def filter_and_normalize_data(self, drug, filter_tissue=True, 
        norm_IC50=True, load_normalizer=False, save_normalizer=False):
        labeled_index = self.tcga_dr[drug].dropna().index

        tissue_list = None
        tcga_ul_expr = self.tcga_expr.loc[~self.tcga_expr.index.isin(labeled_index)]
        tcga_lab_expr = self.tcga_expr.loc[self.tcga_expr.index.isin(labeled_index)]


        if filter_tissue:
            tissue_list = []
            tissue = self.tcga_tissue.loc[labeled_index]
            for t in tissue.columns:
                if tissue[t].sum() > 1:
                    tissue_list.append(t)

            tcga_ul_tissue = self.tcga_tissue.loc[self.tcga_tissue.index.isin(tcga_ul_expr.index)][tissue_list]
            tcga_ul_tissue = tcga_ul_tissue.loc[tcga_ul_tissue.sum(axis=1) > 0]
            tcga_ul_expr = tcga_ul_expr.loc[tcga_ul_expr.index.isin(tcga_ul_tissue.index)]

        if norm_IC50:
            d = self.gdsc_dr[drug].dropna().values
            gdsc_dr = self.gdsc_dr[drug].copy()
            gdsc_dr = (gdsc_dr - d.mean())/(d.std())

        idx = list(gdsc_dr.dropna().index.intersection(self.gdsc_expr.index))
        gdsc_expr = self.gdsc_expr.loc[idx]
        gdsc_dr = gdsc_dr[idx]

        ss = StandardScaler(with_std=True)
        gdsc_expr = pd.DataFrame(ss.fit_transform(gdsc_expr), index=gdsc_expr.index, columns=gdsc_expr.columns)
        
        if load_normalizer:
            norm_ad = '../drp-data/preprocessed/normalizers/%s.p'%drug
            ss = pickle.load(open(norm_ad, 'rb'))
            tcga_ul_expr = pd.DataFrame(ss.transform(tcga_ul_expr), index=tcga_ul_expr.index, columns=tcga_ul_expr.columns)
            tcga_lab_expr = pd.DataFrame(ss.transform(tcga_lab_expr), index=tcga_lab_expr.index, columns=tcga_lab_expr.columns)
        else:
            ss = StandardScaler(with_std=True)
            tcga_ul_expr = pd.DataFrame(ss.fit_transform(tcga_ul_expr), index=tcga_ul_expr.index, columns=tcga_ul_expr.columns)
            tcga_lab_expr = pd.DataFrame(ss.transform(tcga_lab_expr), index=tcga_lab_expr.index, columns=tcga_lab_expr.columns)

        if save_normalizer:
            pickle.dump(ss, open('../drp-data/preprocessed/normalizers/%s.p'%drug, 'wb'))

        gdsc_expr.columns = self.hgnc
        tcga_ul_expr.columns = self.hgnc
        tcga_lab_expr.columns = self.hgnc

        return gdsc_expr, gdsc_dr, tcga_ul_expr, tcga_lab_expr