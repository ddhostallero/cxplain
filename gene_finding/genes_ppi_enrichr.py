import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pathway_file = '../../drp-data/pathways/9606.enrichr_pathway.edge'
pathway = pd.read_csv(pathway_file, sep='\t', header=None)
print("pathways:", pathway[0].nunique())
print("pathway genes:", pathway[1].nunique())

gsc_filtered = '../../KnowEng_GSC/GSC_10mod/drawr_filtered/DraWR_GSC_Enrichr_STRINGExp.xlsx'

ppi_file = '../../drp-data/pathways/9606.STRING_experimental.edge'
ppi = pd.read_csv(ppi_file, sep='\t', header=None)
print("PPI original edges:", len(ppi))
ppi['norm_score'] = ppi[2]/ppi[2].max()
ppi = ppi.loc[ppi['norm_score'] > 0.5]
print("PPI filtered edges:", len(ppi))
nodes = list(set(ppi[0]).union(set(ppi[1])))
print("PPI nodes:", len(nodes) )

folder = 'CX_ens10'
mean_attribution_file = 'results/CX_ens10/all_attributions.csv'
feature_attr = pd.read_csv(mean_attribution_file, index_col=0)

top_genes_file = 'results/CX_ens10/top_genes_mean_aggregation_info.xlsx'
writer_a = pd.ExcelWriter('results/%s/one_hop.xlsx'%folder, engine='xlsxwriter')

drugs = [
    'bleomycin',
    'cisplatin',
    'cyclophosphamide',
    'docetaxel',
    'doxorubicin',
    'etoposide',
    'gemcitabine',
    'irinotecan',
    'oxaliplatin',
    'paclitaxel',
    'pemetrexed',
    'tamoxifen',
    'temozolomide',
    'vinorelbine']

# use dictionary coz it's faster
conv_file = '../../drp-data/lists/hgnc2ensembl.txt'
f = open(conv_file, 'r')
conv_table = {}

for line in f:
    line = line.strip().split(',')
    if line[1] != "":
        conv_table[line[0]] = line[1]

# print(conv_table)

for drug in drugs:

    gsc_pathways = pd.read_excel(gsc_filtered, sheet_name=drug, index_col='property_gene_set_id')
    pathway_genes = pathway.loc[pathway[0].isin(gsc_pathways.index)][1].unique()

    top_features = pd.read_excel(top_genes_file, sheet_name=drug, index_col='ensembl')

    one_hop_from_top_feats_left = ppi.loc[ppi[0].isin(top_features.index)][1]
    one_hop_from_top_feats_right = ppi.loc[ppi[1].isin(top_features.index)][0]
    one_hop_from_top_feats = set(one_hop_from_top_feats_left).union(set(one_hop_from_top_feats_right))

    one_hop_from_pathway_left = ppi.loc[ppi[0].isin(pathway_genes)][1]
    one_hop_from_pathway_right = ppi.loc[ppi[1].isin(pathway_genes)][0]
    one_hop_from_pathway = set(one_hop_from_pathway_left).union(set(one_hop_from_pathway_right))

    one_hop = one_hop_from_top_feats.union(one_hop_from_pathway)
    nodes_of_interest = set(top_features.index).union(set(pathway_genes)).union(one_hop)

    features = feature_attr[drug].sort_values(ascending=False).index
    ranks = pd.Series(range(1, len(features) + 1), index=features)
    paths = list(gsc_pathways.index)

    cols = ['hgnc', 'is_feature', 'attribution', 'rank', 
            'is_top_feat', 'is_1H_from_pathway', 
            'is_1H_from_top_feat'] + paths
    df = pd.DataFrame(columns=cols)

    print(drug)
    print('nodes of interest:', len(nodes_of_interest))

    for node in nodes_of_interest:
        info = {"hgnc": node}

        if node in conv_table: 
            info['hgnc'] = conv_table[node]

        if node in features:
            info['attribution'] = feature_attr.loc[node][drug]
            info['rank'] = ranks[node]
            info['is_feature'] = 1
        else:
            info['attribution'] = np.nan
            info['rank'] = np.nan
            info['is_feature'] = 0

        info['is_1H_from_pathway'] = 1*(node in one_hop_from_pathway)
        info['is_1H_from_top_feat'] = 1*(node in one_hop_from_top_feats)
        info['is_top_feat'] = 1*(node in top_features.index)

        for path in paths:
            info[path] =  1*(node in (pathway.loc[pathway[0] == path][1].unique()))

        df.loc[node] = info




    df['score'] = 0.5*(df['is_1H_from_pathway'] + df['is_1H_from_top_feat']) + df['is_top_feat'] + 1*(df[paths].sum(axis=1) > 0)
    # df['score'] = df['is_1H_from_top_feat']*0.5*(df['is_1H_from_top_feat']==0) + df['is_1H_from_top_feat']  \
                # + 1*(df[paths].sum(axis=1) > 0) + (df[paths].sum(axis=1) == 0)*0.5*df['is_1H_from_pathway']


    df = df.sort_values(['score', 'rank'],ascending=[False, True])

    # df = df.sort_values('rank')
    df.to_excel(writer_a, sheet_name=drug) 


desc = {
    'hgnc':'HGNC gene name',
    'is_feature': 'gene is used as a feature by the model', 
    'attribution': 'attribution value for the feature/gene', 
    'rank': 'ranking of the attribution value for the feature/gene', 
    'is_top_feat': '1 if the feature/gene is in the top features found by kneedle method', 
    'is_1H_from_pathway': '1 if the gene is an immediate neighbor of a member of any of our pathways-of-interest', 
    'is_1H_from_top_feat': '1 if the gene is an immediate neighbor of a top feature/gene',
    'score': 'arbitrary scoring for sorting (0.5*(is_1H_from_pathway+is_1H_from_top_feat) + is_top_feat + is_a_pathway_member)',
    'other columns': '1 if the gene is a member of the specific pathway'
}

# df = pd.Series(df)
df = pd.DataFrame(index=desc.keys())
df['description'] = desc.values()

df.to_excel(writer_a, sheet_name='legend')
writer_a.save()

