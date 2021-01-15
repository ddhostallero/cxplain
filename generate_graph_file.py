import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



TCGA_GENE_EXPRESSION = '../../drp-data/preprocessed/gdsc_tcga/tcga_labeled_log2_gene_expr.csv'
tcga = pd.read_csv(TCGA_GENE_EXPRESSION, index_col=0)
genes = tcga.index
print("feature genes:", len(genes))

pathway_file = '../../drp-data/pathways/9606.enrichr_pathway.edge'
pathway = pd.read_csv(pathway_file, sep='\t', header=None)
print("pathways:", pathway[0].nunique())
print("pathway genes:", pathway[1].nunique())

ppi_file = '../../drp-data/pathways/9606.STRING_experimental.edge'
ppi = pd.read_csv(ppi_file, sep='\t', header=None)
print("PPI original edges:", len(ppi))
ppi['norm_score'] = ppi[2]/ppi[2].max()
ppi = ppi.loc[ppi['norm_score'] > 0.5]
print("PPI filtered edges:", len(ppi))
nodes = list(set(ppi[0]).union(set(ppi[1])))
print("PPI nodes:", len(nodes) )

mean_attribution_file = '../../'



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

folder = 'CX_ens10'

attr_dict = {}
for i, drug in enumerate(drugs):
    print(drug)
    _, _, _, test_tcga_expr = dataset.filter_and_normalize_data(drug)
    exp = CXPlain.load('gene_finding/results/%s/%s/explainer'%(folder, drug), custom_model_loader=None, relpath=True)
    attr,_ = exp.explain(test_tcga_expr.values)
    attr = pd.DataFrame(attr, index=test_tcga_expr.index, columns=dataset.genes)
    attr_dict[drug]=attr

fig, axes = plt.subplots(7, 2, figsize=(14, 35))

writer_a = pd.ExcelWriter('gene_finding/results/%s/top_genes_mean_aggregation.xlsx'%folder, engine='xlsxwriter')
# writer_b = pd.ExcelWriter('gene_finding/results/%s/all_attributions.xlsx'%folder, engine='xlsxwriter')

conv = pd.DataFrame(index=dataset.genes, columns=['hgnc'])
conv['hgnc'] = dataset.hgnc

all_results = pd.DataFrame(index=dataset.genes, columns=['hgnc'] + drugs)
all_results['hgnc'] = dataset.hgnc

for i, drug in enumerate(drugs):
    ax = axes[i%7][i//7]


    attr_drug = attr_dict[drug].mean(axis=0).sort_values(ascending=False) # sort mean attribution
    attr_drug = attr_drug/attr_drug.max()                                 # normalize
    
    all_results[drug] = attr_drug

    # find knee
    kneedle = KneeLocator(np.arange(len(attr_drug)), attr_drug, curve='convex', direction='decreasing')
    thresh = kneedle.knee

    df = pd.DataFrame(index=range(1,thresh+1), columns=['ensembl', 'hgnc'])
    df['ensembl'] = attr_drug.index[:thresh]
    df['hgnc'] = conv.loc[attr_drug.index[:thresh]]['hgnc'].values
    df.to_excel(writer_a, sheet_name=drug) 

    ax.plot(range(len(dataset.hgnc)), attr_drug)
    ax.set_title("%s (%d)"%(drug, thresh))
    ax.set_ylabel('attribution')
    ax.axvline(thresh)

plt.savefig('gene_finding/results/%s/kneedle_thresholds.pdf'%folder)
writer_a.save()

all_results.to_csv('gene_finding/results/%s/all_attributions.csv'%folder)
