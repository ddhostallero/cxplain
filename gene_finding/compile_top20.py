import pandas as pd 
import os
from borda import rank_aggregate_Borda

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


if not os.path.exists('results/%s/aggregated/'%folder):
    os.mkdir('results/%s/aggregated/'%folder)

writer_m = pd.ExcelWriter('results/%s/aggregated/ensemble_samplewise_mean.xlsx'%folder, engine='xlsxwriter')
writer_b = pd.ExcelWriter('results/%s/aggregated/ensemble_samplewise_borda.xlsx'%folder, engine='xlsxwriter')
writer_a = pd.ExcelWriter('results/%s/aggregated/agg_ensemble_samplewise_borda.xlsx'%folder, engine='xlsxwriter')

for drug in drugs:
    drug_genes_mean = pd.DataFrame(index=range(20), columns=[1])
    drug_genes_borda = pd.DataFrame(index=range(20), columns=[1])
    
    genes = pd.read_csv('results/%s/%s/genes.csv'%(folder, drug))
    drug_genes_mean[1] = genes.loc[range(20)]['mean']
    drug_genes_borda[1] = genes.loc[range(20)]['borda']


    from_mean = list(drug_genes_mean[1].values)
    from_borda = list(drug_genes_borda[1].values)
    for g in drug_genes_mean[1].values:
        if g not in from_borda:
            from_borda.append(g)
    for g in drug_genes_borda[1].values:
        if g not in from_mean:
            from_mean.append(g)

    list_of_lists = [from_mean, from_borda]
    agg = rank_aggregate_Borda(list_of_lists, len(from_mean), 'geometric_mean')
    # print(agg, len(agg))

    drug_genes_agg = pd.DataFrame(index=agg, columns=['rank@mean', 'rank@borda'])
    # drug_genes_agg['gene'] = agg

    for g in agg:

        x = genes.loc[genes['mean'] == g]
        if len(x) > 0:
            drug_genes_agg.at[g, 'rank@mean'] = x.index[0]+1
        else:
            drug_genes_agg.at[g, 'rank@mean'] = '200+'
        
        x = genes.loc[genes['borda'] == g]
        if len(x) > 0:
            drug_genes_agg.at[g, 'rank@borda'] = x.index[0]+1
        else:
            drug_genes_agg.at[g, 'rank@borda'] = '200+'

    drug_genes_agg.to_excel(writer_a, sheet_name=drug)   
    drug_genes_mean.to_excel(writer_m, sheet_name=drug, index=False, header=False)  
    drug_genes_borda.to_excel(writer_b, sheet_name=drug, index=False, header=False)   

writer_m.save()
writer_b.save()
writer_a.save()



# if not os.path.exists('results/CX_ind1/aggregated/'):
#   os.mkdir('results/CX_ind1/aggregated/')

# writer_m = pd.ExcelWriter('results/CX_ind1/aggregated/individual_samplewise_mean.xlsx', engine='xlsxwriter')
# writer_b = pd.ExcelWriter('results/CX_ind1/aggregated/individual_samplewise_borda.xlsx', engine='xlsxwriter')

# for drug in drugs:
#   drug_genes_mean = pd.DataFrame(index=range(20), columns=range(1,11))
#   drug_genes_borda = pd.DataFrame(index=range(20), columns=range(1,11))
    
#   for seed in range(1, 11):
#       genes = pd.read_csv('results/CX_ind1/%s/seed%d/genes.csv'%(drug, seed))
#       drug_genes_mean[seed] = genes.loc[range(20)]['mean']
#       drug_genes_borda[seed] = genes.loc[range(20)]['borda']

#   drug_genes_mean.to_excel(writer_m, sheet_name=drug, index=False)    
#   drug_genes_borda.to_excel(writer_b, sheet_name=drug, index=False)   

#   # drug_genes_mean.to_csv('results/CX_ind1/aggregated/%s_mean.csv'%drug, index=False)
#   # drug_genes_borda.to_csv('results/CX_ind1/aggregated/%s_borda.csv'%drug, index=False)

# writer_m.save()
# writer_b.save()