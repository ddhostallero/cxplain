from scipy.stats.mstats import gmean
from numpy import mean

def rank_aggregate_Borda(list_of_list, k, method):
    """
    This function receives a list of list of genes and a number k. The genes are
    ranked based on Borda's method. Final output
    is an aggregated ranked list of k genes.
    """
    dic_tmp = {key:[] for key in list_of_list[0]} #A dictionary with keys being gene names
    list_length = len(list_of_list[0])    
    for list1 in list_of_list:
        for i in range(list_length):
            dic_tmp[list1[i]].append(list_length - i)
            
    if method == 'arithmetic_mean':
        dic_tmp_agg = {key:mean(dic_tmp[key]) for key in dic_tmp}    
    if method == 'geometric_mean':    
        dic_tmp_agg = {key:gmean(dic_tmp[key]) for key in dic_tmp}
    
    ranked_tmp = sorted(dic_tmp_agg, key=dic_tmp_agg.get, reverse=True)   #sort descending
   
    return(ranked_tmp[0:k])