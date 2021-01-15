import pandas as pd 

def get_valid_enrichr_pathways(pathway_nodemap):
	node_map = pd.read_csv(pathway_nodemap, sep='\t', header=None)
	pathways = node_map.loc[node_map[2]=='Property']
	pathways = pathways.loc[pathways[4].str.contains('Homo sapiens')]
	pathways = list(pathways[0].unique())
	print("Number of Homo sapiens pathways: %d"%(len(pathways)))
	return pathways



def load_pathway(pathway_file, genes, list_of_pathways=None, threshold=0, sort=False):
	pathway = pd.read_csv(pathway_file, sep='\t', header=None)
	n_clusters = len(pathway[0].unique())

	df = pd.DataFrame(index=genes)
	genes = pd.Series(genes)

	size = pathway.groupby(0).size()

	if list_of_pathways is not None:
		pathway = pathway.loc[pathway[0].isin(list_of_pathways)]

	pathway_names = []
	for i, clust in enumerate(list_of_pathways):
		if size[clust] < threshold:
			continue

		genes_in_clust = pathway.loc[pathway[0] == clust][1]
		df[i] = list(genes.isin(genes_in_clust)*1)
		pathway_names.append(clust)


	if sort:
		df.columns = pathway_names
		cols = df.sum(axis=0).sort_values().index
		df = df[cols]
		pathway_names = list(cols)

	print("Number of valid pathways: %d"%(len(pathway_names)))
	return df.T.values, pathway_names
