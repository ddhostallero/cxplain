import torch
import numpy as np

def generate_random_masks(n_random_mask, pathway_matrix):
    """
    returns mask of size (n_mask_sets*n_random_mask, n_genes), where n_mask_sets is the number of unique pathway sizes 
    """

    ns_with_mask = []
    random_masks_list = []
    mask_index = []
    n_mask_sets = -1 
    n_pathways, n_genes = pathway_matrix.shape

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
    n_mask_sets += 1 # because we started from -1 (equal to len(random_mask_list))

    mask_index_matrix = torch.zeros((n_mask_sets, n_pathways))
    for i in range(n_pathways):
        mask_index_matrix[mask_index[i], i] = 1

    return random_masks, mask_index_matrix, n_mask_sets

def precalculate_omegas(model, gdsc_expr, gdsc_dr, pathway_matrix, loss, mode='none', return_sum_one=True):


    if mode not in ['none', 'scaled', 'scaled-difference', 'difference', 'delta-of-delta', 'delta-scaled']:
        print(mode + "not in allowed modes")
        exit()
        
    no_expect_modes = ['none', 'scaled', 'delta-scaled']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    x_train = torch.FloatTensor(gdsc_expr.values)
    y_train = torch.FloatTensor(gdsc_dr.values).view(-1, 1).to(device)
    print(pathway_matrix.shape)

    for m in model.model_list:
        m.to(device)

    model.to(device)
    model.eval()

    y_pred = model(x_train.to(device))#.cpu().detach().numpy()
    print(y_pred.shape)
    model_loss = loss(y_pred, y_train) # (n_samples, n_out)
    print("model_loss", model_loss.shape)

    n_genes = x_train.shape[1]
    n_pathways = len(pathway_matrix)
    mask = torch.ones((n_pathways, n_genes)) - torch.Tensor(pathway_matrix)

    if mode not in no_expect_modes:
        n_random_mask = 500    
        print('creating random bl...')
        random_masks, mask_index_matrix, n_mask_sets = generate_random_masks(n_random_mask, pathway_matrix)
        mask_index_matrix= mask_index_matrix.to(device)

    list_of_errors_without_pathway = []
    list_of_baseline_errors = []

    with torch.no_grad():
        for i, sample in enumerate(x_train):
            if (i+1) % 100 == 0: print(i+1)

            # get model outputs without pre-defined pathways
            masked_sample = sample*mask # (n_pathways, n_genes)
            data = torch.utils.data.TensorDataset(masked_sample)
            data = torch.utils.data.DataLoader(data, batch_size=1024, shuffle=False)
            
            out_without_pathway = []
            for [x] in data:
                x = x.to(device)
                out_without_pathway.append(model(x))

            
            out_without_pathway = torch.cat(out_without_pathway, axis=0) # (n_pathways, n_out) 
            error_without_pathway = loss(out_without_pathway, y_train[i].repeat(n_pathways, 1)).mean(axis=1)    # (n_pathways)
            list_of_errors_without_pathway.append(error_without_pathway.unsqueeze(0)) # (1, n_pathways)

            if mode not in no_expect_modes:
                # get model output with random genes zero'd out (i.e. random pathways)
                masked_sample = sample*random_masks #(n_masked_sets*n_random_masks, n_genes)
                data = torch.utils.data.TensorDataset(masked_sample)
                data = torch.utils.data.DataLoader(data, batch_size=1024, shuffle=False)

                out_without_random_set = []
                for [x] in data:
                    x = x.to(device)
                    out_without_random_set.append(model(x))

                out_without_random_set = torch.cat(out_without_random_set, axis=0) # (n_mask_sets*n_random_mask, n_out)
                error_baseline = loss(out_without_random_set, y_train[i].repeat(n_mask_sets*n_random_mask, 1))
                error_baseline = error_baseline.view(n_mask_sets, n_random_mask)
                error_baseline = error_baseline.mean(axis=1).unsqueeze(0)        # (1, n_mask_sets)
                error_baseline = torch.matmul(error_baseline, mask_index_matrix) # (1, n_pathways)
                list_of_baseline_errors.append(error_baseline) #(1, n_pathways)


    # (n_samples. n_pathways)
    if mode not in no_expect_modes:
        error_baseline = torch.cat(list_of_baseline_errors)
    else:
        error_baseline = None

    error_without_pathway = torch.cat(list_of_errors_without_pathway)
    model_loss = model_loss.repeat(1, n_pathways)

    if mode == 'scaled-difference':
        scaler = torch.FloatTensor(pathway_matrix.sum(axis=1)).unsqueeze(0).to(device)
        omega = (error_without_pathway - error_baseline)/scaler - model_loss

    elif mode == 'delta-of-delta':
        omega = error_without_pathway - error_baseline
    
    elif mode == 'difference':
        omega = error_without_pathway - error_baseline - model_loss

    elif mode == 'scaled':
        scaler = torch.FloatTensor(pathway_matrix.sum(axis=1)).unsqueeze(0).to(device)
        omega = error_without_pathway/scaler - model_loss

    elif mode == 'delta-scaled':
        scaler = torch.FloatTensor(pathway_matrix.sum(axis=1)).unsqueeze(0).to(device)
        omega = (error_without_pathway - model_loss)/scaler

    elif mode == 'none':
        omega = error_without_pathway - model_loss


    ones = torch.ones(n_pathways)
    small_number = torch.ones(omega.shape)*1e-7
    omega = torch.max(omega, small_number.to(device))

    if return_sum_one:
        omega = omega/omega.sum(axis=-1, keepdim=True)
    
    if error_baseline is not None:
        error_baseline = error_baseline.cpu().detach().numpy()

    return omega.cpu().detach().numpy(), error_baseline


def precalculate_omegas_scaled(model, gdsc_expr, gdsc_dr, pathway_matrix, loss):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    x_train = torch.FloatTensor(gdsc_expr.values)
    y_train = torch.FloatTensor(gdsc_dr.values).view(-1, 1).to(device)
    print(pathway_matrix.shape)


    # if type(model) == EnsModel:
    for m in model.model_list:
        m.to(device)

    model.to(device)
    model.eval()

    y_pred = model(x_train.to(device))#.cpu().detach().numpy()
    print(y_pred.shape)
    model_loss = loss(y_pred, y_train) # (n_samples, n_out)
    print("model_loss", model_loss.shape)

    n_genes = x_train.shape[1]
    n_pathways = len(pathway_matrix)
    mask = torch.ones((n_pathways, n_genes)) - torch.Tensor(pathway_matrix)




    list_of_errors_without_pathway = []
    with torch.no_grad():
        for i, sample in enumerate(x_train):
            if (i+1) % 100 == 0: print(i+1)

            # get model outputs without pre-defined pathways
            masked_sample = sample*mask # (n_pathways, n_genes)
            data = torch.utils.data.TensorDataset(masked_sample)
            data = torch.utils.data.DataLoader(data, batch_size=1024, shuffle=False)
            
            out_without_pathway = []
            for [x] in data:
                x = x.to(device)
                out_without_pathway.append(model(x))

            
            out_without_pathway = torch.cat(out_without_pathway, axis=0) # (n_pathways, n_out) 
            error_without_pathway = loss(out_without_pathway, y_train[i].repeat(n_pathways, 1)).mean(axis=1)    # (n_pathways)
            list_of_errors_without_pathway.append(error_without_pathway.unsqueeze(0)) # (1, n_pathways)

    # (n_samples. n_pathways)
    # error_baseline = torch.cat(list_of_baseline_errors)
    error_without_pathway = torch.cat(list_of_errors_without_pathway)
    model_loss = model_loss.repeat(1, n_pathways)


    scaler = torch.FloatTensor(pathway_matrix.sum(axis=1)).unsqueeze(0).to(device)

    omega = error_without_pathway/scaler - model_loss
    ones = torch.ones(n_pathways)
    small_number = torch.ones(omega.shape)*1e-7
    omega = torch.max(omega, small_number.to(device))

    omega = omega/omega.sum(axis=-1, keepdim=True)
    print(omega.shape)
    # exit()
    # omega = omega.view(-1, n_pathways)
    return omega.cpu().detach().numpy(), None