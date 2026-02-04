import os
import torch
import random
import numpy as np
import scanpy as sc
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import *
from scipy.optimize import linear_sum_assignment as linear_assignment


class CellDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, loc):
        self.X = X  
        self.y = y
        self.loc = loc

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.loc[index]


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.array((ind[0], ind[1])).T

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def evaluate(y_true, y_pred):
    acc= cluster_acc(y_true, y_pred)
    f1=0
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    homo = homogeneity_score(y_true, y_pred)
    comp = completeness_score(y_true, y_pred)
    purity = purity_score(y_true, y_pred)
    return acc, f1, nmi, ari, homo, comp, purity


def loader_construction(data_path):
    adata = sc.read_h5ad(data_path)
    try:
        X_all = adata.X.toarray()
    except:
        X_all = adata.X
    y_all = np.array(adata.obs.loc[:,'ground_truth'])
    loc = adata.obsm['spatial']
    input_dim = X_all.shape[1]

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(y_all)
    n_clusters = len(np.unique(y_all))

    X_train, X_valid_test, y_train, y_valid_test, loc_train, loc_valid_test = train_test_split(X_all, y_all, loc, test_size=0.2, random_state=1)
    X_valid, X_test, y_valid, y_test, loc_valid, loc_test = train_test_split(X_valid_test, y_valid_test, loc_valid_test, test_size=0.5, random_state=1)
    train_set = CellDataset(X_train, y_train, loc_train)
    valid_set = CellDataset(X_valid, y_valid, loc_valid)
    test_set = CellDataset(X_test, y_test, loc_test)

    return train_set, valid_set, test_set, input_dim, n_clusters

