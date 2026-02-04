import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from torch_geometric.nn.conv import GCNConv
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
device = torch.device("cuda" if torch.cuda.is_available() == True else 'cpu')


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class GeneGraph(nn.Module):
    def __init__(self, input_dim, h, hid_dim, phi, dropout=0):
        super().__init__()
        assert input_dim % h == 0

        self.d_k = input_dim // h
        self.h = h
        self.linears = clones(nn.Linear(input_dim, self.d_k * self.h), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.Wo = nn.Linear(h, 1)
        self.phi = nn.Parameter(torch.tensor(phi), requires_grad=True)

    def forward(self, query, key):
        query, key = [l(x).view(query.size(0), -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]

        attns = self.attention(query.squeeze(2), key.squeeze(2))
        adj = torch.where(attns >= self.phi, torch.ones(attns.shape).to(device), torch.zeros(attns.shape).to(device))

        return adj

    def attention(self, query, key):
        d_k = query.size(-1)
        scores = torch.bmm(query.permute(1, 0, 2), key.permute(1, 2, 0)) \
                 / math.sqrt(d_k)
        scores = self.Wo(scores.permute(1, 2, 0)).squeeze(2)
        p_attn = F.softmax(scores, dim=1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return p_attn


def LocationGraph(loc, k=6, metric="euclidean"):
    if not isinstance(loc, np.ndarray):
        loc = np.asarray(loc)

    knn = kneighbors_graph(
        loc,
        n_neighbors=k,
        mode="connectivity",   
        metric=metric,
        include_self=False
    )  

    adj = knn.maximum(knn.T)
    adj = adj.tocoo()

    return adj


def masked_intersection_cosine(X: csr_matrix, i: int, j: int, min_inter=20, eps=1e-8):
    si, ei = X.indptr[i], X.indptr[i+1]
    sj, ej = X.indptr[j], X.indptr[j+1]
    idx_i, val_i = X.indices[si:ei], X.data[si:ei]
    idx_j, val_j = X.indices[sj:ej], X.data[sj:ej]

    p = q = 0
    dot = 0.0
    ni = 0.0
    nj = 0.0
    inter = 0
    while p < len(idx_i) and q < len(idx_j):
        if idx_i[p] == idx_j[q]:
            vi = val_i[p]; vj = val_j[q]
            dot += vi * vj
            ni += vi * vi
            nj += vj * vj
            inter += 1
            p += 1; q += 1
        elif idx_i[p] < idx_j[q]:
            p += 1
        else:
            q += 1

    if inter < min_inter or ni < eps or nj < eps:
        return 0.0, inter, len(idx_i), len(idx_j)

    cos_I = dot / (np.sqrt(ni) * np.sqrt(nj) + eps)
    overlap_penalty = inter / (np.sqrt(len(idx_i) * len(idx_j)) + eps)
    w = cos_I * overlap_penalty
    return w, inter, len(idx_i), len(idx_j)


def build_view2_weights(X, edges, min_inter=20):
    E = edges.shape[1]
    w = np.zeros(E, dtype=np.float32)
    for e in range(E):
        i, j = int(edges[0, e]), int(edges[1, e])
        wij, inter, di, dj = masked_intersection_cosine(X, i, j, min_inter=min_inter)
        w[e] = wij
    return w


def topk_per_node(edges, w, N, k=10):
    src = edges[0]
    keep = np.zeros(w.shape[0], dtype=bool)
    for i in range(N):
        idx = np.where(src == i)[0]
        if len(idx) == 0:
            continue
        
        top = idx[np.argsort(w[idx])[-k:]]
        keep[top] = True
    return edges[:, keep], w[keep]


def sim(z1, z2, hidden_norm):
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.T)


def gcl_loss(z, z_aug, adj, tau, hidden_norm=True):
    f = lambda x: torch.exp(x / tau)
    intra_view_sim = f(sim(z, z, hidden_norm))
    inter_view_sim = f(sim(z, z_aug, hidden_norm))

    positive = inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)

    loss = positive / (intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag())

    adj_count = torch.sum(adj, 1) * 2 + 1
    loss = torch.log(loss) / adj_count

    return -torch.mean(loss, 0)


def gcl_loss_all(alpha1, alpha2, z, z_aug, adj, adj_aug, tau, hidden_norm=True):
    loss = alpha1 * gcl_loss(z, z_aug, adj, tau, hidden_norm) + alpha2 * gcl_loss(z_aug, z, adj_aug, tau, hidden_norm)

    return loss


def sample_view_mask(
    num_nodes: int,
    num_views: int = 3,
    p_drop: float = 0.1,
    device=None,
    per_node: bool = True,
):
    assert 0.0 <= p_drop < 1.0
    device = device or torch.device("cpu")

    if per_node:
        keep = (torch.rand(num_nodes, num_views, device=device) > p_drop).float()
        
        row_sum = keep.sum(dim=1, keepdim=True)  
        bad = (row_sum == 0).squeeze(1)          
        if bad.any():
            idx = bad.nonzero(as_tuple=False).squeeze(1)
            rand_view = torch.randint(0, num_views, (idx.numel(),), device=device)
            keep[idx, rand_view] = 1.0
        return keep
    else:
        keep = (torch.rand(num_views, device=device) > p_drop).float()  
        if keep.sum() == 0:
            keep[torch.randint(0, num_views, (1,), device=device)] = 1.0
        return keep.unsqueeze(0).repeat(num_nodes, 1)


class ViewDropout(nn.Module):
    def __init__(self, d_model: int, p_drop: float = 0.1, per_node: bool = True):
        super().__init__()
        self.p_drop = p_drop
        self.per_node = per_node
        self.missing = nn.Parameter(torch.zeros(3, d_model))
        nn.init.normal_(self.missing, mean=0.0, std=0.02)

    def forward(self, z1, z2, z3, training: bool = True):
        if (not training) or (self.p_drop <= 0.0):
            N = z1.size(0)
            mask = torch.ones(N, 3, device=z1.device, dtype=z1.dtype)
            return z1, z2, z3, mask

        N = z1.size(0)
        mask = sample_view_mask(
            num_nodes=N,
            num_views=3,
            p_drop=self.p_drop,
            device=z1.device,
            per_node=self.per_node,
        ).to(dtype=z1.dtype)

        m1 = mask[:, 0:1]  
        m2 = mask[:, 1:2]
        m3 = mask[:, 2:3]

        z1d = m1 * z1 + (1.0 - m1) * self.missing[0].unsqueeze(0)
        z2d = m2 * z2 + (1.0 - m2) * self.missing[1].unsqueeze(0)
        z3d = m3 * z3 + (1.0 - m3) * self.missing[2].unsqueeze(0)

        return z1d, z2d, z3d, mask


class MLPExpert(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class MLPSoftmaxRouter(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_experts: int,
        hidden_dim: int = None,
        temperature: float = 0.89,
        top_k: int = None,          
    ):
        super().__init__()
        h = hidden_dim or in_dim
        self.fc1 = nn.Linear(in_dim, h)
        self.fc2 = nn.Linear(h, num_experts)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(float(temperature))))
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x):
        temp = torch.exp(self.log_temp).clamp(min=1e-3, max=100.0)
        h = F.gelu(self.fc1(x))
        logits = self.fc2(h) / temp  

        if self.top_k is None or self.top_k >= self.num_experts:
            return F.softmax(logits, dim=-1)

        topv, topi = torch.topk(logits, self.top_k, dim=-1)
        mask = torch.zeros_like(logits).scatter_(-1, topi, 1.0)
        masked_logits = logits.masked_fill(mask == 0, float("-inf"))
        return F.softmax(masked_logits, dim=-1)  


class FuseMoEFusionLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        expert_hidden_dim: int = None,
        router_hidden_dim: int = 256,
        router_temperature: float = 1.0,
        top_k: int = None,               
        append_mask: bool = True,

        expert_dropout: float = 0.0,
        
        use_residual: bool = True,
        aux_type: str = "none",            
        entropy_target: float = None,    
        eps: float = 1e-8,
    ):
        super().__init__()
        self.d_model = d_model
        self.append_mask = append_mask
        self.use_residual = use_residual
        self.aux_type = aux_type.lower()
        self.entropy_target = entropy_target
        self.eps = eps

        in_dim = 9 * d_model + (3 if append_mask else 0)

        self.ln = nn.LayerNorm(d_model)
        self.ln_pair = nn.LayerNorm(d_model)

        self.router = MLPSoftmaxRouter(
            in_dim=in_dim,
            num_experts=num_experts,
            hidden_dim=router_hidden_dim,
            temperature=router_temperature,
            top_k=top_k,
        )

        h = expert_hidden_dim or d_model
        self.experts = nn.ModuleList([
            MLPExpert(in_dim=in_dim, hidden_dim=h, out_dim=d_model, dropout=expert_dropout)
            for _ in range(num_experts)
        ])

        self.out_ln = nn.LayerNorm(d_model)

    def build_fusion_input(self, z1, z2, z3, mask=None):
        z1 = self.ln(z1)
        z2 = self.ln(z2)
        z3 = self.ln(z3)

        d12 = self.ln_pair(z1 - z2)
        d13 = self.ln_pair(z1 - z3)
        d23 = self.ln_pair(z2 - z3)

        p12 = self.ln_pair(z1 * z2)
        p13 = self.ln_pair(z1 * z3)
        p23 = self.ln_pair(z2 * z3)

        x = torch.cat([z1, z2, z3, d12, d13, d23, p12, p13, p23], dim=-1)  

        if self.append_mask:
            assert mask is not None, "mask (N, 3) must be provided when append_mask=True"
            if mask.dtype != x.dtype:
                mask = mask.to(dtype=x.dtype)
            x = torch.cat([x, mask], dim=-1)

        return x, z1, z2, z3
    
    def forward(self, z1, z2, z3, mask=None, return_gate: bool = False):
        x, z1n, z2n, z3n = self.build_fusion_input(z1, z2, z3, mask=mask)

        gate = self.router(x)             
        
        outs = torch.stack([exp(x) for exp in self.experts], dim=1)  
        z = torch.sum(gate.unsqueeze(-1) * outs, dim=1)              

        if self.use_residual and (mask is not None):
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            z_res = (mask[:, 0:1] * z1n + mask[:, 1:2] * z2n + mask[:, 2:3] * z3n) / denom
            z = z + z_res
        elif self.use_residual:
            z = z + (z1n + z2n + z3n) / 3.0

        z = self.out_ln(z)

        if return_gate:
            return z, gate
        return z


def compute_prototypes_hard(
    z: torch.Tensor,          
    labels: torch.Tensor,     
    K: int,
    eps: float = 1e-12):
    
    assert z.dim() == 2 and labels.dim() == 1
    N, d = z.shape
    mu = torch.zeros((K, d), device=z.device, dtype=z.dtype)
    mu.index_add_(0, labels, z)                 
    mu = F.normalize(mu, p=2, dim=1, eps=eps)   
    counts = torch.bincount(labels, minlength=K).to(device=z.device)
    return mu, counts


def prototype_alignment_loss(
    mu_v1: torch.Tensor,      
    mu_v2: torch.Tensor,      
    mu_v3: torch.Tensor,      
    sigma: float = 1e-3,
    valid_mask: torch.Tensor = None,  
    renorm_after_noise: bool = True):
    
    assert mu_v1.shape == mu_v2.shape == mu_v3.shape

    noise = torch.randn_like(mu_v2)
    u_hat = mu_v2 + sigma * noise                       
    if renorm_after_noise:
        u_hat = F.normalize(u_hat, p=2, dim=1, eps=1e-12)

    per_k = (u_hat - mu_v1).pow(2).sum(dim=1) + (u_hat - mu_v3).pow(2).sum(dim=1)  

    if valid_mask is not None:
        per_k = per_k[valid_mask]

    if per_k.numel() == 0:
        return torch.zeros((), device=mu_v1.device, dtype=mu_v1.dtype)
    return per_k.mean()


class Model(nn.Module):
    def __init__(
        self,
        input_dim,
        num_head,
        hidden_dim,
        gcn_dim1,
        gcn_dim2,
        mlp_dim1,
        mlp_dim2,
        num_experts,
        K,
        eta,
        sigma,
        tau,
        a_1,
        a_2,
        lambda_1,
        lambda_2,
        v_drop,
        p_drop,
        seed):
        
        super().__init__()
        self.gene_graph = GeneGraph(input_dim, num_head, hidden_dim, eta, p_drop) 
        self.gcn_1 = GCNConv(input_dim, gcn_dim1)
        self.gcn_2 = GCNConv(mlp_dim1, gcn_dim2)
        self.mlp_1 = nn.Linear(gcn_dim1, mlp_dim1)
        self.mlp_2 = nn.Linear(gcn_dim2, mlp_dim2)
        self.view_dropout = ViewDropout(d_model=mlp_dim1, p_drop=v_drop, per_node=True)
        self.moe = fusion = FuseMoEFusionLayer(mlp_dim1, num_experts)
        
        self.tau_1 = tau
        self.sigma = sigma
        
        self.K = K
        self.a_1 = a_1
        self.a_2 = a_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = nn.Dropout(p=p_drop)
        self.seed = seed
    
    def forward(self, x, loc):
        view1_adj = self.gene_graph(x,x)   
        view3_adj = torch.from_numpy(
            LocationGraph(loc.cpu().detach().numpy()).toarray()
        ).float().to(x.device)

        view1_edge_index = torch.nonzero(view1_adj == 1).T
        view3_edge_index = torch.nonzero(view3_adj == 1).T

        X_csr = csr_matrix(x.detach().cpu().numpy())  
        
        w_view2 = build_view2_weights(X_csr, view1_edge_index, min_inter=20)

        view2_edge_index, _ = topk_per_node(view1_edge_index.cpu().detach().numpy(), w_view2, N = x.shape[0], k=10)
        view2_edge_index = torch.tensor(view2_edge_index).to(device)
        
        view2_edge_index_T = view2_edge_index.T
        view2_adj = torch.zeros(x.shape[0], x.shape[0]).to(device)
        view2_adj[view2_edge_index_T[:, 0], view2_edge_index_T[:, 1]] = 1
        
        z1_emb = self.mlp_1(self.gcn_1(x, view1_edge_index))   
        z2_emb = self.mlp_1(self.gcn_1(x, view2_edge_index))    
        z3_emb = self.mlp_1(self.gcn_1(x, view3_edge_index)) 
        
        loss_1 = gcl_loss_all(self.a_1, self.a_2, z1_emb, z2_emb, view1_adj, view2_adj, self.tau_1)

        z1d, z2d, z3d, mask = self.view_dropout(z1_emb, z2_emb, z3_emb, training=True)
        z_tild_emb = self.moe(z1d, z2d, z3d, mask)
        
        kmeans = KMeans(n_clusters=self.K, random_state=self.seed, n_init=20).fit(z_tild_emb.cpu().detach().numpy())
        pseudo_label = kmeans.labels_
        
        pseudo_label = pseudo_label.astype(int)
        refine_matrix = (pseudo_label[:, None] == pseudo_label[None, :]).astype(int)   
        
        refine_matrix = torch.tensor(refine_matrix).to(x.device)
        view1_adj = view1_adj * refine_matrix
        view2_adj = view2_adj * refine_matrix
        view3_adj = view3_adj * refine_matrix
        
        view1_edge_index = torch.nonzero(view1_adj == 1).T  
        view2_edge_index = torch.nonzero(view2_adj == 1).T    
        view3_edge_index = torch.nonzero(view3_adj == 1).T    
        
        z1_emb = self.mlp_2(self.gcn_2(z1_emb, view1_edge_index))   
        z2_emb = self.mlp_2(self.gcn_2(z2_emb, view2_edge_index))    
        z3_emb = self.mlp_2(self.gcn_2(z3_emb, view3_edge_index))
        
        z1_emb = self.dropout(z1_emb)
        z2_emb = self.dropout(z2_emb)
        z3_emb = self.dropout(z3_emb)

        labels = torch.as_tensor(pseudo_label, device=device, dtype=torch.long)  
        K = int(labels.max().item()) + 1
        
        mu1, cnt1 = compute_prototypes_hard(z1_emb, labels, K)
        mu2, cnt2 = compute_prototypes_hard(z2_emb, labels, K)
        mu3, cnt3 = compute_prototypes_hard(z3_emb, labels, K)
        
        valid_LA = (cnt1 > 0) & (cnt2 > 0) & (cnt3 > 0)
        valid_LC = (cnt1 > 0) & (cnt3 > 0)
        
        loss_2 = prototype_alignment_loss(mu1, mu3, mu2, sigma=self.sigma, valid_mask=valid_LA)

        loss_total = self.lambda_1 * loss_1 + self.lambda_2 * loss_2   

       
        return z_tild_emb, loss_total, loss_1

