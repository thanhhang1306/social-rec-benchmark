"""HAGN: LightGCN with Jaccard-weighted social aggregation gated per-user by a learned MLP."""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

from recbole_gnn.model.abstract_recommender import SocialRecommender


class HAGN(SocialRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(HAGN, self).__init__(config, dataset)

        self.latent_dim      = config['embedding_size']
        self.n_layers        = config['n_layers']
        self.reg_weight      = config['reg_weight']
        self.require_pow     = config['require_pow']

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        # net_matrix() takes no 'form' kwarg in RecBole-GNN
        self.social_matrix = dataset.net_matrix().tocoo().astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim)

        # gate MLP: [φ_u, log|R_u|+1, log|S_u|+1] → α_u ∈ (0,1)
        self.gate_mlp = nn.Linear(3, 1, bias=True)

        self.mf_loss  = BPRLoss()
        self.reg_loss = EmbLoss()

        self.norm_adj_matrix = self._build_norm_adj().to(self.device)
        (
            self.social_src,
            self.social_dst,
            self.jaccard_weights,
            self.phi,
            self.log_n_interactions,
            self.log_social_deg,
        ) = self._build_social_tensors()

        self.restore_user_e = None
        self.restore_item_e = None

        self.apply(xavier_uniform_initialization)
        # zero init so α_u = sigmoid(0) = 0.5 at start
        nn.init.zeros_(self.gate_mlp.weight)
        nn.init.zeros_(self.gate_mlp.bias)

        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def _build_norm_adj(self):
        """Symmetrically-normalized user-item bipartite adjacency."""
        inter_M   = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        rows = np.concatenate([inter_M.row,                inter_M_t.row + self.n_users])
        cols = np.concatenate([inter_M.col + self.n_users, inter_M_t.col])
        vals = np.ones(len(rows), dtype=np.float32)
        A = sp.csr_matrix(
            (vals, (rows, cols)),
            shape=(self.n_users + self.n_items, self.n_users + self.n_items),
        )
        sumArr = (A > 0).sum(axis=1)
        diag   = np.array(sumArr.flatten())[0] + 1e-7
        diag   = np.power(diag, -0.5)
        D      = sp.diags(diag)
        L      = sp.coo_matrix(D * A * D)
        i      = torch.LongTensor(np.array([L.row, L.col]))
        data   = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, data, torch.Size(L.shape))

    def _build_social_tensors(self):
        """Jaccard per edge plus per-user mean Jaccard (φ_u) and log-degree features."""
        R_csr  = self.interaction_matrix.tocsr()
        S      = self.social_matrix
        src_np = S.row.astype(np.int64)
        dst_np = S.col.astype(np.int64)

        jaccard_vals = np.zeros(len(src_np), dtype=np.float32)
        for e, (u, v) in enumerate(zip(src_np, dst_np)):
            r_u   = R_csr[u]
            r_v   = R_csr[v]
            inter = r_u.minimum(r_v).nnz
            union = r_u.maximum(r_v).nnz
            jaccard_vals[e] = inter / union if union > 0 else 0.0

        phi_np     = np.zeros(self.n_users, dtype=np.float32)
        social_deg = np.zeros(self.n_users, dtype=np.float32)
        np.add.at(phi_np,     src_np, jaccard_vals)
        np.add.at(social_deg, src_np, 1.0)
        phi_np = np.where(social_deg > 0, phi_np / social_deg, 0.0)

        inter_deg = np.array(R_csr.sum(axis=1)).flatten()

        dev         = self.device
        src         = torch.LongTensor(src_np).to(dev)
        dst         = torch.LongTensor(dst_np).to(dev)
        jaccard_w   = torch.FloatTensor(jaccard_vals).to(dev)
        phi         = torch.FloatTensor(phi_np).to(dev)
        log_inter   = torch.log(torch.FloatTensor(inter_deg).to(dev) + 1.0)
        log_soc_deg = torch.log(torch.FloatTensor(social_deg).to(dev) + 1.0)
        return src, dst, jaccard_w, phi, log_inter, log_soc_deg

    def get_ego_embeddings(self):
        return torch.cat([self.user_embedding.weight,
                          self.item_embedding.weight], dim=0)

    def forward(self):
        all_emb  = self.get_ego_embeddings()
        emb_list = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj_matrix, all_emb)
            emb_list.append(all_emb)
        cf_all = torch.mean(torch.stack(emb_list, dim=1), dim=1)
        user_cf, item_cf = torch.split(cf_all, [self.n_users, self.n_items])

        # h_u = Σ r_{uv} e_v / Σ r_{uv}
        n_u, d = self.n_users, self.latent_dim
        weighted_nbr = self.jaccard_weights.unsqueeze(1) * user_cf[self.social_dst]
        h_social = torch.zeros(n_u, d, device=self.device)
        h_social.scatter_add_(0, self.social_src.unsqueeze(1).expand(-1, d), weighted_nbr)
        w_sum = torch.zeros(n_u, device=self.device)
        w_sum.scatter_add_(0, self.social_src, self.jaccard_weights)
        h_social = h_social / w_sum.clamp(min=1e-8).unsqueeze(1)

        gate_input = torch.stack([self.phi, self.log_n_interactions, self.log_social_deg], dim=1)
        alpha = torch.sigmoid(self.gate_mlp(gate_input))

        user_final = user_cf + alpha * h_social
        return user_final, item_cf

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e = self.restore_item_e = None

        user     = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all, item_all = self.forward()
        u_emb   = user_all[user]
        pos_emb = item_all[pos_item]
        neg_emb = item_all[neg_item]

        pos_scores = torch.mul(u_emb, pos_emb).sum(dim=1)
        neg_scores = torch.mul(u_emb, neg_emb).sum(dim=1)
        mf_loss    = self.mf_loss(pos_scores, neg_scores)

        u_ego   = self.user_embedding(user)
        pos_ego = self.item_embedding(pos_item)
        neg_ego = self.item_embedding(neg_item)
        emb_reg = self.reg_loss(u_ego, pos_ego, neg_ego, require_pow=self.require_pow)

        return mf_loss + self.reg_weight * emb_reg

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_all, item_all = self.forward()
        return torch.mul(user_all[user], item_all[item]).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        u_emb  = self.restore_user_e[user]
        scores = torch.matmul(u_emb, self.restore_item_e.transpose(0, 1))
        return scores.view(-1)


class HAGNUniform(HAGN):
    """HAGN ablation: uniform edge weights, gate MLP still fed real Jaccard features."""

    def _build_social_tensors(self):
        src, dst, jaccard_w, phi, log_inter, log_soc_deg = super()._build_social_tensors()
        uniform_w = torch.ones_like(jaccard_w)
        return src, dst, uniform_w, phi, log_inter, log_soc_deg


class HAGNGlobal(HAGN):
    """HAGN ablation: single global learned scalar α = sigmoid(b) instead of the per-user MLP."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        del self.gate_mlp
        self._gate_bias = nn.Parameter(torch.zeros(1))

    def forward(self):
        all_emb  = self.get_ego_embeddings()
        emb_list = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj_matrix, all_emb)
            emb_list.append(all_emb)
        cf_all = torch.mean(torch.stack(emb_list, dim=1), dim=1)
        user_cf, item_cf = torch.split(cf_all, [self.n_users, self.n_items])

        n_u, d = self.n_users, self.latent_dim
        weighted_nbr = self.jaccard_weights.unsqueeze(1) * user_cf[self.social_dst]
        h_social = torch.zeros(n_u, d, device=self.device)
        h_social.scatter_add_(0, self.social_src.unsqueeze(1).expand(-1, d), weighted_nbr)
        w_sum = torch.zeros(n_u, device=self.device)
        w_sum.scatter_add_(0, self.social_src, self.jaccard_weights)
        h_social = h_social / w_sum.clamp(min=1e-8).unsqueeze(1)

        alpha = torch.sigmoid(self._gate_bias)
        user_final = user_cf + alpha * h_social
        return user_final, item_cf


class HAGNFixed(HAGN):
    """HAGN ablation: fixed α=1 (no gate MLP); architecturally equivalent to CSGCN at β=1."""

    def forward(self):
        all_emb  = self.get_ego_embeddings()
        emb_list = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj_matrix, all_emb)
            emb_list.append(all_emb)
        cf_all = torch.mean(torch.stack(emb_list, dim=1), dim=1)
        user_cf, item_cf = torch.split(cf_all, [self.n_users, self.n_items])

        n_u, d = self.n_users, self.latent_dim
        weighted_nbr = self.jaccard_weights.unsqueeze(1) * user_cf[self.social_dst]
        h_social = torch.zeros(n_u, d, device=self.device)
        h_social.scatter_add_(0, self.social_src.unsqueeze(1).expand(-1, d), weighted_nbr)
        w_sum = torch.zeros(n_u, device=self.device)
        w_sum.scatter_add_(0, self.social_src, self.jaccard_weights)
        h_social = h_social / w_sum.clamp(min=1e-8).unsqueeze(1)

        user_final = user_cf + h_social
        return user_final, item_cf

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e = self.restore_item_e = None

        user     = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all, item_all = self.forward()
        u_emb   = user_all[user]
        pos_emb = item_all[pos_item]
        neg_emb = item_all[neg_item]

        pos_scores = torch.mul(u_emb, pos_emb).sum(dim=1)
        neg_scores = torch.mul(u_emb, neg_emb).sum(dim=1)
        mf_loss    = self.mf_loss(pos_scores, neg_scores)

        u_ego   = self.user_embedding(user)
        pos_ego = self.item_embedding(pos_item)
        neg_ego = self.item_embedding(neg_item)
        emb_reg = self.reg_loss(u_ego, pos_ego, neg_ego, require_pow=self.require_pow)

        return mf_loss + self.reg_weight * emb_reg


def _build_social_tensors_fast(interaction_matrix, social_matrix, n_users, device):
    """Vectorized drop-in replacement for HAGN._build_social_tensors() on large graphs."""
    R_csr  = interaction_matrix.tocsr().astype(np.float32)
    S      = social_matrix.tocoo()
    src_np = S.row.astype(np.int64)
    dst_np = S.col.astype(np.int64)
    deg_np = np.array(R_csr.astype(bool).sum(axis=1)).flatten()

    n_items = R_csr.shape[1]
    if n_users * n_items * 4 < 2 * (1024 ** 3):
        R_dense      = torch.FloatTensor(R_csr.toarray())
        intersection = (R_dense[src_np] * R_dense[dst_np]).sum(dim=1).numpy()
    else:
        chunk        = 10000
        intersection = np.zeros(len(src_np), dtype=np.float32)
        for start in range(0, len(src_np), chunk):
            end = min(start + chunk, len(src_np))
            u_d = torch.FloatTensor(R_csr[src_np[start:end]].toarray())
            v_d = torch.FloatTensor(R_csr[dst_np[start:end]].toarray())
            intersection[start:end] = (u_d * v_d).sum(dim=1).numpy()

    union        = deg_np[src_np] + deg_np[dst_np] - intersection
    jaccard_vals = np.where(union > 0, intersection / union, 0.0).astype(np.float32)

    phi_np     = np.zeros(n_users, dtype=np.float32)
    social_deg = np.zeros(n_users, dtype=np.float32)
    np.add.at(phi_np,     src_np, jaccard_vals)
    np.add.at(social_deg, src_np, 1.0)
    phi_np = np.where(social_deg > 0, phi_np / social_deg, 0.0)

    src         = torch.LongTensor(src_np).to(device)
    dst         = torch.LongTensor(dst_np).to(device)
    jw          = torch.FloatTensor(jaccard_vals).to(device)
    phi         = torch.FloatTensor(phi_np).to(device)
    log_inter   = torch.log(torch.FloatTensor(deg_np.astype(np.float32)).to(device) + 1.0)
    log_soc_deg = torch.log(torch.FloatTensor(social_deg).to(device) + 1.0)
    return src, dst, jw, phi, log_inter, log_soc_deg