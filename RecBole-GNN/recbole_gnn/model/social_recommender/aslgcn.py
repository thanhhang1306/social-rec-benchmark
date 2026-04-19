"""ASLGCN: LightGCN with a Jaccard-weighted asymmetric social channel gated by a learned scalar."""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

from recbole_gnn.model.abstract_recommender import SocialRecommender


class ASLGCN(SocialRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(ASLGCN, self).__init__(config, dataset)

        self.latent_dim    = config['embedding_size']
        self.n_layers      = config['n_layers']
        self.social_layers = config['social_layers']
        self.reg_weight    = config['reg_weight']
        self.require_pow   = config['require_pow']

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        # net_matrix() takes no 'form' kwarg in RecBole-GNN
        self.social_matrix = dataset.net_matrix().tocoo().astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim)

        # softplus keeps λ_social strictly positive
        self._lambda_raw = nn.Parameter(torch.tensor(-2.0))

        self.mf_loss  = BPRLoss()
        self.reg_loss = EmbLoss()

        self.norm_adj_matrix = self._build_norm_adj().to(self.device)
        self.social_adj      = self._build_social_adj().to(self.device)

        self.restore_user_e = None
        self.restore_item_e = None

        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def _build_norm_adj(self):
        """Symmetrically-normalized user-item bipartite adjacency."""
        R   = self.interaction_matrix
        R_t = R.transpose()
        rows = np.concatenate([R.row,                R_t.row + self.n_users])
        cols = np.concatenate([R.col + self.n_users, R_t.col])
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

    def _build_social_adj(self):
        """Jaccard-weighted asymmetric social adjacency: w_{uv} = r_{uv} / sqrt(out_deg(u) * in_deg(v))."""
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

        out_deg = np.zeros(self.n_users, dtype=np.float32)
        in_deg  = np.zeros(self.n_users, dtype=np.float32)
        np.add.at(out_deg, src_np, 1.0)
        np.add.at(in_deg,  dst_np, 1.0)

        deg_u_sqrt_inv = 1.0 / np.sqrt(out_deg[src_np].clip(min=1.0))
        deg_v_sqrt_inv = 1.0 / np.sqrt(in_deg[dst_np].clip(min=1.0))
        edge_weights   = jaccard_vals * deg_u_sqrt_inv * deg_v_sqrt_inv

        indices = torch.LongTensor(np.vstack([src_np, dst_np]))
        values  = torch.FloatTensor(edge_weights)
        return torch.sparse.FloatTensor(
            indices, values, torch.Size([self.n_users, self.n_users])
        )

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

        h           = user_cf
        social_embs = [h]
        for _ in range(self.social_layers):
            h = torch.sparse.mm(self.social_adj, h)
            social_embs.append(h)
        h_social = torch.mean(torch.stack(social_embs, dim=1), dim=1)

        lambda_social = F.softplus(self._lambda_raw)
        user_final    = user_cf + lambda_social * h_social
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

    @torch.no_grad()
    def get_lambda_social(self):
        return F.softplus(self._lambda_raw).item()


class ASLGCNSymmetric(ASLGCN):
    """ASLGCN ablation: symmetric degree normalization (deg = out_deg + in_deg)."""

    def _build_social_adj(self):
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

        out_deg = np.zeros(self.n_users, dtype=np.float32)
        in_deg  = np.zeros(self.n_users, dtype=np.float32)
        np.add.at(out_deg, src_np, 1.0)
        np.add.at(in_deg,  dst_np, 1.0)
        deg = out_deg + in_deg

        deg_u_sqrt_inv = 1.0 / np.sqrt(deg[src_np].clip(min=1.0))
        deg_v_sqrt_inv = 1.0 / np.sqrt(deg[dst_np].clip(min=1.0))
        edge_weights   = jaccard_vals * deg_u_sqrt_inv * deg_v_sqrt_inv

        indices = torch.LongTensor(np.vstack([src_np, dst_np]))
        values  = torch.FloatTensor(edge_weights)
        return torch.sparse.FloatTensor(
            indices, values, torch.Size([self.n_users, self.n_users])
        )