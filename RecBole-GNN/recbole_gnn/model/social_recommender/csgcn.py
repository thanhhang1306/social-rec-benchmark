"""CSGCN: LightGCN with Jaccard-weighted social aggregation gated by a curriculum weight β(t)."""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

from recbole_gnn.model.abstract_recommender import SocialRecommender


class CSGCN(SocialRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(CSGCN, self).__init__(config, dataset)

        self.latent_dim    = config['embedding_size']
        self.n_layers      = config['n_layers']
        self.reg_weight    = config['reg_weight']
        self.warmup_epochs = config['warmup_epochs']
        self.anneal_epochs = config['anneal_epochs']
        self.require_pow   = config['require_pow']

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        # net_matrix() takes no 'form' kwarg in RecBole-GNN
        self.social_matrix = dataset.net_matrix().tocoo().astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim)

        self.mf_loss  = BPRLoss()
        self.reg_loss = EmbLoss()

        # curriculum state — updated externally by CSGCNTrainer
        self.current_epoch = 0
        self._beta         = 0.0

        self.norm_adj_matrix = self._build_norm_adj().to(self.device)
        self.social_src, self.social_dst, self.jaccard_weights = \
            self._build_social_tensors()

        self.restore_user_e = None
        self.restore_item_e = None

        self.apply(xavier_uniform_initialization)
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
        """Jaccard similarity per social edge."""
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

        dev = self.device
        src = torch.LongTensor(src_np).to(dev)
        dst = torch.LongTensor(dst_np).to(dev)
        jw  = torch.FloatTensor(jaccard_vals).to(dev)
        return src, dst, jw

    def set_epoch(self, epoch: int):
        """Update curriculum weight β (called by CSGCNTrainer before each epoch)."""
        self.current_epoch = epoch
        t = epoch - self.warmup_epochs
        if t <= 0:
            self._beta = 0.0
        elif t >= self.anneal_epochs:
            self._beta = 1.0
        else:
            self._beta = float(t) / float(self.anneal_epochs)

    @property
    def beta(self) -> float:
        return self._beta

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

        user_final = user_cf + self._beta * h_social
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
        reg     = self.reg_loss(u_ego, pos_ego, neg_ego, require_pow=self.require_pow)

        return mf_loss + self.reg_weight * reg

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


class CSGCNNoWarmup(CSGCN):
    """CSGCN ablation: β starts ramping at epoch 0 (no warmup phase)."""
    pass


from recbole.trainer import Trainer


class CSGCNTrainer(Trainer):
    """Trainer that calls model.set_epoch before each epoch to update β."""

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.set_epoch(epoch_idx)
        return super()._train_epoch(
            train_data, epoch_idx,
            loss_func=loss_func,
            show_progress=show_progress
        )