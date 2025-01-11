import torch
from torch import nn

from models.utils import get_index_embedding, calc_distogram

class EdgeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        """
        Initialize the EdgeFeatureNet module.

        Parameters:
            module_cfg: Configuration object containing hyperparameters.

        The module is designed to compute edge features from node features and structural information.
        """
        super(EdgeFeatureNet, self).__init__()
        self._cfg = module_cfg

        self.c_s = self._cfg.c_s # Dimension of input node features
        self.c_p = self._cfg.c_p # Dimension of processed node features
        self.feat_dim = self._cfg.feat_dim # Dimension of feature embeddings

        # Linear layer to project node features to feature_dim
        self.linear_s_p = nn.Linear(self.c_s, self.feat_dim)
        # Linear layer to process relative position embeddings
        self.linear_relpos = nn.Linear(self.feat_dim, self.feat_dim)

        # Total dimension of edge features
        total_edge_feats = self.feat_dim * 3 + self._cfg.num_bins * 2
        if self._cfg.embed_chain:
            total_edge_feats += 1
        if self._cfg.embed_diffuse_mask:
            total_edge_feats += 2

        # Sequential layers to embed edge features
        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.LayerNorm(self.c_p),
        )

    def embed_relpos(self, r):
        # AlphaFold 2 Algorithm 4 & 5
        # Based on OpenFold utils/tensor_utils.py
        """
        Embed relative positions between residues.

        Parameters:
            r: Tensor of shape [b, n_res], representing residue indices.

        Returns:
            relpos_feats: Processed relative position features of shape [b, n_res, n_res, feat_dim].

        This method computes relative position embeddings based on the difference between residue indices.
        """
        # Compute differences between residue indices
        d = r[:, :, None] - r[:, None, :]
        # Get positional embeddings
        pos_emb = get_index_embedding(d, self._cfg.feat_dim, max_len=2056)
        # Process embeddings with a linear layer
        return self.linear_relpos(pos_emb)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        """
        Concatenate node features in a pairwise manner.

        Parameters:
            feats_1d: Tensor of shape [b, n_res, feat_dim], node features.
            num_batch: Number of batches.
            num_res: Number of residues.

        Returns:
            concatenated_feats: Tensor of shape [b, n_res, n_res, 2*feat_dim].

        This method creates pairwise features by repeating and concatenating node features.
        """
        # Tile and concatenate to form pairwise features
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res, num_res, -1])

    def forward(self, s, t, sc_t, p_mask, diffuse_mask):
        """
        Forward pass to compute edge features.

        Parameters:
            s: Node features of shape [b, n_res, c_s].
            t: Backbone coordinates of shape [b, n_res, 3].
            sc_t: Side chain coordinates of shape [b, n_res, 3].
            p_mask: Padding mask of shape [b, n_res].
            diffuse_mask: Diffuse mask of shape [b, n_res].

        Returns:
            edge_feats: Computed edge features of shape [b, n_res, n_res, c_p].

        This method computes edge features from node features and structural information.
        """
        num_batch, num_res, _ = s.shape

        # [b, n_res, c_p]
        p_i = self.linear_s_p(s)
        cross_node_feats = self._cross_concat(p_i, num_batch, num_res)

        # [b, n_res]
        r = torch.arange(
            num_res, device=s.device).unsqueeze(0).repeat(num_batch, 1)
        relpos_feats = self.embed_relpos(r)

        dist_feats = calc_distogram(
            t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)
        sc_feats = calc_distogram(
            sc_t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)

        all_edge_feats = [cross_node_feats, relpos_feats, dist_feats, sc_feats]
        if self._cfg.embed_diffuse_mask:
            diff_feat = self._cross_concat(diffuse_mask[..., None], num_batch, num_res)
            all_edge_feats.append(diff_feat)
        edge_feats = self.edge_embedder(torch.concat(all_edge_feats, dim=-1))
        edge_feats *= p_mask.unsqueeze(-1)
        return edge_feats