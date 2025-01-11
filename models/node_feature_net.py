import torch
from torch import nn
from models.utils import get_index_embedding, get_time_embedding


class NodeFeatureNet(nn.Module):
    """
    NodeFeatureNet is a neural network module designed to process node features 
    in a graph, incorporating positional embeddings, time embeddings, and other 
    relevant features.
    """
    def __init__(self, module_cfg):
        """
        Initialize the NodeFeatureNet with the given configuration.
        
        Parameters:
            module_cfg: Configuration object containing hyperparameters.
        """
        super(NodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        embed_size = self._cfg.c_pos_emb + self._cfg.c_timestep_emb * 2 + 1
        if self._cfg.embed_chain:
            embed_size += self._cfg.c_pos_emb
        self.linear = nn.Linear(embed_size, self.c_s)

    def embed_t(self, timesteps, mask):
        """
        Embed the timesteps into a higher-dimensional space and repeat them 
        for each residue in the sequence.
        
        Parameters:
            timesteps: Tensor of timesteps, shape [batch_size].
            mask: Residue mask tensor, shape [batch_size, num_residues].
        
        Returns:
            Tensor of time embeddings, shape [batch_size, num_residues, c_timestep_emb].
        """
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self, so3_t, r3_t, res_mask, diffuse_mask, pos):
        """
        Forward pass to compute node features from input embeddings and masks.
        
        Parameters:
            so3_t: Tensor of SO3 timesteps, shape [batch_size, num_residues].
            r3_t: Tensor of R3 timesteps, shape [batch_size, num_residues].
            res_mask: Residue mask tensor, shape [batch_size, num_residues].
            diffuse_mask: Diffusion mask tensor, shape [batch_size, num_residues].
            pos: Position indices tensor, shape [batch_size, num_residues].
        
        Returns:
            Tensor of node features, shape [batch_size, num_residues, c_s].
        """
        # s: [b]

        b, num_res, device = res_mask.shape[0], res_mask.shape[1], res_mask.device

        # [b, n_res, c_pos_emb]
        # pos = torch.arange(num_res, dtype=torch.float32).to(device)[None]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb * res_mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        input_feats = [
            pos_emb,
            diffuse_mask[..., None],
            self.embed_t(so3_t, res_mask),
            self.embed_t(r3_t, res_mask)
        ]
        return self.linear(torch.cat(input_feats, dim=-1))
