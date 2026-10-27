import torch
import torch.nn as nn


class Backbone(nn.Module):
    def __init__(self, data, cfg):
        super().__init__()
        d_model = cfg["d_model"]
        window = data.window
        self.graph_hops = cfg["graph_hops"]
        self.graph_scale = 0.0
        self.register_buffer("poi_cat", data.poi_cat.clone(), persistent=False)
        self.register_buffer("poi_latlon", data.poi_latlon.clone(), persistent=False)
        self.register_buffer("adjacency", data.adjacency.clone(), persistent=False)
        self.user_emb = nn.Embedding(data.num_users, d_model)
        self.poi_emb = nn.Embedding(data.num_pois, cfg["poi_dim"])
        self.cat_emb = nn.Embedding(data.num_cats, cfg["cat_dim"])
        self.token_proj = nn.Linear(cfg["poi_dim"] + cfg["cat_dim"] + 2, d_model)
        self.graph_token_dropout = nn.Dropout(cfg["dropout"])
        self.embedding_dropout = nn.Dropout(cfg["embedding_dropout"])
        self.pos_emb = nn.Parameter(torch.zeros(1, window + 2, d_model))
        layer = nn.TransformerEncoderLayer(d_model, cfg["heads"], cfg["ffn"], cfg["dropout"], activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg["layers"], enable_nested_tensor=False)
        self.output_dropout = nn.Dropout(cfg["output_dropout"])
        self.out = nn.Linear(d_model, data.num_pois)
        self.graph_mlp = nn.Sequential(nn.Linear(cfg["poi_dim"], cfg["graph_hidden"]), nn.GELU(),
                                       nn.Dropout(cfg["graph_dropout"]), nn.Linear(cfg["graph_hidden"], d_model))
        self.graph_norm = nn.LayerNorm(d_model)

    def forward(self, user, window, pad):
        cat = self.poi_cat[window].masked_fill(pad, 0)
        latlon = self.poi_latlon[window].masked_fill(pad.unsqueeze(-1), 0.0)
        tokens = self.token_proj(torch.cat([self.poi_emb(window), self.cat_emb(cat), latlon], dim=-1))
        x = torch.cat([self.user_emb(user).unsqueeze(1), tokens], dim=1)
        keep = torch.zeros(user.shape[0], 1, dtype=torch.bool, device=user.device)
        mask = [keep, pad]
        if self.graph_scale > 0:
            g = self.poi_emb.weight
            for _ in range(self.graph_hops):
                g = torch.sparse.mm(self.adjacency, g)
            g = self.graph_norm(self.graph_mlp(g[window[:, -1]])) * self.graph_scale
            x = torch.cat([x, self.graph_token_dropout(g).unsqueeze(1)], dim=1)
            mask.append(keep)
        x = self.embedding_dropout(x + self.pos_emb[:, :x.shape[1]])
        x = self.encoder(x, src_key_padding_mask=torch.cat(mask, dim=1))
        return self.out(self.output_dropout(x[:, 0]))


class RevisitScorer(nn.Module):
    def __init__(self, num_features, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(num_features, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x, mask):
        return self.net(x).squeeze(-1).masked_fill(~mask, float("-inf"))


def revisit_bonus(log_prob, mask):
    k = mask.sum(1, keepdim=True).clamp(min=1).float()
    bonus = torch.log1p(torch.exp((log_prob + torch.log(k)).clamp(max=60.0)))
    return torch.where(mask, bonus, torch.zeros_like(bonus))


def final_scores(core_logits, cand, bonus, lam):
    extra = torch.zeros_like(core_logits).scatter_add_(1, cand, lam * bonus)
    return torch.log_softmax(core_logits, 1) + extra
