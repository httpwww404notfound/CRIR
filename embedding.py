from torch import nn
import torch

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class ItemsEmbeddings(nn.Module):
    """Construct the embeddings from item, position.
    """

    def __init__(self, item_size, hidden_size, max_seq_length, hidden_dropout_prob):
        super(ItemsEmbeddings, self).__init__()
        self.item_embedding = nn.Embedding(item_size, hidden_size, padding_idx=0) 
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        # self.dropout = nn.Dropout(hidden_dropout_prob)


    def forward(self, input_ids, use_position=False):
        items_embeddings = self.item_embedding(input_ids)
        embeddings = items_embeddings
        if use_position:
            seq_length = input_ids.size(-1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

            position_embeddings = self.position_embeddings(position_ids)
            embeddings = items_embeddings + position_embeddings
        # 修改属性
        embeddings = self.LayerNorm(embeddings)
        # embeddings = self.dropout(embeddings)
        return embeddings

class UserEmbeddings(nn.Module):
    """
    Construct embeddings from user ids
    """
    def __init__(self, users_num, hidden_size):
        super(UserEmbeddings, self).__init__()

        self.user_embedding = nn.Embedding(users_num, hidden_size)

    def forward(self, user_id):

        embedding = self.user_embedding(user_id)
        return embedding

class vtbUserEncoder(nn.Module):

    def __init__(self, feature_num, hidden_dim):
        super(vtbUserEncoder, self).__init__()
        self.user_encoder = nn.Linear(feature_num, hidden_dim)

    def forward(self, features):
        embedding = self.user_encoder(features)
        return embedding

class statePreEncoder(nn.Module):

    def __init__(self, feature_num, hidden_dim):
        super(statePreEncoder, self).__init__()

        self.item_encoder = nn.Linear(feature_num, hidden_dim)
        self.LayerNorm = nn.LayerNorm(hidden_dim, eps=1e-12)

    def forward(self, feature_seq):
        embedding = self.item_encoder(feature_seq)
        embedding = self.LayerNorm(embedding)
        return embedding
