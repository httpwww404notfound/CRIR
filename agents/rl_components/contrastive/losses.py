import torch
import torch.nn as nn
import math

class NCELoss(nn.Module):
    """
    Eq. (12): L_{NCE}
    """

    def __init__(self, temperature, device):
        super(NCELoss, self).__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1).to(self.device)

    # modified based on impl: https://github.com/ae-foster/pytorch-simclr/blob/dc9ac57a35aec5c7d7d5fe6dc070a975f493c1a5/critic.py#L5
    def forward(self, batch_sample_one, batch_sample_two, com_id=None):
        """
        parameter features:
        tensor.size([batch_size, seq_dim])
        """
        # sim11 = self.cossim(batch_sample_one.unsqueeze(-2), batch_sample_one.unsqueeze(-3)) / self.temperature
        # sim22 = self.cossim(batch_sample_two.unsqueeze(-2), batch_sample_two.unsqueeze(-3)) / self.temperature
        # sim12 = self.cossim(batch_sample_one.unsqueeze(-2), batch_sample_two.unsqueeze(-3)) / self.temperature
        sim11 = torch.matmul(batch_sample_one, batch_sample_one.T) / self.temperature
        sim22 = torch.matmul(batch_sample_two, batch_sample_two.T) / self.temperature
        sim12 = torch.matmul(batch_sample_one, batch_sample_two.T) / self.temperature
        d = sim12.shape[-1]
        # avoid contrast against positive intents
        if com_id is not None:
            com_id = com_id.contiguous().view(-1, 1)
            mask_11_22 = torch.eq(com_id, com_id.T).long().to(self.device)
            sim11[mask_11_22 == 1] = float("-inf")
            sim22[mask_11_22 == 1] = float("-inf")
            eye_metrix = torch.eye(d, dtype=torch.long).to(self.device)
            mask_11_22[eye_metrix == 1] = 0
            sim12[mask_11_22 == 1] = float("-inf")
        else:
            mask = torch.eye(d, dtype=torch.long).to(self.device)
            sim11[mask == 1] = float("-inf")
            sim22[mask == 1] = float("-inf")
            # sim22 = sim22.masked_fill_(mask, -np.inf)
            # sim11[..., range(d), range(d)] = float('-inf')
            # sim22[..., range(d), range(d)] = float('-inf')

        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss

class ICL_NCELoss(nn.Module):
    """
    EQ.2
    """
    def __init__(self, device):
        super(ICL_NCELoss, self).__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

    def forward(self, batch_items, real_len, weight):
        # batch_items: [batch_size pad_len dim]
        rank_first, rank_others = batch_items[:,0:1,:], batch_items[:,1:,:]
        rank_first = torch.transpose(rank_first, -1, -2)
        sim_with_first = torch.matmul(rank_others, rank_first)
        sim_with_first = torch.squeeze(sim_with_first)

        for i in range(sim_with_first.shape[0]):
            sim_with_first[i,real_len[i]-1:] = float("-inf")

        labels = torch.zeros(sim_with_first.shape[0], device=self.device, dtype=torch.long)

        weight = torch.tensor(weight, device=self.device)

        nce_losses = self.criterion(sim_with_first, labels)
        nce_loss = torch.mean(nce_losses * weight)

        return nce_loss
