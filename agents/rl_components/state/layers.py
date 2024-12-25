from torch import nn
import torch

class AttentionSequencePoolingLayer(nn.Module):
    def __init__(self, max_len=50, embedding_dim=4):
        super(AttentionSequencePoolingLayer, self).__init__()

        # TODO: DICE acitivation function
        # TODO: attention weight normalization
        self.local_att = LocalActivationUnit(bias=[True, True], embedding_dim=embedding_dim,batch_norm=False)
        # self.fc = nn.Sequential(nn.Linear(max_len*embedding_dim, embedding_dim), nn.PReLU())

    def forward(self, query_user, user_behavior):
        # query user          : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        # output              : size -> batch_size * 1 * embedding_size

        attention_score = self.local_att(query_user, user_behavior) # B * T * 1
        attention_score = torch.transpose(attention_score, 1, 2)
        # output = attention_score * user_behavior
        # output = torch.flatten(output, start_dim=-2, end_dim=-1)
        # output = self.fc(output).unsqueeze(dim=1)

        # score_behavior = torch.cat([attention_score, user_behavior], dim=-2)
        output = torch.matmul(attention_score, user_behavior) # batch_size * 1 * embedding_size

        return output, torch.squeeze(attention_score, dim=-2)


class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_size=[80, 40], bias=[True, True], embedding_dim=4, batch_norm=False):
        super(LocalActivationUnit, self).__init__()
        self.fc1 = FullyConnectedLayer(input_size=4*embedding_dim,
                                       hidden_size=hidden_size,
                                       bias=bias,
                                       batch_norm=batch_norm,
                                       activation='dice',
                                       dice_dim=3)

        self.fc2 = FullyConnectedLayer(input_size=hidden_size[-1],
                                       hidden_size=[1],
                                       bias=[True],
                                       batch_norm=batch_norm,
                                       activation='dice',
                                       dice_dim=3)
        # TODO: fc_2 initialization

    def forward(self, query, user_behavior):
        # query user          : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size

        user_behavior_len = user_behavior.size(1)
        queries = torch.cat([query for _ in range(user_behavior_len)], dim=1)

        attention_input = torch.cat([queries, user_behavior, queries-user_behavior, queries*user_behavior], dim=-1)
        attention_output = self.fc1(attention_input)
        attention_output = self.fc2(attention_output)

        return attention_output


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, hidden_size, bias, batch_norm=True, dropout_rate=0.5, activation='relu',
                 sigmoid=False, dice_dim=2):
        super(FullyConnectedLayer, self).__init__()
        assert len(hidden_size) >= 1 and len(bias) >= 1
        assert len(bias) == len(hidden_size)
        self.sigmoid = sigmoid

        layers = []
        layers.append(nn.Linear(input_size, hidden_size[0], bias=bias[0]))

        for i, h in enumerate(hidden_size[:-1]):
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size[i]))

            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'dice':
                assert dice_dim
                layers.append(Dice(hidden_size[i], dim=dice_dim))
            elif activation.lower() == 'prelu':
                layers.append(nn.PReLU())
            else:
                raise NotImplementedError

            layers.append(nn.Dropout(p=dropout_rate))
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1], bias=bias[i]))

        self.fc = nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = nn.Sigmoid()

        # weight initialization xavier_normal (or glorot_normal in keras, tf)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.sigmoid else self.fc(x)


class Dice(nn.Module):
    def __init__(self, num_features, dim=2):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3
        self.bn = nn.BatchNorm1d(num_features, eps=1e-9)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        if self.dim == 3:
            self.alpha = torch.zeros((num_features, 1)).cuda()
        elif self.dim == 2:
            self.alpha = torch.zeros((num_features,)).cuda()

    def forward(self, x):
        out = None
        if self.dim == 3:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)

        elif self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x

        return out

