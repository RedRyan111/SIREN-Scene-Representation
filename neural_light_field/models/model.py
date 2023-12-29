import torch
from torch import nn


class NLFModel(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=256, out_dim=3):
        super().__init__()
        self.name = 'NLFModel'
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
'''
class NLFModel(torch.nn.Module):
    def __init__(self, num_pos_encoding_functions, num_dir_encoding_functions):
        super(NLFModel, self).__init__()
        filter_size = 200

        #inp_size = (3 + 3 * 2 * num_pos_encoding_functions)# + (3 + 3 * 2 * num_dir_encoding_functions)
        inp_size = 2

        self.layer1 = torch.nn.Linear(inp_size, filter_size)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        self.layer3 = torch.nn.Linear(filter_size, filter_size)
        self.layer4 = torch.nn.Linear(filter_size, filter_size)
        self.layer5 = torch.nn.Linear(filter_size, filter_size)

        self.rgb_layer = torch.nn.Linear(filter_size, 3)

        self.relu = torch.nn.functional.relu
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):

        y = self.relu(self.layer1(x))
        y = self.relu(self.layer2(y))
        y = self.relu(self.layer3(y))
        y = self.relu(self.layer4(y))
        y = self.relu(self.layer5(y))

        rgb = self.rgb_layer(y) #sigmoid REALLY hurts rgb layer here... why?

        return rgb
'''