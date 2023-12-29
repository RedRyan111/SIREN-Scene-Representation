import torch


class NLFModel(torch.nn.Module):
    def __init__(self, num_pos_encoding_functions, num_dir_encoding_functions):
        super(NLFModel, self).__init__()
        filter_size = 20

        inp_size = (3 + 3 * 2 * num_pos_encoding_functions) + (3 + 3 * 2 * num_dir_encoding_functions)

        self.layer1 = torch.nn.Linear(inp_size, filter_size)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        self.layer3 = torch.nn.Linear(filter_size, filter_size)
        self.layer4 = torch.nn.Linear(filter_size, filter_size)
        self.layer5 = torch.nn.Linear(filter_size, filter_size)

        self.rgb_layer = torch.nn.Linear(filter_size, 3)

        self.relu = torch.nn.functional.relu
        self.sig = torch.nn.Sigmoid()

    def forward(self, x, d):
        y = torch.concatenate((x, d), dim=1)

        y = self.relu(self.layer1(y))
        y = self.relu(self.layer2(y))
        y = self.relu(self.layer3(y))
        y = self.relu(self.layer4(y))
        y = self.relu(self.layer5(y))

        rgb = self.sig(self.rgb_layer(y))

        return rgb
