import torch
import torch.nn as nn
import torch.nn.functional as F
from model.efficientnet import *

class PRETRAINNET(nn.Module):
    def __init__(self, mode=None):
        super().__init__()

        self.mode = mode
        self.encoder = efficientnet_b0(True)
        self.num_features = 1280
        self.num_classess = 200
        self.temperature = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_features, self.num_classess, bias=False)

    def forward_metric(self, x):
        x = self.encode(x)
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.temperature * x

        return x

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def forward(self, input):
        input = self.forward_metric(input)
        return input

    def update_fc(self,dataloader,class_list):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

        self.update_fc_avg(data, label, class_list)

    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc