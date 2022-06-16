import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.efficientnet import efficientnet_b0

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        ## 1. MatMul
        attn = torch.bmm(q, k.transpose(1, 2))
        ## 2. Scale
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)

        ## 3. Softmax
        attn = self.softmax(attn)

        ## 4. Mask (optional)
        attn = self.dropout(attn)

        ## 5. Matmul
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        ## query, key, value에 대한 weight metric
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        ## query, key, value에 대한 weight metric 초기화
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        ## similarity function = dot product
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        ## 1. Linear query, key, value
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n_head*sz_b) x len_q x d_k
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n_head*sz_b) x len_k x d_k
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n_head*sz_b) x len_v x d_v

        ## 2. Self Attention
        output, _, _ = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # sz_b x len_q x (n_head*d_v)
        ## 3. Concat and 4. Linear
        output = self.fc(output)

        # ## 5. Mask
        # output = self.dropout(output)

        ## 6. Layer normalize
        output = self.layer_norm(output + residual)

        return output

class TRAINNET(nn.Module):
    def __init__(self, mode=None):
        super().__init__()
        self.mode = mode
        self.backbone = efficientnet_b0(pretrained=True)
        self.num_features = 1280
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_features, 200, bias=False)
        self.slf_attn = MultiHeadAttention(8, self.num_features, self.num_features, self.num_features, dropout=0.5)
        self.temperature = 16
        self.base_class = 100
        self.novel_class = 100
        self.epochs = 100
    
    def encode(self, input):
        output = self.backbone(input)
        output = F.adaptive_avg_pool2d(output, 1)
        output = output.squeeze(-1).squeeze(-1)

        return output
    
    def _forward(self, support, query):
        emb_dim = support.size(-1)
        # get mean of the support
        proto = support.mean(dim=1)
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = query.shape[1]*query.shape[2]#num of query*way

        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        query = query.view(-1, emb_dim).unsqueeze(1)

        proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
        proto = proto.view(num_batch*num_query, num_proto, emb_dim)

        combined = torch.cat([proto, query], 1) # Nk x (N + 1) x d, batch_size = NK
        combined = self.slf_attn(combined, combined, combined)
        # compute distance for all batches
        proto, query = combined.split(num_proto, 1)

        logits = F.cosine_similarity(query,proto,dim=-1)
        logits = logits * self.temperature

        return logits
    
    def update_fc(self, dataloader, class_list):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

        self.update_fc_avg(data, label, class_list)
    
    def update_fc_avg(self, data, label, class_list):
        new_fc=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def forward(self, input):
        if self.mode == 'encoder':
            output = self.encode(input)

            return output
        else:
            support_idx, query_idx = input
            logits = self._forward(support_idx, query_idx)

            return logits