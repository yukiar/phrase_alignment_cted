import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TripletMarginLoss, MarginRankingLoss
from pytorch_pretrained_bert.modeling import BertModel
from functools import reduce
from multihead_attention import MultiheadAttention
from position_wise_ffnn import Position_wise_FFNN

"""
Initialize BERT
"""
try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")


    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

"""
Idea 1-c
"""


class BERT1E(nn.Module):
    def __init__(self, config):
        super(BERT1E, self).__init__()
        self.device = config['device']

        # BERT
        self.bert_layer_indexes = config['bert_layers']
        self.bert_emb_dim = config['bert_emb_dim']
        self.pooling = config['pooling']
        self.bert = BertModel.from_pretrained(config['bert_model']).cuda()
        self.init_bert_weights(self.bert)
        self.dropout = nn.Dropout(config['dropout'])
        self.ft_bert = config['ft_bert']

        # Multihead attention
        self.num_heads = config['num_heads']
        self.atten = MultiheadAttention(self.bert_emb_dim, self.num_heads)
        self.layernorm1 = nn.LayerNorm(self.bert_emb_dim)
        self.layernorm2 = nn.LayerNorm(self.bert_emb_dim)
        self.res_dropout = nn.Dropout(config['res_dropout'])

        # FFNN
        self.ffnn = Position_wise_FFNN(self.bert_emb_dim)

        self.loss_name = config['lossfnc']
        self.loss_func = eval(self.loss_name)(margin=config['margin'])

    def forward(self, features, s_span_set, t_span_set, labels):
        # Compute BERT embedding
        sent_emb = self._get_bert_embedding(features)

        pair_num = int(len(reduce(lambda a, b: a + b, labels)) / 2)
        cls_emb = torch.empty((pair_num, self.bert_emb_dim), requires_grad=True).to(self.device)
        s_phrase_emb = torch.empty((pair_num, self.bert_emb_dim), requires_grad=True).to(self.device)
        p_phrase_emb = torch.empty((pair_num, self.bert_emb_dim), requires_grad=True).to(self.device)
        n_phrase_emb = torch.empty((pair_num, self.bert_emb_dim), requires_grad=True).to(self.device)
        label_tensor = torch.empty((pair_num, 1)).to(self.device)

        idx = 0
        for sidx in range(len(features)):
            for pidx in range(0, len(t_span_set[sidx]), 2):
                s_span = s_span_set[sidx][pidx]
                p_span = t_span_set[sidx][pidx]
                n_span = t_span_set[sidx][pidx + 1]

                cls_emb[idx, :] = sent_emb[sidx, 0, :]
                label_tensor[idx] = labels[sidx][pidx]
                if self.pooling == 'max':
                    s_phrase_emb[idx, :] = torch.max(sent_emb[sidx, s_span[0]:s_span[1], :], dim=0)[0]
                    p_phrase_emb[idx, :] = torch.max(sent_emb[sidx, p_span[0]:p_span[1], :], dim=0)[0]
                    n_phrase_emb[idx, :] = torch.max(sent_emb[sidx, n_span[0]:n_span[1], :], dim=0)[0]
                else:  # mean pooling
                    s_phrase_emb[idx, :] = torch.mean(sent_emb[sidx, s_span[0]:s_span[1], :], dim=0)
                    p_phrase_emb[idx, :] = torch.mean(sent_emb[sidx, p_span[0]:p_span[1], :], dim=0)
                    n_phrase_emb[idx, :] = torch.mean(sent_emb[sidx, n_span[0]:n_span[1], :], dim=0)
                idx += 1

        mem = torch.cat((cls_emb.unsqueeze(0), s_phrase_emb.unsqueeze(0)), dim=0)
        attn_p_output, attn_p_weights = self.atten(p_phrase_emb.unsqueeze(0), mem, mem)
        attn_p_output = self.res_dropout(attn_p_output)
        atten_p = (p_phrase_emb + attn_p_output).squeeze()
        atten_p = self.layernorm1(atten_p)
        fc_atten_p = self.ffnn(atten_p)
        fc_atten_p = self.res_dropout(fc_atten_p)
        atten_p = atten_p + fc_atten_p
        atten_p = self.layernorm2(atten_p)

        attn_n_output, attn_n_weights = self.atten(n_phrase_emb.unsqueeze(0), mem, mem)
        attn_n_output = self.res_dropout(attn_n_output)
        atten_n = (n_phrase_emb + attn_n_output).squeeze()
        atten_n = self.layernorm1(atten_n)
        fc_atten_n = self.ffnn(atten_n)
        fc_atten_n = self.res_dropout(fc_atten_n)
        atten_n = atten_n + fc_atten_n
        atten_n = self.layernorm2(atten_n)

        loss = self.loss_func(s_phrase_emb, atten_p, atten_n)

        return loss

    def _get_bert_embedding(self, features):
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).cuda()

        if self.ft_bert:
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids=None, attention_mask=input_mask,
                                              output_all_encoded_layers=False)
        else:
            self.bert.eval()
            with torch.no_grad():
                all_encoder_layers, _ = self.bert(input_ids, token_type_ids=None, attention_mask=input_mask,
                                                  output_all_encoded_layers=False)
        all_encoder_layers = self.dropout(all_encoder_layers)

        return all_encoder_layers

    """
    BERT initializer
    """

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
