import torch
import torch.nn as nn
from torch.nn import CosineEmbeddingLoss, TripletMarginLoss
from pytorch_pretrained_bert.modeling import BertModel
from functools import reduce

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
Fine-tuning BERT
"""


class BERTTrainer(nn.Module):
    def __init__(self, config):
        super(BERTTrainer, self).__init__()
        self.bert = BertModel.from_pretrained(config['bert_model']).cuda()
        self.bert_layer_indexes = config['bert_layers']
        self.bert_emb_dim = config['bert_emb_dim']
        self.dropout = nn.Dropout(config['dropout'])
        self.init_bert_weights(self.bert)
        self.margin = config['margin']
        self.pooling = config['pooling']
        self.loss_type = config['lossfnc']
        self.loss_func = eval(self.loss_type)(margin=self.margin)
        self.device = config['device']

    def forward(self, features, s_span_set, t_span_set, labels):
        # Compute BERT embedding
        sent_emb = self._get_bert_embedding(features)
        idx = 0
        if self.loss_type == 'CosineEmbeddingLoss':
            label_tensor = torch.Tensor(reduce(lambda a, b: a + b, labels)).unsqueeze(dim=1).to(self.device)
            s_phrase_emb = torch.empty((label_tensor.shape[0], self.bert_emb_dim), requires_grad=True).to(self.device)
            t_phrase_emb = torch.empty((label_tensor.shape[0], self.bert_emb_dim), requires_grad=True).to(self.device)

            for sidx in range(len(features)):
                for s_span, t_span in zip(s_span_set[sidx], t_span_set[sidx]):
                    if self.pooling == 'max':
                        s_phrase_emb[idx, :] = torch.max(sent_emb[sidx, s_span[0]:s_span[1], :], dim=0)[0]
                        t_phrase_emb[idx, :] = torch.max(sent_emb[sidx, t_span[0]:t_span[1], :], dim=0)[0]
                    else:  # mean pooling
                        s_phrase_emb[idx, :] = torch.mean(sent_emb[sidx, s_span[0]:s_span[1], :], dim=0)
                        t_phrase_emb[idx, :] = torch.mean(sent_emb[sidx, t_span[0]:t_span[1], :], dim=0)
                    idx += 1

            loss = self.loss_func(s_phrase_emb, t_phrase_emb, label_tensor)

        elif self.loss_type == 'TripletMarginLoss':
            pair_num = int(len(reduce(lambda a, b: a + b, labels)) / 2)
            s_phrase_emb = torch.empty((pair_num, self.bert_emb_dim), requires_grad=True).to(self.device)
            p_phrase_emb = torch.empty((pair_num, self.bert_emb_dim), requires_grad=True).to(self.device)
            n_phrase_emb = torch.empty((pair_num, self.bert_emb_dim), requires_grad=True).to(self.device)

            for sidx in range(len(features)):
                for pidx in range(0, len(t_span_set[sidx]), 2):
                    s_span = s_span_set[sidx][pidx]
                    p_span = t_span_set[sidx][pidx]
                    n_span = t_span_set[sidx][pidx + 1]

                    if self.pooling == 'max':
                        s_phrase_emb[idx, :] = torch.max(sent_emb[sidx, s_span[0]:s_span[1], :], dim=0)[0]
                        p_phrase_emb[idx, :] = torch.max(sent_emb[sidx, p_span[0]:p_span[1], :], dim=0)[0]
                        n_phrase_emb[idx, :] = torch.max(sent_emb[sidx, n_span[0]:n_span[1], :], dim=0)[0]
                    else:  # mean pooling
                        s_phrase_emb[idx, :] = torch.mean(sent_emb[sidx, s_span[0]:s_span[1], :], dim=0)
                        p_phrase_emb[idx, :] = torch.mean(sent_emb[sidx, p_span[0]:p_span[1], :], dim=0)
                        n_phrase_emb[idx, :] = torch.mean(sent_emb[sidx, n_span[0]:n_span[1], :], dim=0)
                    idx += 1

            loss = self.loss_func(s_phrase_emb, p_phrase_emb, n_phrase_emb)

        return loss

    def _get_bert_embedding(self, features):
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).cuda()
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
