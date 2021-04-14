import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel
from cnn import CNN
from multihead_attention import MultiheadAttention
from position_wise_ffnn import Position_wise_FFNN

MAXPOOLING = 0
MEANPOOLING = 1

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
Transfer Fine-tuned BERT Wrapper
"""


class BERTWrapper(nn.Module):
    def __init__(self, config):
        super(BERTWrapper, self).__init__()
        if config['model_path'] in ['bert-base-uncased', 'bert-large-uncased']:
            self.bert = BertModel.from_pretrained(config['bert_model']).cuda()
        else:
            self.bert = BertModel.from_pretrained(config['bert_model'],
                                                  state_dict=torch.load(config['model_path'])).cuda()

        self.bert_layer_indexes = config['bert_layers']
        self.bert_emb_dim = config['bert_emb_dim']
        self.init_bert_weights(self.bert)
        self.bert.eval()

    def encode(self, features):
        # Compute BERT embedding
        bert_embedding = self._get_bert_embedding(features)
        return bert_embedding.cpu().numpy()

    def _get_bert_embedding(self, features):
        embeddings = []
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).cuda()
        example_indices = torch.arange(input_ids.size(0), dtype=torch.long)

        with torch.no_grad():
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids=None, attention_mask=input_mask,
                                              output_all_encoded_layers=False)
        all_encoder_layers = all_encoder_layers

        for b, example_index in enumerate(example_indices):
            embeddings.append(all_encoder_layers[b])

        return torch.stack(embeddings)

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


"""
Model 1-b BERT Wrapper
"""


class BERT1BWrapper(nn.Module):
    def __init__(self, config):
        super(BERT1BWrapper, self).__init__()
        self.bert_layer_indexes = config['bert_layers']
        self.bert_emb_dim = config['bert_emb_dim']
        self.bert = BertModel.from_pretrained(config['bert_model']).cuda()
        self.ff = nn.Linear(self.bert_emb_dim * 2, self.bert_emb_dim)

        self.load_state_dict(torch.load(config['model_path']))
        self.init_bert_weights(self.bert)
        self.bert.eval()
        self.ff.eval()

    def encode_sentence(self, features):
        # Compute BERT embedding
        bert_embedding = self._get_bert_embedding(features)
        return bert_embedding

    def encode_phrase(self, sent_emb, sidx, n, m, pooling):
        if pooling == MAXPOOLING:
            s_phrase_emb = self._ff(sent_emb[sidx, 0, :],
                                    torch.max(sent_emb[sidx, n.start:n.end, :], dim=0)[0])
            t_phrase_emb = self._ff(sent_emb[sidx, 0, :],
                                    torch.max(sent_emb[sidx, m.start:m.end, :], dim=0)[0])
        else:  # mean pooling
            s_phrase_emb = self._ff(sent_emb[sidx, 0, :],
                                    torch.mean(sent_emb[sidx, n.start:n.end, :], dim=0))
            t_phrase_emb = self._ff(sent_emb[sidx, 0, :],
                                    torch.mean(sent_emb[sidx, m.start:m.end, :], dim=0))

        return s_phrase_emb.cpu().numpy(), t_phrase_emb.cpu().numpy()

    def _ff(self, cls_emb, phrase_emb):
        input = torch.cat((cls_emb, phrase_emb), dim=0)
        with torch.no_grad():
            embed = self.ff(input)
        return embed

    def _get_bert_embedding(self, features):
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).cuda()
        with torch.no_grad():
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids=None, attention_mask=input_mask,
                                              output_all_encoded_layers=False)
        all_encoder_layers = all_encoder_layers

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


"""
Model 1-c BERT Wrapper
"""


class BERT1CWrapper(nn.Module):
    def __init__(self, config):
        super(BERT1CWrapper, self).__init__()
        self.bert_layer_indexes = config['bert_layers']
        self.bert_emb_dim = config['bert_emb_dim']
        self.bert = BertModel.from_pretrained(config['bert_model']).cuda()
        self.fc = nn.Linear(self.bert_emb_dim * 3, 1)
        self.activ = nn.Sigmoid()

        self.load_state_dict(torch.load(config['model_path']))
        self.init_bert_weights(self.bert)
        self.bert.eval()
        self.fc.eval()

    def encode_sentence(self, features):
        # Compute BERT embedding
        bert_embedding = self._get_bert_embedding(features)
        return bert_embedding

    def similarity(self, sent_emb, sidx, n, m, pooling):
        cls_emb = sent_emb[sidx, 0, :]
        if pooling == MAXPOOLING:
            s_phrase_emb = torch.max(sent_emb[sidx, n.start:n.end, :], dim=0)[0]
            t_phrase_emb = torch.max(sent_emb[sidx, m.start:m.end, :], dim=0)[0]
        else:  # mean pooling
            s_phrase_emb = torch.mean(sent_emb[sidx, n.start:n.end, :], dim=0)
            t_phrase_emb = torch.mean(sent_emb[sidx, m.start:m.end, :], dim=0)

        with torch.no_grad():
            x = self.fc(torch.cat((cls_emb, s_phrase_emb, t_phrase_emb)))
            sim = self.activ(x)

        return sim.cpu().numpy()

    def _get_bert_embedding(self, features):
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).cuda()
        with torch.no_grad():
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids=None, attention_mask=input_mask,
                                              output_all_encoded_layers=False)
        all_encoder_layers = all_encoder_layers

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


"""
Model 1-d BERT Wrapper
"""
class BERT1DWrapper(nn.Module):
    def __init__(self, config):
        super(BERT1DWrapper, self).__init__()
        self.epsilon = 1e-08

        # BERT
        self.bert_layer_indexes = config['bert_layers']
        self.bert_emb_dim = config['bert_emb_dim']
        self.bert = BertModel.from_pretrained(config['bert_model']).cuda()

        # CNN
        self.cnn = CNN(config['map_embed_size'])
        self.max_seq_len = config['max_seq_len']

        # FFNN
        self.fc = nn.Linear(self.bert_emb_dim * 3, 1)
        self.activ = nn.Sigmoid()

        self.load_state_dict(torch.load(config['model_path']))
        self.init_bert_weights(self.bert)
        self.bert.eval()
        self.cnn.eval()
        self.fc.eval()
        self.activ.eval()

    def encode_sentence(self, features):
        with torch.no_grad():
            # Compute BERT embedding
            sent_emb = self._get_bert_embedding(features)

            # Compute word alignment map
            maps = torch.zeros((len(features), 1, self.max_seq_len, self.max_seq_len)).cuda()  # 1dim for channel
            for i in range(len(features)):
                s = torch.cat((sent_emb[i, 0:len(features[i].input_ids_a), :],
                               torch.zeros(
                                   (self.max_seq_len - len(features[i].input_ids_a), self.bert_emb_dim)).cuda()))
                t = torch.cat((sent_emb[i,
                               len(features[i].input_ids_a):len(features[i].input_ids_a) + len(features[i].input_ids_b),
                               :],
                               torch.zeros(
                                   (self.max_seq_len - len(features[i].input_ids_b), self.bert_emb_dim)).cuda()))
                maps[i, 0, :, :] = torch.tensordot(s, t.t(), dims=1) / torch.max(
                    torch.tensordot(torch.norm(s, dim=1, keepdim=True), torch.norm(t, dim=1, keepdim=True).t(), dims=1),
                    torch.full((self.max_seq_len, self.max_seq_len), self.epsilon).cuda())
            maps = self.cnn(maps)

        return sent_emb, maps

    def similarity(self, sent_emb, map_emb, sidx, n, m, pooling):
        cls_emb = map_emb[sidx, :]
        if pooling == MAXPOOLING:
            s_phrase_emb = torch.max(sent_emb[sidx, n.start:n.end, :], dim=0)[0]
            t_phrase_emb = torch.max(sent_emb[sidx, m.start:m.end, :], dim=0)[0]
        else:  # mean pooling
            s_phrase_emb = torch.mean(sent_emb[sidx, n.start:n.end, :], dim=0)
            t_phrase_emb = torch.mean(sent_emb[sidx, m.start:m.end, :], dim=0)

        with torch.no_grad():
            x = self.fc(torch.cat((cls_emb, s_phrase_emb, t_phrase_emb)))
            predict = self.activ(x)

        return predict.cpu().numpy()

    def _get_bert_embedding(self, features):
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).cuda()

        all_encoder_layers, _ = self.bert(input_ids, token_type_ids=None, attention_mask=input_mask,
                                          output_all_encoded_layers=False)
        all_encoder_layers = all_encoder_layers

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


"""
Model 1-e BERT Wrapper
"""


class BERT1EWrapper(nn.Module):
    def __init__(self, config):
        super(BERT1EWrapper, self).__init__()
        # BERT
        self.bert_layer_indexes = config['bert_layers']
        self.bert_emb_dim = config['bert_emb_dim']
        self.bert = BertModel.from_pretrained(config['bert_model']).cuda()

        # Multihead attention
        self.num_heads = config['num_heads']
        self.atten = MultiheadAttention(self.bert_emb_dim, self.num_heads)
        self.layernorm1 = nn.LayerNorm(self.bert_emb_dim)
        self.layernorm2 = nn.LayerNorm(self.bert_emb_dim)

        # FFNN
        self.ffnn = Position_wise_FFNN(self.bert_emb_dim)

        self.load_state_dict(torch.load(config['model_path']))
        self.init_bert_weights(self.bert)
        self.atten.eval()
        self.layernorm1.eval()
        self.layernorm2.eval()
        self.bert.eval()
        self.ffnn.eval()

    def _get_bert_embedding(self, features):
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).cuda()

        with torch.no_grad():
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids=None, attention_mask=input_mask,
                                              output_all_encoded_layers=False)
        all_encoder_layers = all_encoder_layers

        return all_encoder_layers

    def encode_sentence(self, features):
        # Compute BERT embedding
        sent_emb = self._get_bert_embedding(features)
        return sent_emb

    def similarity(self, sent_emb, sidx, n, m, pooling):
        cls_emb = sent_emb[sidx, 0, :].reshape((1, 1, self.bert_emb_dim))
        if pooling == MAXPOOLING:
            s_phrase_emb = torch.max(sent_emb[sidx, n.start:n.end, :], dim=0)[0].reshape((1, 1, self.bert_emb_dim))
            t_phrase_emb = torch.max(sent_emb[sidx, m.start:m.end, :], dim=0)[0].reshape((1, 1, self.bert_emb_dim))
        else:  # mean pooling
            s_phrase_emb = torch.mean(sent_emb[sidx, n.start:n.end, :], dim=0).reshape((1, 1, self.bert_emb_dim))
            t_phrase_emb = torch.mean(sent_emb[sidx, m.start:m.end, :], dim=0).reshape((1, 1, self.bert_emb_dim))

        with torch.no_grad():
            mem = torch.cat((cls_emb, s_phrase_emb), dim=0)
            attn_output, attn_weights = self.atten(t_phrase_emb, mem, mem)
            atten = (t_phrase_emb + attn_output).squeeze(0)
            atten=self.layernorm1(atten)
            fc_atten = self.ffnn(atten)
            atten = atten + fc_atten
            atten=self.layernorm2(atten)

        return s_phrase_emb.squeeze().cpu().numpy(), atten.squeeze().cpu().numpy()

    def _get_bert_embedding(self, features):
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).cuda()
        with torch.no_grad():
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids=None, attention_mask=input_mask,
                                              output_all_encoded_layers=False)
        all_encoder_layers = all_encoder_layers

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


"""
Model 1-f BERT Wrapper
"""
class BERT1FWrapper(nn.Module):
    def __init__(self, config):
        super(BERT1FWrapper, self).__init__()
        self.epsilon = 1e-08

        # BERT
        self.bert_layer_indexes = config['bert_layers']
        self.bert_emb_dim = config['bert_emb_dim']
        self.bert = BertModel.from_pretrained(config['bert_model']).cuda()

        # CNN
        self.cnn = CNN(config['map_embed_size'])
        self.max_seq_len = config['max_seq_len']

        # Multihead attention
        self.num_heads = config['num_heads']
        self.atten = MultiheadAttention(self.bert_emb_dim, self.num_heads)
        self.layernorm1 = nn.LayerNorm(self.bert_emb_dim)
        self.layernorm2 = nn.LayerNorm(self.bert_emb_dim)

        # FFNN
        self.ffnn = Position_wise_FFNN(self.bert_emb_dim)

        self.load_state_dict(torch.load(config['model_path']))
        self.init_bert_weights(self.bert)
        self.cnn.eval()
        self.atten.eval()
        self.layernorm1.eval()
        self.layernorm2.eval()
        self.bert.eval()
        self.ffnn.eval()

    def _get_bert_embedding(self, features):
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).cuda()

        all_encoder_layers, _ = self.bert(input_ids, token_type_ids=None, attention_mask=input_mask,
                                          output_all_encoded_layers=False)
        all_encoder_layers = all_encoder_layers

        return all_encoder_layers

    def encode_sentence(self, features):
        with torch.no_grad():
            # Compute BERT embedding
            sent_emb = self._get_bert_embedding(features)

            # Compute word alignment map
            maps = torch.zeros((len(features), 1, self.max_seq_len, self.max_seq_len)).cuda()  # 1dim for channel
            for i in range(len(features)):
                s = torch.cat((sent_emb[i, 0:len(features[i].input_ids_a), :],
                               torch.zeros(
                                   (self.max_seq_len - len(features[i].input_ids_a), self.bert_emb_dim)).cuda()))
                t = torch.cat((sent_emb[i,
                               len(features[i].input_ids_a):len(features[i].input_ids_a) + len(features[i].input_ids_b),
                               :],
                               torch.zeros(
                                   (self.max_seq_len - len(features[i].input_ids_b), self.bert_emb_dim)).cuda()))
                maps[i, 0, :, :] = torch.tensordot(s, t.t(), dims=1) / torch.max(
                    torch.tensordot(torch.norm(s, dim=1, keepdim=True), torch.norm(t, dim=1, keepdim=True).t(), dims=1),
                    torch.full((self.max_seq_len, self.max_seq_len), self.epsilon).cuda())
            maps = self.cnn(maps)

        return sent_emb, maps

    def similarity(self, sent_emb, map_emb, sidx, n, m, pooling):
        cls_emb = map_emb[sidx, :].reshape((1, 1, self.bert_emb_dim))
        if pooling == MAXPOOLING:
            s_phrase_emb = torch.max(sent_emb[sidx, n.start:n.end, :], dim=0)[0].reshape((1, 1, self.bert_emb_dim))
            t_phrase_emb = torch.max(sent_emb[sidx, m.start:m.end, :], dim=0)[0].reshape((1, 1, self.bert_emb_dim))
        else:  # mean pooling
            s_phrase_emb = torch.mean(sent_emb[sidx, n.start:n.end, :], dim=0).reshape((1, 1, self.bert_emb_dim))
            t_phrase_emb = torch.mean(sent_emb[sidx, m.start:m.end, :], dim=0).reshape((1, 1, self.bert_emb_dim))

        with torch.no_grad():
            mem = torch.cat((cls_emb, s_phrase_emb), dim=0)
            attn_output, attn_weights = self.atten(t_phrase_emb, mem, mem)
            atten = (t_phrase_emb + attn_output).squeeze(0)
            atten=self.layernorm1(atten)
            fc_atten = self.ffnn(atten)
            atten = atten + fc_atten
            atten=self.layernorm2(atten)

        return s_phrase_emb.squeeze().cpu().numpy(), atten.squeeze().cpu().numpy()

    def _get_bert_embedding(self, features):
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).cuda()
        with torch.no_grad():
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids=None, attention_mask=input_mask,
                                              output_all_encoded_layers=False)
        all_encoder_layers = all_encoder_layers

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
