from phrase_sim import PhraseSim
from bertwrapper import BERTWrapper
from bertutil import MyBertTokenizer, PhraseAlignDualDataset, prepare_input_dual
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import numpy as np
import torch
import os
import random
from tqdm import tqdm


class BERT_sim(PhraseSim):
    batch_size = 32
    max_seq_length = 128
    bert_model = 'bert-base-uncased'

    def __init__(self, model_dir, model_name, pooling, seed):
        ############ Hyper Paramer #############
        self.NULL_SCORE = np.abs(1 - 0.5) * 500
        self.set_pooling(pooling)
        self.set_seed(seed)
        ########################################
        self.model_name = model_name
        if model_name == 'BERT':
            self.model_path = self.bert_model
        else:
            self.model_path = model_dir + self.model_name

        # set up gpu environment
        # For debugging... set gpu device
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        self.device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
        # set seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # BERT config
        do_lower_case = True if 'uncased' in self.bert_model else False
        self.bert_emb_dim = 768 if 'base' in self.bert_model else 1024

        # Model config
        config_model = {
            'bert_model': self.bert_model,
            'bert_emb_dim': self.bert_emb_dim,
            'bert_layers': [int(x) for x in "-1,-2,-3,-4".split(",")],
            'bsize': self.batch_size,
            'model_path': self.model_path
        }

        # Prepare model
        self.model = BERTWrapper(config_model)
        self.model.to(self.device)
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Bert tokenizer
        self.tokenizer = MyBertTokenizer.from_pretrained(self.bert_model, do_lower_case=do_lower_case)

    def encode(self, s_tokens, t_tokens, s_trees, t_trees):
        features = prepare_input_dual(s_tokens, t_tokens, s_trees, t_trees, self.tokenizer, self.max_seq_length)
        input = PhraseAlignDualDataset(features)
        dataloader_idx = DataLoader(
            dataset=TensorDataset(torch.tensor([i for i in range(len(input))], dtype=torch.int)),
            sampler=SequentialSampler(input), batch_size=self.batch_size)

        # Encode sentences
        self.sent_vectors = np.zeros((len(s_trees), self.max_seq_length, self.bert_emb_dim))
        in_idx = 0
        for step, batch_idx in enumerate(tqdm(dataloader_idx, desc="Iteration")):
            batch = input.get_batch(batch_idx)
            bert_embeddings = self.model.encode(batch['features'])
            self.sent_vectors[in_idx:in_idx + len(batch['features'])] = bert_embeddings
            in_idx += len(batch['features'])

    def align_score(self, n, m, sent_idx):
        sent_vec = self.sent_vectors[sent_idx]

        if self.pooling_method == self.MEANPOOLING:
            # Mean pooling
            s_emb = np.mean(sent_vec[n.start:n.end, :], axis=0)
            t_emb = np.mean(sent_vec[m.start:m.end, :], axis=0)
        elif self.pooling_method == self.MAXPOOLING:
            # Max pooling
            s_emb = np.max(sent_vec[n.start:n.end, :], axis=0)
            t_emb = np.max(sent_vec[m.start:m.end, :], axis=0)

        # Cosine similarity
        cos_sim = self._scaled_cossim(s_emb, t_emb)

        return cos_sim

    def null_align_score(self, n):
        return self.NULL_SCORE
