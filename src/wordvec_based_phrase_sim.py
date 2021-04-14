from phrase_sim import PhraseSim
from gensim.models.fasttext import FastText as FT_gensim
import numpy as np


class wordvec_sim(PhraseSim):
    MAXPOOLING = 0
    MEANPOOLING = 1
    model_name = 'FastText'

    def __init__(self, pooling, path_to_fasttext_model):
        ############ Hyper Paramer #############
        self.NULL_SCORE = np.abs(1 - 0.5) * 500
        self.set_pooling(pooling)
        ########################################

        self.model = FT_gensim.load_fasttext_format(path_to_fasttext_model)

    def align_score(self, n, m, sent_idx):
        n_vec = self._get_vec(n)
        m_vec = self._get_vec(m)
        cos_sim = self._scaled_cossim(n_vec, m_vec)

        return cos_sim

    def null_align_score(self, n):
        return self.NULL_SCORE

    def _get_vec(self, node):
        vecs = np.zeros((len(node.tokens), self.model.vector_size))
        for i, w in enumerate(node.tokens):
            vecs[i] = self.model[w]

        if self.pooling_method == self.MAXPOOLING:
            pooled_vec = vecs.max(axis=0)
        else:
            pooled_vec = vecs.mean(axis=0)

        return pooled_vec
