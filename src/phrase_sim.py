from abc import ABCMeta, abstractmethod
import numpy as np


class PhraseSim(metaclass=ABCMeta):
    MAXPOOLING = 0
    MEANPOOLING = 1
    POOLING = [MAXPOOLING, MEANPOOLING]
    MAXCOST= 10**12

    @classmethod
    @abstractmethod
    def align_score(self, n, m, sent_idx):
        pass

    @classmethod
    @abstractmethod
    def null_align_score(self, n):
        pass

    def _scaled_cossim(self, n_vec, m_vec):
        if np.all(n_vec == 0) or np.all(m_vec == 0):
            cos_sim = -1
        else:
            cos_sim = np.dot(n_vec, m_vec) / (np.linalg.norm(n_vec) * np.linalg.norm(m_vec))

        # scaling
        cos_sim = self._scaling(cos_sim, -1, 1)

        return cos_sim

    def _scaling(self, val, min, max):
        # scaling to range from 0 to max_cost
        scaled_val = int(np.abs(max - val) * self.MAXCOST / (max - min))

        return scaled_val

    def set_pooling(self, pooling):
        if pooling == 'max':
            self.pooling_method = self.MAXPOOLING
        else:
            self.pooling_method = self.MEANPOOLING

    def get_pooling(self):
        return 'MeanPool' if self.pooling_method == self.MEANPOOLING else 'MaxPool'

    def set_null_thresh(self, thresh, min, max):
        self.NULL_SCORE = int(np.abs(max - thresh) * self.MAXCOST / (max - min))

    def get_model_name(self):
        return self.model_name.replace('.pkl', '')

    def set_seed(self, seed):
        self.seed = seed


class PhraseSimSample(PhraseSim):
    @classmethod
    def align_score(self, n, m, sent_idx):
        if n.txt == m.txt:
            return 1
        else:
            return 10

    def null_align_score(self, n):
        return 100
