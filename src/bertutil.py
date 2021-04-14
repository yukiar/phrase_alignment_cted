from tqdm import tqdm
from torch.utils.data import Dataset
from pytorch_pretrained_bert.tokenization import BertTokenizer
import unicodedata, random
import numpy as np


class MyBertTokenizer(BertTokenizer):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def tokenize(self, text):
        ctext = []
        for w in text:
            cw = self._clean_text(w)
            if len(cw) == 0:
                return None, None
            else:
                ctext.append(cw)

        split_tokens = []
        subword_map = {}
        word_idx = 0
        char_idx = 0
        subw_cnt = 1  # Skip the special label "[CLS]"
        for token in self.basic_tokenizer.tokenize(" ".join(ctext)):
            if char_idx == 0:
                subword_map[word_idx + 1] = subw_cnt  # Skip the special label "[CLS]"

            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
                subw_cnt += 1

            if len(ctext[word_idx]) > char_idx + len(token):
                char_idx += len(token)
            else:
                word_idx += 1
                char_idx = 0

        # For [SEP]
        subword_map[len(ctext) + 1] = len(split_tokens) + 1

        return split_tokens, subword_map

    """
            Copy fomr the original BertTokenizer
    """

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_whitespace(self, char):
        """Checks whether `chars` is a whitespace character."""
        # \t, \n, and \r are technically contorl characters but we treat them
        # as whitespace since they are generally considered as such.
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

    def _is_control(self, char):
        """Checks whether `chars` is a control character."""
        # These are technically control characters but we count them as whitespace
        # characters.
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


class InputExampleDual(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeaturesDual(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_ids_a, input_ids_b, input_mask):
        self.input_ids = input_ids
        self.input_ids_a = input_ids_a
        self.input_ids_b = input_ids_b
        self.input_mask = input_mask


class PhraseAlignDualDataset(Dataset):
    """Phrase Alignment dataset."""

    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = {'features': self.features[idx]}
        return sample

    def get_batch(self, ids):
        batch = {}
        batch['features'] = []
        for i in ids[0][:]:
            item = self.__getitem__(i)
            batch['features'].append(item['features'])
        return batch


class PhraseAlignmentDataset(Dataset):
    """Phrase Alignment dataset."""

    def __init__(self, features, s_trees, t_trees, align_labels, loss_type):
        self.features = features
        self.s_trees = s_trees
        self.t_trees = t_trees
        self.align_labels = align_labels

        # Generate phrase pairs
        self.s_span_set, self.t_span_set, self.label_set = [], [], []
        for pairidx in range(len(features)):
            s_id_span_dic = self._create_phrase_dic(s_trees[pairidx])
            t_id_span_dic = self._create_phrase_dic(t_trees[pairidx])
            s_spans, t_spans, labels = [], [], []
            s_to_t, t_to_s = {}, {}

            if loss_type == 'CosineEmbeddingLoss':
                # positive examples
                for (s_idx, t_idx) in align_labels[pairidx]:
                    if s_idx != '-1' and t_idx != '-1':
                        s_spans.append(s_id_span_dic[s_idx])
                        t_spans.append(t_id_span_dic[t_idx])
                        labels.append(1)
                    s_to_t[s_idx] = t_idx
                    t_to_s[t_idx] = s_idx

                # negative examples
                s_idxs = list(s_id_span_dic.keys())
                t_idxs = list(t_id_span_dic.keys())
                for id, span in s_id_span_dic.items():
                    rand_id = self._get_random_pair(t_idxs, s_to_t[id])
                    s_spans.append(span)
                    t_spans.append(t_id_span_dic[rand_id])
                    labels.append(-1)
                for id, span in t_id_span_dic.items():
                    rand_id = self._get_random_pair(s_idxs, t_to_s[id])
                    s_spans.append(s_id_span_dic[rand_id])
                    t_spans.append(span)
                    labels.append(-1)

            elif loss_type == 'TripletMarginLoss' or loss_type == 'SoftMarginLoss' or loss_type == 'MarginRankingLoss':
                t_idxs = list(t_id_span_dic.keys())
                for (s_idx, t_idx) in align_labels[pairidx]:
                    if s_idx != '-1' and t_idx != '-1':
                        # positive examples
                        s_spans.append(s_id_span_dic[s_idx])
                        t_spans.append(t_id_span_dic[t_idx])
                        labels.append(1)

                        # negative examples
                        rand_id = self._get_random_pair(t_idxs, t_idx)
                        s_spans.append(s_id_span_dic[s_idx])
                        t_spans.append(t_id_span_dic[rand_id])
                        labels.append(-1)

            # elif loss_type=='MSELoss':
            #     t_idxs = list(t_id_span_dic.keys())
            #     for (s_idx, t_idx) in align_labels[pairidx]:
            #         if s_idx != '-1' and t_idx != '-1':
            #             # positive examples
            #             s_spans.append(s_id_span_dic[s_idx])
            #             t_spans.append(t_id_span_dic[t_idx])
            #             labels.append(1)
            #
            #             # negative examples
            #             rand_id = self._get_random_pair(t_idxs, t_idx)
            #             s_spans.append(s_id_span_dic[s_idx])
            #             t_spans.append(t_id_span_dic[rand_id])
            #             labels.append(0)
            else:
                raise ValueError('Unsupported Loss function!')

            self.s_span_set.append(s_spans)
            self.t_span_set.append(t_spans)
            self.label_set.append(labels)

    def _get_random_pair(self, id_list, avoid_id):
        # Avoid adjuscent nodes
        avoid_num = int(avoid_id.replace('c', ''))
        avoid_ids = [avoid_id, 'c' + str(avoid_num - 1), 'c' + str(avoid_num + 1)]

        rand_id = avoid_id
        while rand_id in avoid_ids:
            rand_id = random.choice(id_list)

        return rand_id

    def _create_phrase_dic(self, trees):
        # Create phrase id - span dictionary
        id_span_dic = {}
        for node in trees:
            id_span_dic[node.id] = (node.start, node.end)
        return id_span_dic

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = {'feature': self.features[idx], 's_spans': self.s_span_set[idx], 't_spans': self.t_span_set[idx],
                  'labels': self.label_set[idx]}
        return sample

    def get_batch(self, ids):
        batch = {}
        batch['features'] = []
        batch['s_span_set'] = []
        batch['t_span_set'] = []
        batch['label_set'] = []
        for i in ids[0][:]:
            item = self.__getitem__(i)
            batch['features'].append(item['feature'])
            batch['s_span_set'].append(item['s_spans'])
            batch['t_span_set'].append(item['t_spans'])
            batch['label_set'].append(item['labels'])
        return batch


def prepare_input_dual(source, target, s_trees, t_trees, tokenizer, max_seq_length):
    # Convert sentences into examples
    examples = []
    for i in range(len(source)):
        examples.append(InputExampleDual(unique_id=i, text_a=source[i], text_b=target[i]))

    features = _convert_examples_to_features_dual(
        examples, s_trees, t_trees, max_seq_length=max_seq_length, tokenizer=tokenizer)

    return features


def _convert_examples_to_features_dual(examples, s_trees, t_trees, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in tqdm(enumerate(examples)):
        tokens_a, subword_map_a = tokenizer.tokenize(example.text_a)
        if tokens_a == None:
            continue
        tokens_b, subword_map_b = tokenizer.tokenize(example.text_b)
        if tokens_b == None:
            continue

        if len(tokens_a) + len(tokens_b) <= max_seq_length - 3:
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            margin = len(tokens) - 1
            tokens += tokens_b + ["[SEP]"]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids_a = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens_a + ["[SEP]"])
            input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b + ["[SEP]"])

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length

            # Adjust phrase indexes for tokenization and concatenation
            map_word_to_token(s_trees[ex_index], subword_map_a)
            map_word_to_token(t_trees[ex_index], subword_map_b, margin=margin)

            features.append(InputFeaturesDual(input_ids=input_ids, input_ids_a=input_ids_a, input_ids_b=input_ids_b,
                                              input_mask=input_mask))

    return features


def map_word_to_token(tree, map, margin=0):
    # Comvert word-based phrase indexes to BERT subwords based indexes
    for node in tree:
        node.start = map[node.start] + margin
        node.end = map[node.end] + margin
