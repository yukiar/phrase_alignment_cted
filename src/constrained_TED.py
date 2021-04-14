import sys
import numpy as np
from MinCostFlow import MinCostFlow


class Operation(object):
    delete = 0
    insert = 1
    align = 2

    def __init__(self, op, score, arg1=None, arg2=None):
        self.type = op
        self.score = score
        self.arg1 = arg1
        self.arg2 = arg2

    def to_tuple(self):
        if self.type == self.delete:
            return (self.arg1.id, '-1')
        elif self.type == self.insert:
            return ('-1', self.arg2.id)
        else:
            return (self.arg1.id, self.arg2.id)

    def __repr__(self):
        if self.type == self.delete:
            return '<Operation S-Null: (' + self.arg1.id + ') ' + self.arg1.txt + ' <--> (-1) NULL> TED:' + format(
                self.score, '.2f')
        elif self.type == self.insert:
            return '<Operation T-Null: (-1) NULL <--> (' + self.arg2.id + ') ' + self.arg2.txt + '> TED:' + format(
                self.score, '.2f')
        else:
            return '<Operation Align: (' + self.arg1.id + ') ' + self.arg1.txt + ' <--> (' + self.arg2.id + ') ' + self.arg2.txt + '> TED:' + format(
                self.score, '.2f')

    def __eq__(self, other):
        if isinstance(other, Operation):
            return self.__dict__ == other.__dict__
        return False

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))

    def __lt__(self, other):
        if not isinstance(other, Operation):
            raise TypeError("Must compare against type Node")
        if self.arg1 is None: return -1
        if other.arg1 is None: return 1
        return self.arg1 < other.arg1


def constrained_ted(sentence_idx, s_tree, t_tree, sim):
    DELETE = Operation.delete
    INSERT = Operation.insert
    ALIGN = Operation.align

    DF = np.zeros((len(s_tree) + 1, len(t_tree) + 1))
    DT = np.zeros((len(s_tree) + 1, len(t_tree) + 1))
    OPF = [[[] for _ in range(len(t_tree) + 1)] for _ in range(len(s_tree) + 1)]
    OPT = [[[] for _ in range(len(t_tree) + 1)] for _ in range(len(s_tree) + 1)]

    for i in range(len(s_tree)):
        idx = i + 1  # 1-base indexing
        for cidx in s_tree[i].cidx_list:
            DF[idx][0] += DT[cidx + 1][0]  # 1-base indexing
            OPF[idx][0].extend(OPT[cidx + 1][0])

        DT[idx][0] = DF[idx][0] + sim.null_align_score(s_tree[i])
        OPT[idx][0] = OPF[idx][0][:] + [Operation(DELETE, DT[idx][0], s_tree[i])]

    for j in range(len(t_tree)):
        idx = j + 1
        for cidx in t_tree[j].cidx_list:
            DF[0][idx] += DT[0][cidx + 1]  # 1-base indexing
            OPF[0][idx].extend(OPT[0][cidx + 1])

        DT[0][idx] = DF[0][idx] + sim.null_align_score(t_tree[j])
        OPT[0][idx] = OPF[0][idx][:] + [Operation(INSERT, DT[0][idx], arg2=t_tree[j])]

    for i in range(len(s_tree)):
        i_idx = i + 1  # 1-base indexing
        for j in range(len(t_tree)):
            j_idx = j + 1  # 1-base indexing

            # Compute forests
            scores = np.zeros(3)
            if (len(s_tree[i].cidx_list) == 1 and s_tree[i].cidx_list[0] == -1) or (
                    len(t_tree[j].cidx_list) == 1 and t_tree[j].cidx_list[0] == -1):  # Pre-terminal nodes
                scores[ALIGN] = sys.float_info.max
            else:
                solver = MinCostFlow(s_tree[i].cidx_list, t_tree[j].cidx_list, DT)
                scores[ALIGN], ops_min_cost_flow = solver.min_cost_flow(OPT)
            scores[INSERT], minidx_insert = _min_deletion_insertion(i_idx, j_idx, t_tree[j].cidx_list, DF,
                                                                    False)  # Insertion
            scores[DELETE], minidx_delete = _min_deletion_insertion(i_idx, j_idx, s_tree[i].cidx_list, DF,
                                                                    True)  # Deletion

            minidx = scores.argmin()
            DF[i_idx][j_idx] = scores[minidx]
            if minidx == INSERT:
                OPF[i_idx][j_idx] = _consolidate_operations(OPF[i_idx][minidx_insert], OPF[0][j_idx],
                                                            OPF[0][minidx_insert])
            elif minidx == DELETE:
                OPF[i_idx][j_idx] = _consolidate_operations(OPF[minidx_delete][j_idx], OPF[i_idx][0],
                                                            OPF[minidx_delete][0])
            else:
                OPF[i_idx][j_idx] = ops_min_cost_flow

            # Compute trees
            scores = np.zeros(3)
            scores[ALIGN] = DF[i_idx][j_idx] + sim.align_score(s_tree[i], t_tree[j], sentence_idx)  # Alignment
            scores[INSERT], minidx_insert = _min_deletion_insertion(i_idx, j_idx, t_tree[j].cidx_list, DT,
                                                                    False)  # Insertion
            scores[DELETE], minidx_delete = _min_deletion_insertion(i_idx, j_idx, s_tree[i].cidx_list, DT,
                                                                    True)  # Deletion

            minidx = scores.argmin()
            DT[i_idx][j_idx] = scores[minidx]
            if minidx == INSERT:
                OPT[i_idx][j_idx] = _consolidate_operations(OPT[i_idx][minidx_insert], OPT[0][j_idx],
                                                            OPT[0][minidx_insert])
            elif minidx == DELETE:
                OPT[i_idx][j_idx] = _consolidate_operations(OPT[minidx_delete][j_idx], OPT[i_idx][0],
                                                            OPT[minidx_delete][0])
            else:
                OPT[i_idx][j_idx] = OPF[i_idx][j_idx][:] + [Operation(ALIGN, DT[i_idx][j_idx], s_tree[i], t_tree[j])]

    return DT[-1][-1], OPT[-1][-1], DT, OPT


def _consolidate_operations(to_concat, to_filter, to_be_removed):
    final_ops = [ops for ops in to_concat] + [ops for ops in to_filter if ops not in to_be_removed]
    return final_ops


def _min_deletion_insertion(sidx, tidx, cidx_list, D, is_deletion):
    if is_deletion:
        scores = np.zeros((len(cidx_list)))
        for i, cidx in enumerate(cidx_list):
            scores[i] = D[cidx + 1][tidx] - D[cidx + 1][0]

        minidx = scores.argmin()
        minscore = D[sidx][0] + scores[minidx]
        minidx = cidx_list[minidx]
    else:
        scores = np.zeros((len(cidx_list)))
        for i, cidx in enumerate(cidx_list):
            scores[i] = D[sidx][cidx + 1] - D[0][cidx + 1]

        minidx = scores.argmin()
        minscore = D[0][tidx] + scores[minidx]
        minidx = cidx_list[minidx]

    return minscore, minidx + 1  # 1-base indexing
