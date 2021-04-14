import networkx as nx


#####################################################
#  Need to write codes to make costs positive integers!!
#  https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/flow/mincost.html#max_flow_min_cost
#####################################################

class MinCostFlow(object):
    def __init__(self, i_idxes, j_idxes, DT):
        self.start_nodes = []
        self.end_nodes = []
        self.capacities = []
        self.unit_costs = []
        self.G = nx.DiGraph()
        self.source, self.sink, self.e_i, self.e_j, self.idx_id_dic = self._construct_graph(i_idxes, j_idxes, DT)

    def _construct_graph(self, i_ids, j_ids, DT):
        # Create nodeid-index dictionary
        idx_id_dic = {}
        for i_id in i_ids:
            idx_id_dic[len(idx_id_dic)] = i_id
        for j_id in j_ids:
            idx_id_dic[len(idx_id_dic)] = j_id
        e_i = len(idx_id_dic) * 10
        e_j = len(idx_id_dic) * 10 + 1
        source = len(idx_id_dic) * 100
        sink = len(idx_id_dic) * 100 + 1
        idx_id_dic[e_i]="e_1"
        idx_id_dic[e_j]="e_j"
        idx_id_dic[source]="s"
        idx_id_dic[sink]="t"

        # Add link from the source to source-nodes
        for i, i_id in enumerate(i_ids):
            self._add_arc(source, i, 1, 0)

        # Add link from the source to e_i
        self._add_arc(source, e_i, len(j_ids), 0)

        # Add link from source nodes to target nodes
        for i, i_id in enumerate(i_ids):
            for j, j_id in enumerate(j_ids):
                self._add_arc(i, j + len(i_ids), 1, int(DT[i_id + 1][j_id + 1]))  # DT is 1-base indexing
            # Add link to e_j
            self._add_arc(i, e_j, 1, int(DT[i_id + 1][0]))  # DT is 1-base indexing
        # Add link from e_i
        for j, j_id in enumerate(j_ids):
            self._add_arc(e_i, j + len(i_ids), 1, int(DT[0][j_id + 1]))  # DT is 1-base indexing

        # Add link from e_i to e_j
        # Below capacity is written in the paper
        # cap_empties = max(len(i_idxes), len(j_idxes)) - min(len(i_idxes), len(j_idxes))
        # But it makes a degenerate optima that generates unnecessary null alignments
        # Capacity should be: ni+nj - min(ni, nj) - {max(ni, nj) - min(ni,nj)} = min(ni, nj)
        cap_empties = min(len(i_ids), len(j_ids))
        self._add_arc(e_i, e_j, cap_empties, 0)

        # Add link from target nodes to sink
        for j, j_id in enumerate(j_ids):
            self._add_arc(j + len(i_ids), sink, 1, 0)

        # Add link from e_j to sink
        self._add_arc(e_j, sink, len(i_ids), 0)

        # Add each arc.
        for i in range(0, len(self.start_nodes)):
            self.G.add_edge(self.start_nodes[i], self.end_nodes[i], weight=self.unit_costs[i],
                            capacity=self.capacities[i])
        return source, sink, e_i, e_j, idx_id_dic

    def min_cost_flow(self, OPT, varbose=False):
        # Find the minimum cost flow
        flowDict = nx.max_flow_min_cost(self.G, self.source, self.sink)
        mincost = nx.cost_of_flow(self.G, flowDict)
        ops = []

        if varbose:
            print('Minimum cost:', mincost)
            print('  Arc    Flow ')

        for i, dic in flowDict.items():
            for j, flow in dic.items():
                if varbose:
                    print('%1s -> %1s   %3s' % (self.idx_id_dic[i], self.idx_id_dic[j], flow))

                if i != self.source and j != self.sink and flow > 0:
                    if i == self.e_i:
                        i_idx = 0
                    else:
                        i_idx = self.idx_id_dic[i] + 1  # 1-base indexing
                    if j == self.e_j:
                        j_idx = 0
                    else:
                        j_idx = self.idx_id_dic[j] + 1  # 1-base indexing

                    ops.extend(OPT[i_idx][j_idx])

        return mincost, ops

    def _add_arc(self, s, t, cap, cost):
        self.start_nodes.append(s)
        self.end_nodes.append(t)
        self.capacities.append(cap)
        self.unit_costs.append(cost)
