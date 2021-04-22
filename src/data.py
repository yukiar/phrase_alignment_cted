import os, glob, pathlib
from xml.etree import ElementTree


class Node:
    def __init__(self, id, tokens, start, end, pidx, cidx_list, pa1, pa2, pa3):
        self.id = id  # node id, e.g., c0 and t10
        self.txt = ' '.join(tokens).strip(' ')
        self.tokens = tokens
        self.start = start
        self.end = end
        self.pidx = pidx
        self.cidx_list = cidx_list
        self.pa1 = [aid.strip() for aid in pa1.split(' ')]
        self.pa2 = [aid.strip() for aid in pa2.split(' ')]
        self.pa3 = [aid.strip() for aid in pa3.split(' ')]

    def update_pid(self, id_idx_dic):
        self.pidx = id_idx_dic[self.pidx]

    def get_alignment(self, annotator_label, swap=False):
        if swap:
            if annotator_label == 'pa1':
                return [(aid, self.id) for aid in self.pa1]
            elif annotator_label == 'pa2':
                return [(aid, self.id) for aid in self.pa2]
            elif annotator_label == 'pa3':
                return [(aid, self.id) for aid in self.pa3]
        else:
            if annotator_label == 'pa1':
                return [(self.id, aid) for aid in self.pa1]
            elif annotator_label == 'pa2':
                return [(self.id, aid) for aid in self.pa2]
            elif annotator_label == 'pa3':
                return [(self.id, aid) for aid in self.pa3]

    def __eq__(self, other):
        if other is None: return False
        if not isinstance(other, Node):
            raise TypeError("Must compare against type Node")
        return self.id == other.id

    def __lt__(self, other):
        if not isinstance(other, Node):
            raise TypeError("Must compare against type Node")
        return self.id < other.id


def is_proposer_ancestor(nid, p_cand, tree):
    node_k, k_idx = get_node_by_id(tree, nid)
    if k_idx == node_k.pidx:  # root does not have a proper ancestor
        return False

    ans_k = []
    get_all_ancestors(ans_k, k_idx, tree)
    if p_cand in ans_k:
        return True
    else:
        return False


def is_ancestor(iid, jid, tree):
    _, i_idx = get_node_by_id(tree, iid)
    _, j_idx = get_node_by_id(tree, jid)

    ans_i = [i_idx]
    get_all_ancestors(ans_i, i_idx, tree)
    if j_idx in ans_i:
        return True
    else:
        return False


def find_lca(i, j, tree):
    _, i_idx = get_node_by_id(tree, i)
    _, j_idx = get_node_by_id(tree, j)
    ans_i, ans_j = [i_idx], [j_idx]
    get_all_ancestors(ans_i, i_idx, tree)
    get_all_ancestors(ans_j, j_idx, tree)

    common_ans = set(ans_i) & set(ans_j)
    lca = min(common_ans)  # In tree, deeper nodes have smaller idxs

    return lca


def get_all_ancestors(ans, i, tree):
    ans.append(tree[i].pidx)
    if tree[i].pidx == i:  # root
        return
    get_all_ancestors(ans, tree[i].pidx, tree)


def get_all_descendants(des, i, tree):
    if len(tree[i].cidx_list) == 1 and tree[i].cidx_list[0] == -1:  # leaf
        return

    for cidx in tree[i].cidx_list:
        des.append(cidx)
        get_all_descendants(des, cidx, tree)


def get_node_by_id(tree, id):
    for nidx in range(len(tree)):
        if tree[nidx].id == id:
            return tree[nidx], nidx
    return None


def get_idx_by_id(tree, id):
    for nidx in range(len(tree)):
        if tree[nidx].id == id:
            return nidx
    return None


def load_corpus(path, dev_or_test, bos_eos, text_match):
    # Load trees
    s_trees, t_trees, s_tokens, t_tokens = [], [], [], []

    if dev_or_test is None:
        xml_dir = path
    else:
        xml_dir = os.path.join(path, dev_or_test)
    pair_ids = [pathlib.Path(annot_path).stem[2:] for annot_path in glob.glob(os.path.join(xml_dir, 's-*.xml'))]
    for pair_id in pair_ids:
        # Read source
        tree, tokens = _read_xml(os.path.join(xml_dir, 's-' + pair_id + '.xml'), bos_eos)
        s_trees.append(tree)
        s_tokens.append(tokens)

        # Read target
        tree, tokens = _read_xml(os.path.join(xml_dir, 't-' + pair_id + '.xml'), bos_eos)
        t_trees.append(tree)
        t_tokens.append(tokens)

    # Load annotations
    annotator_A = get_annotations(s_trees, t_trees, 'pa1')
    annotator_B = get_annotations(s_trees, t_trees, 'pa2')
    annotator_C = get_annotations(s_trees, t_trees, 'pa3')

    if text_match:
        annotator_A = convert_id_to_text(annotator_A, s_trees, t_trees)
        annotator_B = convert_id_to_text(annotator_B, s_trees, t_trees)
        annotator_C = convert_id_to_text(annotator_C, s_trees, t_trees)

    return s_tokens, t_tokens, s_trees, t_trees, annotator_A, annotator_B, annotator_C


def get_annotations(s_trees, t_trees, annot_label):
    all_annotations = []
    for s_tree, t_tree in zip(s_trees, t_trees):
        annotations = []
        for node in s_tree:
            # Add alignment of source -> target
            annotations += node.get_alignment(annot_label)
        for node in t_tree:
            # Add alignment of target -> source
            annotations += node.get_alignment(annot_label, swap=True)
        all_annotations.append(set(annotations))
    return all_annotations


def convert_id_to_text(annotation, s_trees, t_trees, verbose=False):
    text_base_annotation = []
    for i, pairs in enumerate(annotation):
        txt_pairs = set()
        for s_id, t_id in pairs:
            s_txt = _find_and_get_text(s_id, s_trees[i], verbose)
            t_txt = _find_and_get_text(t_id, t_trees[i], verbose)
            txt_pairs.add((s_txt, t_txt))
        text_base_annotation.append(txt_pairs)

    return text_base_annotation


def _find_and_get_text(id, tree, verbose):
    if id == '-1':
        return '---'
    node = list(filter(lambda x: id == x.id, tree))[0]
    txt = ' '.join(node.tokens)

    if verbose:
        return node.id + ' ' + txt.strip(' ')
    else:
        return txt.strip(' ')


def _read_xml(xml_path, bos_eos):
    tree, tokens = [], []
    id_idx_dic = {}
    root = ElementTree.parse(xml_path).getroot()[0]
    _recursive_trace_postorder(root, root.get('id').strip(' '), tree, tokens, id_idx_dic, bos_eos)

    for node in tree:
        node.update_pid(id_idx_dic)

    return tree, tokens


def _recursive_trace_postorder(node, pid, tree, tokens, id_idx_dic, bos_eos):
    if len(node) == 1:  # unary or pre-terminal
        child = node[0]
        nid = node.get('id').strip(' ')
        pa1 = node.get('pa1').strip(' ') if 'pa1' in node.attrib else '-1'
        pa2 = node.get('pa2').strip(' ') if 'pa2' in node.attrib else '-1'
        pa3 = node.get('pa3').strip(' ') if 'pa3' in node.attrib else '-1'
        if child.get('id')[0] == 't':  # pre-terminal
            tokens.append(child.text)

            if bos_eos:  # increment index for <s> symbol
                tree.append(Node(nid, [child.text], len(tokens), len(tokens) + 1, pid, [-1], pa1, pa2, pa3))
            else:
                tree.append(Node(nid, [child.text], len(tokens) - 1, len(tokens), pid, [-1], pa1, pa2, pa3))

            id_idx_dic[nid] = len(tree) - 1
            return len(tokens) - 1, len(tokens), len(tree) - 1
        else:  # unary
            start, end, cidx = _recursive_trace_postorder(child, nid, tree, tokens, id_idx_dic, bos_eos)

            if bos_eos:
                tree.append(Node(nid, tokens[start:end], start + 1, end + 1, pid, [cidx], pa1, pa2, pa3))
            else:
                tree.append(Node(nid, tokens[start:end], start, end, pid, [cidx], pa1, pa2, pa3))

            id_idx_dic[nid] = len(tree) - 1
            return start, end, len(tree) - 1
    else:
        nid = node.get('id').strip(' ')
        pa1 = node.get('pa1').strip(' ') if 'pa1' in node.attrib else '-1'
        pa2 = node.get('pa2').strip(' ') if 'pa2' in node.attrib else '-1'
        pa3 = node.get('pa3').strip(' ') if 'pa3' in node.attrib else '-1'
        start, _, left_child_idx = _recursive_trace_postorder(node[0], nid, tree, tokens, id_idx_dic, bos_eos)
        _, end, right_child_idx = _recursive_trace_postorder(node[1], nid, tree, tokens, id_idx_dic, bos_eos)

        if bos_eos:  # increment index for <s> symbol
            tree.append(
                Node(nid, tokens[start:end], start + 1, end + 1, pid, [left_child_idx, right_child_idx], pa1, pa2, pa3))
        else:
            tree.append(Node(nid, tokens[start:end], start, end, pid, [left_child_idx, right_child_idx], pa1, pa2, pa3))

        id_idx_dic[nid] = len(tree) - 1
        return start, end, len(tree) - 1
