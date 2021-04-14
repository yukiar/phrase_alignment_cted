from data import load_corpus, find_lca, is_proposer_ancestor, is_ancestor
import itertools, os, codecs
from tqdm import trange


def load_annotation(s_trees, t_trees, annotator1, annotator2, annotator3):
    # Satisfaction of the condition of constrained TED
    annotator12_and = [annotator1[i] & annotator2[i] for i in range(len(annotator1))]
    # annotator12_or = [annotator1[i] | annotator2[i] for i in range(len(annotator1))]
    annotator23_and = [annotator2[i] & annotator3[i] for i in range(len(annotator2))]
    # annotator23_or = [annotator2[i] | annotator3[i] for i in range(len(annotator2))]
    annotator31_and = [annotator3[i] & annotator1[i] for i in range(len(annotator3))]
    # annotator31_or = [annotator3[i] | annotator1[i] for i in range(len(annotator3))]
    agreed_at_leat_two = [annotator12_and[i] | annotator23_and[i] | annotator31_and[i] for i in range(len(annotator1))]

    agreed_no_null = [[pair for pair in pairs if pair[0] != '-1' and pair[1] != '-1'] for pairs in
                      agreed_at_leat_two]

    break_CTED_triples, break_Monotonicity_pairs = get_noncompositional_alignments(agreed_no_null, s_trees, t_trees)

    return break_CTED_triples, break_Monotonicity_pairs


def get_noncompositional_alignments(alignments, s_trees, t_trees):
    break_CTED_triples = []
    for idx in trange(len(alignments)):
        triples = []
        if len(alignments[idx]) >= 3:
            for i, j, k in itertools.permutations(alignments[idx], 3):
                lca_ij_source = find_lca(i[0], j[0], s_trees[idx])
                lca_ij_target = find_lca(i[1], j[1], t_trees[idx])
                prop_ans_source = is_proposer_ancestor(k[0], lca_ij_source, s_trees[idx])
                prop_ans_target = is_proposer_ancestor(k[1], lca_ij_target, t_trees[idx])
                if prop_ans_source and not prop_ans_target:
                    triples.append((i, j, k))
                elif prop_ans_target and not prop_ans_source:
                    triples.append((i, j, k))
        break_CTED_triples.append(triples)

    break_Monotonicity_pairs = []
    for idx in trange(len(alignments)):
        pairs = []
        if len(alignments[idx]) >= 2:
            for i, j in itertools.permutations(alignments[idx], 2):
                ans_source = is_ancestor(i[0], j[0], s_trees[idx])
                ans_target = is_ancestor(i[1], j[1], t_trees[idx])
                if ans_source and not ans_target:
                    pairs.append((i, j))
                elif ans_target and not ans_source:
                    pairs.append((i, j))
        break_Monotonicity_pairs.append(pairs)

    return break_CTED_triples, break_Monotonicity_pairs


def load_alignments(path, prefix, s_trees, t_trees):
    A = []
    for i in range(len(s_trees)):
        f_path = os.path.join(path, prefix + str(i) + '.txt')
        alignments = []
        with codecs.open(f_path, 'r', 'utf-8') as f:
            for line in f:
                pair = line.split('|||')
                s_id = pair[0].split()[0].strip()
                t_id = pair[1].split()[0].strip()
                if s_id != '---' and t_id != '---':
                    alignments.append((s_id, t_id))
        A.append(alignments)

    break_CTED_triples, break_Monotonicity_pairs = get_noncompositional_alignments(A, s_trees, t_trees)

    return break_CTED_triples, break_Monotonicity_pairs


def eval_noncompositional_alignments(CTED, Mono, break_CTED_triples, break_Monotonicity_pairs):
    eps = 1e-15
    all_gold_triples = [len(gold) for gold in break_CTED_triples]
    all_gold_pairs = [len(gold) for gold in break_Monotonicity_pairs]
    all_pred_triples = [len(pred) for pred in CTED]
    all_pred_pairs = [len(pred) for pred in Mono]
    success_triples, success_pairs = [], []

    for idx in range(len(CTED)):
        success_triples += list(set(CTED[idx]) & set(break_CTED_triples[idx]))
        success_pairs += list(set(Mono[idx]) & set(break_Monotonicity_pairs[idx]))

    prec_cted = len(success_triples) * 100 / (sum(all_pred_triples) + eps)
    prec_mono = len(success_pairs) * 100 / (sum(all_pred_pairs) + eps)
    print('\tPrecision of aligned tripes breaking the CTED constraint: {0:.2f}%'.format(prec_cted))
    print('\tPrecision of aligned pairs breaking the Monotinicity constraint: {0:.2f}%'.format(prec_mono))

    recall_cted = len(success_triples) * 100 / (sum(all_gold_triples) + eps)
    recall_mono = len(success_pairs) * 100 / (sum(all_gold_pairs) + eps)
    print('\tRecall of aligned tripes breaking the CTED constraint: {0:.2f}%'.format(recall_cted))
    print('\tRecall of aligned pairs breaking the Monotinicity constraint: {0:.2f}%'.format(recall_mono))

    f1_cted = 2 * prec_cted * recall_cted / (prec_cted + recall_cted + eps)
    f1_mono = 2 * prec_mono * recall_mono / (prec_mono + recall_mono + eps)
    print('\tF1 of aligned tripes breaking the CTED constraint: {0:.2f}%'.format(f1_cted))
    print('\tF1 of aligned pairs breaking the Monotinicity constraint: {0:.2f}%'.format(f1_mono))


def stat_noncompositional_alignments(s_trees, t_trees, path):
    alignments = []
    for i in range(len(s_trees)):
        f_path = os.path.join(path, 'pp_' + str(i) + '.txt')
        a = []
        with codecs.open(f_path, 'r', 'utf-8') as f:
            for line in f:
                pair = line.split('|||')
                s_id = pair[0].split()[0].strip()
                t_id = pair[1].split()[0].strip()
                if s_id != '---' and t_id != '---':
                    a.append((s_id, t_id))
        alignments.append(a)

    satisfy_constraint = 0
    break_constraint = 0
    dont_care = 0
    for idx in trange(len(alignments)):
        if len(alignments[idx]) >= 3:
            for i, j, k in itertools.permutations(alignments[idx], 3):
                lca_ij_source = find_lca(i[0], j[0], s_trees[idx])
                lca_ij_target = find_lca(i[1], j[1], t_trees[idx])
                prop_ans_source = is_proposer_ancestor(k[0], lca_ij_source, s_trees[idx])
                prop_ans_target = is_proposer_ancestor(k[1], lca_ij_target, t_trees[idx])
                if prop_ans_source and prop_ans_target:
                    satisfy_constraint += 1
                elif prop_ans_source and not prop_ans_target:
                    break_constraint += 1
                elif prop_ans_target and not prop_ans_source:
                    break_constraint += 1
                else:
                    dont_care += 1
    break_constraint_ratio = float(break_constraint) / float(satisfy_constraint + break_constraint) * 100
    break_constraint_ratio_whole = float(break_constraint) / float(
        satisfy_constraint + break_constraint + dont_care) * 100
    print('Percentage that the CTED condition is satisfied only in one side\t{0:.2f}'.format(break_constraint_ratio))
    print('Percentage of unsatisfaction of the CTED condition\t{0:.2f}'.format(break_constraint_ratio_whole))

    satisfy_monotonicity = 0
    break_monotonicity = 0
    dont_care = 0
    for idx in trange(len(alignments)):
        if len(alignments[idx]) >= 2:
            for i, j in itertools.permutations(alignments[idx], 2):
                ans_source = is_ancestor(i[0], j[0], s_trees[idx])
                ans_target = is_ancestor(i[1], j[1], t_trees[idx])
                if ans_source and ans_target:
                    satisfy_monotonicity += 1
                elif ans_source and not ans_target:
                    break_monotonicity += 1
                elif ans_target and not ans_source:
                    break_monotonicity += 1
                else:
                    dont_care += 1
    break_monotonicity_ratio = float(break_monotonicity) / float(satisfy_monotonicity + break_monotonicity) * 100
    break_monotonicity_ratio_whole = float(break_monotonicity) / float(
        satisfy_monotonicity + break_monotonicity + dont_care) * 100
    print('Percentage that the monotonicity condition is satisfied only in one side\t{0:.2f}'.format(
        break_monotonicity_ratio))
    print('Percentage of violation of the monotonicity condition\t{0:.2f}'.format(break_monotonicity_ratio_whole))


def main():
    out_dir = '../out_alignments/'
    models = ['BERT1F_TripletMarginLoss_margin-1.0_lr-3e-05_mean_100_ft-bert-base-uncased',
              'BERT1E_TripletMarginLoss_margin-1.0_lr-5e-05_mean_100_ft-bert-base-uncased',
              'BERTTrainer_TripletMarginLoss_margin-1.0_lr-1e-05_mean_100_ft-bert-base-uncased']

    print("SPADE test")
    s_tokens, t_tokens, s_trees, t_trees, annotator1, annotator2, annotator3 = load_corpus('../data/SPADE/', 'test',
                                                                                           False, text_match=False)
    # Get stat of non-compositional alignment
    for m in models:
        print('Eval: ' + m)
        stat_noncompositional_alignments(s_trees, t_trees, os.path.join(out_dir, m))

    # # Eval non-compositional alignments
    # break_CTED_triples, break_Monotonicity_pairs = load_annotation(s_trees, t_trees, annotator1,
    #                                                                annotator2, annotator3)
    # for m in models:
    #     print('Eval: ' + m)
    #     A_CTED, A_mono = load_alignments(os.path.join(out_dir, m), '', s_trees, t_trees)
    #     PP_CTED, PP_mono = load_alignments(os.path.join(out_dir, m), 'pp_', s_trees, t_trees)
    #     print('\tCTED:')
    #     eval_noncompositional_alignments(A_CTED, A_mono, break_CTED_triples, break_Monotonicity_pairs)
    #     print('\tCTED with post-processing:')
    #     eval_noncompositional_alignments(PP_CTED, PP_mono, break_CTED_triples, break_Monotonicity_pairs)


if __name__ == "__main__":
    main()