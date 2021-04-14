from data import load_corpus, convert_id_to_text, get_node_by_id
from constrained_TED import constrained_ted
from bert_phrase_sim import BERT_sim
from model1e_phrase_sim import BERT1E_sim
from model1f_phrase_sim import BERT1F_sim
from wordvec_based_phrase_sim import wordvec_sim
import numpy as np
import codecs, argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--model_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="Path to a model")
parser.add_argument("--model_name",
                    default=None,
                    type=str,
                    required=True,
                    help="Model name")
parser.add_argument("--pooling",
                    default=None,
                    type=str,
                    required=True,
                    help="Pooling method: max or mean")
parser.add_argument("--null_thresh",
                    default=None,
                    type=float,
                    help="Null-alignment threshold")
parser.add_argument("--seed",
                    default=42,
                    type=int,
                    help="Seed for initialization")
parser.add_argument("--decode", action='store_true', help="Output alignments")
args = parser.parse_args()


def decode():
    print('Alignment by ' + args.model_name)
    print('Load a similarity model...')

    sim, model_type, min, max = setup_model()
    add_bos_eos = True if 'bert' in model_type.lower() else False

    print('Load xml files...')
    s_tokens, t_tokens, s_trees, t_trees, _, _, _ = load_corpus('../data/SPADE/', 'test', add_bos_eos, True)
    if 'bert' in model_type.lower():
        print('Encoding sentences by BERT...')
        sim.encode(s_tokens, t_tokens, s_trees, t_trees)

    sim.set_null_thresh(args.null_thresh, min, max)
    print('Start alignment: Threshold ' + format(args.null_thresh, '.2f'))
    output_alignment(sim, s_trees, t_trees)


def output_alignment(sim, s_trees, t_trees):
    output, output_postprocess = align(sim, s_trees, t_trees, verbose=True)
    out_path = args.out_dir + sim.get_model_name() + '/'
    os.makedirs(out_path, exist_ok=True)

    for idx in range(len(output)):
        # Save results
        with codecs.open(out_path + str(idx) + '.txt', 'w', encoding='utf-8') as f:
            for a in sorted(output[idx], key=lambda a: len(a[0]), reverse=True):
                f.write('{0} ||| {1} \n'.format(a[0], a[1]))

        with codecs.open(out_path + 'pp_' + str(idx) + '.txt', 'w', encoding='utf-8') as f:
            for a in sorted(output_postprocess[idx], key=lambda a: len(a[0]), reverse=True):
                f.write('{0} ||| {1} \n'.format(a[0], a[1]))


def eval_average_performance():
    print('Evaluate: ' + args.model_name)
    print('Seed: ' + str(args.seed))
    print('Load a similarity model...')

    sim, model_type, min, max = setup_model()
    add_bos_eos = True if 'bert' in model_type.lower() else False

    print('Load xml files...')
    s_tokens, t_tokens, s_trees, t_trees, annotator1, annotator2, annotator3 = load_corpus('../data/SPADE/', 'test',
                                                                                           add_bos_eos, True)

    annotator12_and = [annotator1[i] & annotator2[i] for i in range(len(annotator1))]
    annotator12_or = [annotator1[i] | annotator2[i] for i in range(len(annotator1))]
    annotator23_and = [annotator2[i] & annotator3[i] for i in range(len(annotator2))]
    annotator23_or = [annotator2[i] | annotator3[i] for i in range(len(annotator2))]
    annotator31_and = [annotator3[i] & annotator1[i] for i in range(len(annotator3))]
    annotator31_or = [annotator3[i] | annotator1[i] for i in range(len(annotator3))]

    if 'bert' in model_type.lower():
        print('Encoding sentences by BERT...')
        sim.encode(s_tokens, t_tokens, s_trees, t_trees)
    # [2020/9/25] allennlp is no longer compatible. Disabled ELMo.
    # elif 'ELMo' in model_type:
    #     print('Encoding sentences by ELMo...')
    #     sim.encode(s_tokens, t_tokens)

    sim.set_null_thresh(args.null_thresh, min, max)
    print('Start alignment: Threshold ' + format(args.null_thresh, '.2f'))
    scores, scores_breakdown, scores_pp, scores_breakdown_pp = eval_alignments(sim, s_trees, t_trees, annotator12_or,
                                                                               annotator12_and, annotator23_or,
                                                                               annotator23_and, annotator31_or,
                                                                               annotator31_and, verbose=True)
    print('ted: ' + str(scores) + ' +post-process: ' + str(scores_pp))

    # Save results
    with codecs.open(
            args.out_dir + sim.get_model_name() + '_' + sim.get_pooling() + '_ALIR_ALIP_' + str(args.seed) + '.txt',
            'w', encoding='utf-8') as f:
        f.write('Average performance for 3 pairs of annotators\n')
        f.write('{0:.2f}\t{1:.2f}\t{2:.2f}\n'.format(args.null_thresh, scores[0], scores[1]))
        f.write('Performance breakdown for each pair of annotators\n')
        for alir, alip in zip(scores_breakdown[0], scores_breakdown[1]):
            f.write('{0:.2f}\t{1:.2f}\n'.format(alir, alip))

    with codecs.open(
            args.out_dir + sim.get_model_name() + '_' + sim.get_pooling() + '_postprocess_ALIR_ALIP_' + str(
                args.seed) + '.txt',
            'w', encoding='utf-8') as f:
        f.write('Average performance for 3 pairs of annotators\n')
        f.write('{0:.2f}\t{1:.2f}\t{2:.2f}\n'.format(args.null_thresh, scores_pp[0], scores_pp[1]))
        f.write('Performance breakdown for each pair of annotators\n')
        for alir, alip in zip(scores_breakdown_pp[0], scores_breakdown_pp[1]):
            f.write('{0:.2f}\t{1:.2f}\n'.format(alir, alip))


def setup_model():
    model_parts = args.model_name.split('_')
    model_type = model_parts[0]
    if len(model_parts) > 1:
        loss_func = model_parts[1]
        if loss_func in ['MarginRankingLoss']:
            min = 0
            max = 1
        elif loss_func in ['CosineEmbeddingLoss', 'TripletMarginLoss']:
            min = -1
            max = 1
        else:
            raise NotImplementedError("Undefined loss function!")
    else:
        min = -1
        max = 1

    if model_type == 'FastText':
        sim = wordvec_sim(args.pooling, args.model_dir)
    elif model_type == 'BERT1E':
        sim = BERT1E_sim(args.model_dir, args.model_name, args.pooling, args.seed)
    elif model_type == 'BERT1F':
        sim = BERT1F_sim(args.model_dir, args.model_name, args.pooling, args.seed)
    else:
        sim = BERT_sim(args.model_dir, args.model_name, args.pooling, args.seed)

    return sim, model_type, min, max


def gridsearch_lambda():
    print('Evaluate: ' + args.model_name)
    print('Load a similarity model...')
    # sim=PhraseSimSample()  # For debugging

    sim, model_type, min, max = setup_model()
    add_bos_eos = True if 'bert' in model_type.lower() else False

    for TASK in ['dev', 'test']:
        print('Load xml files...')
        # s_trees, t_trees, s_texts, t_texts = load('../data/sample/', True)
        s_tokens, t_tokens, s_trees, t_trees, annotator1, annotator2, annotator3 = load_corpus('../data/SPADE/', TASK,
                                                                                               add_bos_eos, True)
        annotator12_and = [annotator1[i] & annotator2[i] for i in range(len(annotator1))]
        annotator12_or = [annotator1[i] | annotator2[i] for i in range(len(annotator1))]
        annotator23_and = [annotator2[i] & annotator3[i] for i in range(len(annotator2))]
        annotator23_or = [annotator2[i] | annotator3[i] for i in range(len(annotator2))]
        annotator31_and = [annotator3[i] & annotator1[i] for i in range(len(annotator3))]
        annotator31_or = [annotator3[i] | annotator1[i] for i in range(len(annotator3))]

        if 'bert' in model_type.lower():
            print('Encoding sentences by BERT...')
            sim.encode(s_tokens, t_tokens, s_trees, t_trees)
        # [2020/9/25] allennlp is no longer compatible. Disabled ELMo.
        # elif 'ELMo' in model_type:
        #     print('Encoding sentences by ELMo...')
        #     sim.encode(s_tokens, t_tokens)

        results = {}
        for thresh in np.arange(0.05, 0.99, 0.05, dtype='float64'):
            sim.set_null_thresh(thresh, min, max)
            print('Start alignment: Threshold ' + format(thresh, '.2f'))
            results[thresh] = eval_alignments(sim, s_trees, t_trees, annotator12_or, annotator12_and, annotator23_or,
                                              annotator23_and, annotator31_or, annotator31_and)
            print('ted: ' + str(results[thresh][0]) + ' +post-process: ' + str(results[thresh][1]))

        # Save results
        with codecs.open(
                args.out_dir + TASK + '_' + sim.get_model_name() + '_' + sim.get_pooling() + '_ALIR_ALIP_curve.txt',
                'w', encoding='utf-8') as f:
            for th, val in results.items():
                f.write('{0:.2f}\t{1:.2f}\t{2:.2f}\n'.format(th, val[0][0], val[0][1]))

        with codecs.open(
                args.out_dir + TASK + '_' + sim.get_model_name() + '_' + sim.get_pooling() + '_postprocess_ALIR_ALIP_curve.txt',
                'w', encoding='utf-8') as f:
            for th, val in results.items():
                f.write('{0:.2f}\t{1:.2f}\t{2:.2f}\n'.format(th, val[1][0], val[1][1]))


def eval_alignments(sim, s_trees, t_trees, annotator12_or, annotator12_and, annotator23_or, annotator23_and,
                    annotator31_or, annotator31_and, verbose=False):
    output, output_postprocess = align(sim, s_trees, t_trees)

    alirs, alips = compute_average(output, annotator12_or, annotator12_and, annotator23_or, annotator23_and,
                                   annotator31_or,
                                   annotator31_and)
    alirs_pp, alips_pp = compute_average(output_postprocess, annotator12_or, annotator12_and, annotator23_or,
                                         annotator23_and, annotator31_or,
                                         annotator31_and)
    if verbose:
        return (np.mean(alirs), np.mean(alips)), (alirs, alips), (np.mean(alirs_pp), np.mean(alips_pp)), (
            alirs_pp, alips_pp)
    else:
        return (np.mean(alirs), np.mean(alips)), (np.mean(alirs_pp), np.mean(alips_pp))


def align(sim, s_trees, t_trees, verbose=False):
    output = []
    output_postprocess = []
    for sentence_idx, (s_tree, t_tree) in enumerate(zip(s_trees, t_trees)):
        ted, ops, DT, OPT = constrained_ted(sentence_idx, s_tree, t_tree, sim)

        # print("Constrained TED: {0}".format(str(ted)))
        # ops.sort()
        alignments = set()
        for opr in ops:
            opr_tuple = opr.to_tuple()
            alignments.add(opr_tuple)
        output.append(alignments)

        # Postprocessing
        aligned_s = set()
        aligned_t = set()
        spans = []
        alignments = set()
        null_alignments = set()
        for opr in ops:
            opr_tuple = opr.to_tuple()
            if opr_tuple[1] == '-1':
                s_node, s_idx = get_node_by_id(s_tree, opr_tuple[0])
                span = s_node.end - s_node.start
                spans.append((span, 's', s_idx, opr_tuple))
                null_alignments.add(opr_tuple)
            elif opr_tuple[0] == '-1':
                t_node, t_idx = get_node_by_id(t_tree, opr_tuple[1])
                span = t_node.end - t_node.start
                spans.append((span, 't', t_idx, opr_tuple))
                null_alignments.add(opr_tuple)
            else:
                aligned_s.add(opr_tuple[0])
                aligned_t.add(opr_tuple[1])
                alignments.add(opr_tuple)

        # Search non-compositional alignments
        spans = sorted(spans, key=lambda tup: tup[0], reverse=True)
        for (span, s_or_t, idx, opr_tuple) in spans:
            if s_or_t == 's':
                costs = DT[idx + 1, :]
                min_t_idxs = np.where(costs == costs.min())
                for min_t_idx in min_t_idxs[0]:
                    if min_t_idx > 0 and isCompatible(OPT[idx + 1][min_t_idx], aligned_s, aligned_t):
                        updateAlignments(OPT[idx + 1][min_t_idx], aligned_s, aligned_t, alignments, null_alignments)
            else:
                costs = DT[:, idx + 1]
                min_s_idxs = np.where(costs == costs.min())
                for min_s_idx in min_s_idxs[0]:
                    if min_s_idx > 0 and isCompatible(OPT[min_s_idx][idx + 1], aligned_s, aligned_t):
                        updateAlignments(OPT[min_s_idx][idx + 1], aligned_s, aligned_t, alignments, null_alignments)

        # Add leftover (null-alignments)
        for opr in null_alignments:
            alignments.add(opr)

        output_postprocess.append(alignments)

    # Convert to text-based alignment pairs
    output = convert_id_to_text(output, s_trees, t_trees, verbose)
    # Convert to text-based alignment pairs
    output_postprocess = convert_id_to_text(output_postprocess, s_trees, t_trees, verbose)

    return output, output_postprocess


def updateAlignments(operations, aligned_s, aligned_t, alignments, null_alignments):
    for opr in operations:
        opr_tuple = opr.to_tuple()
        if opr_tuple[0] != '-1' and opr_tuple[1] != '-1':
            aligned_s.add(opr_tuple[0])
            aligned_t.add(opr_tuple[1])
            alignments.add(opr_tuple)
            null_alignments.remove((opr_tuple[0], '-1'))
            null_alignments.remove(('-1', opr_tuple[1]))


def isCompatible(operations, aligned_s, aligned_t):
    for opr in operations:
        opr_tuple = opr.to_tuple()
        if opr_tuple[0] != '-1' and opr_tuple[1] != '-1':
            if opr_tuple[0] in aligned_s:
                return False
            elif opr_tuple[1] in aligned_t:
                return False
    return True

def compute_average(output, annotator12_or, annotator12_and, annotator23_or, annotator23_and, annotator31_or,
                    annotator31_and):
    # Evaluate
    alirs, alips = [], []

    val = alir_alip(output, annotator12_or, annotator12_and)
    alirs.append(val[0])
    alips.append(val[1])

    val = alir_alip(output, annotator23_or, annotator23_and)
    alirs.append(val[0])
    alips.append(val[1])

    val = alir_alip(output, annotator31_or, annotator31_and)
    alirs.append(val[0])
    alips.append(val[1])

    return alirs, alips


def alir_alip(output, gold_or, gold_and):
    alir = float(sum([len(output[i] & gold_and[i]) for i in range(len(gold_and))])) * 100 / float(
        sum([len(gold_and[i]) for i in range(len(gold_and))]))
    alip = float(sum([len(output[i] & gold_or[i]) for i in range(len(gold_or))])) * 100 / float(
        sum([len(output[i]) for i in range(len(output))]))

    return alir, alip


if __name__ == "__main__":
    # execute only if run as a script
    if args.decode:
        decode()
    elif args.null_thresh is None:
        gridsearch_lambda()
    else:
        eval_average_performance()
