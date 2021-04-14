from data import load_corpus, convert_id_to_text
from bert_phrase_sim import BERT_sim
from model1e_phrase_sim import BERT1E_sim
from model1f_phrase_sim import BERT1F_sim
from wordvec_based_phrase_sim import wordvec_sim
import numpy as np
import codecs, argparse

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
args = parser.parse_args()


def eval_average_performance():
    print('Evaluate: ' + args.model_name)
    print('Seed: ' + str(args.seed))
    print('Load a similarity model...')
    # sim=PhraseSimSample()  # For debugging

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

    add_bos_eos = True if 'bert' in model_type.lower() else False

    print('Load xml files...')
    # s_trees, t_trees, s_texts, t_texts = load('../data/sample/', True)
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
    elif 'ELMo' in model_type:
        print('Encoding sentences by ELMo...')
        sim.encode(s_tokens, t_tokens)

    sim.set_null_thresh(args.null_thresh, min, max)
    print('Start alignment: Threshold ' + format(args.null_thresh, '.2f'))
    scores, scores_breakdown = eval_alignments(sim, s_trees, t_trees, annotator12_or, annotator12_and, annotator23_or,
                                               annotator23_and, annotator31_or, annotator31_and, verbose=True)
    print('baseline: ' + str(scores))

    # Save results
    with codecs.open(
            args.out_dir + sim.get_model_name() + '_' + sim.get_pooling() + '_ALIR_ALIP_' + str(args.seed) + '.txt',
            'w', encoding='utf-8') as f:
        f.write('{0:.2f}\t{1:.2f}\t{2:.2f}\n'.format(args.null_thresh, scores[0], scores[1]))
        for alir, alip in zip(scores_breakdown[0], scores_breakdown[1]):
            f.write('{0:.2f}\t{1:.2f}\n'.format(alir, alip))


def gridsearch_lambda():
    print('Evaluate: ' + args.model_name)
    print('Load a similarity model...')
    # sim=PhraseSimSample()  # For debugging

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
        elif 'ELMo' in model_type:
            print('Encoding sentences by ELMo...')
            sim.encode(s_tokens, t_tokens)

        results = {}
        for thresh in np.arange(0.05, 0.99, 0.05, dtype='float64'):
            sim.set_null_thresh(thresh, min, max)
            print('Start alignment: Threshold ' + format(thresh, '.2f'))
            results[thresh] = eval_alignments(sim, s_trees, t_trees, annotator12_or, annotator12_and, annotator23_or,
                                              annotator23_and, annotator31_or, annotator31_and)
            print('baseline: ' + str(results[thresh]))

        # Save results
        with codecs.open(
                args.out_dir + TASK + '_' + sim.get_model_name() + '_' + sim.get_pooling() + '_ALIR_ALIP_curve.txt',
                'w', encoding='utf-8') as f:
            for th, val in results.items():
                f.write('{0:.2f}\t{1:.2f}\t{2:.2f}\n'.format(th, val[0], val[1]))


def eval_alignments(sim, s_trees, t_trees, annotator12_or, annotator12_and, annotator23_or, annotator23_and,
                    annotator31_or, annotator31_and, verbose=False):
    output = []
    for sentence_idx, (s_tree, t_tree) in enumerate(zip(s_trees, t_trees)):
        s_null, t_null, non_null = [], [], []
        scores = {}
        for snode in s_tree:
            for tnode in t_tree:
                scores[(snode.id, tnode.id)] = sim.align_score(snode, tnode, sentence_idx)

        alignments = []
        used_s, used_t = [], []
        for k, v in sorted(scores.items(), key=lambda kv: kv[1]):
            if k[0] not in used_s and k[1] not in used_t:
                if v <= sim.NULL_SCORE:
                    alignments.append((k[0], k[1]))
                else:
                    alignments.append((k[0], '-1'))
                    alignments.append(('-1', k[1]))
                used_s.append(k[0])
                used_t.append(k[1])
        alignments += [(snode.id, '-1') for snode in s_tree if snode.id not in used_s]
        alignments += [('-1', tnode.id) for tnode in t_tree if tnode.id not in used_t]
        output.append(set(alignments))

    # Convert to text-based alignment pairs
    output = convert_id_to_text(output, s_trees, t_trees)
    alirs, alips = compute_average(output, annotator12_or, annotator12_and, annotator23_or, annotator23_and,
                                   annotator31_or,
                                   annotator31_and)

    if verbose:
        return (np.mean(alirs), np.mean(alips)), (alirs, alips)
    else:
        return (np.mean(alirs), np.mean(alips))


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
    if args.null_thresh is None:
        gridsearch_lambda()
    else:
        eval_average_performance()
