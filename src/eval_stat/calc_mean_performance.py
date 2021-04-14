import os, glob, codecs
import numpy as np
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return (m, m - h, m + h)


def load(paths):
    alirs, alips, alifs = [], [], []
    alir_bd, alip_bd, alif_bd = [], [], []
    for p in paths:
        with open(p) as f:
            th_alir_alip = f.readline()
            parts = [float(e) for e in th_alir_alip.split('\t')]
            alirs.append(parts[1])
            alips.append(parts[2])
            alifs.append(2 * parts[1] * parts[2] / (parts[1] + parts[2]))
            for line in f:
                parts = [float(e) for e in line.split('\t')]
                alir_bd.append(parts[0])
                alip_bd.append(parts[1])
                alif_bd.append(2 * parts[0] * parts[1] / (parts[0] + parts[1]))

    return (alirs, alir_bd), (alips, alip_bd), (alifs, alif_bd)


def randomized_sigtest(target_out, comp_out):
    B = 5000
    if len(target_out) != len(comp_out):
        minlen = int(min(len(target_out), len(comp_out)))
        print('Inconsistent numbers of outputs for significance test. Use {0} outputs.'.format(minlen))
        X = np.array(target_out[0:minlen])
        Y = np.array(comp_out[0:minlen])
    else:
        X = np.array(target_out)
        Y = np.array(comp_out)

    d = np.abs(np.mean(X) - np.mean(Y))
    cnt = 0
    for b in range(B):
        rand_X = np.zeros_like(X)
        rand_Y = np.zeros_like(Y)
        for i in range(X.size):
            if np.random.randint(0, 1.0e+12) % 2 == 0:
                rand_X[i] = X[i]
                rand_Y[i] = Y[i]
            else:
                rand_X[i] = Y[i]
                rand_Y[i] = X[i]
        new_d = np.abs(np.mean(rand_X) - np.mean(rand_Y))
        if new_d >= d:
            cnt += 1
    p = float(cnt) / float(B)

    return p


def sigtest_two_models():
    ##### Setting ###################
    out_path = '../out_ave_cted/out/BERT1E-CTED_vs_BERT1E-Naive.txt'
    model_x_dir = '../out_ave_cted/'
    model_x_name = 'BERT1E'
    model_y_dir = '../out_ave_wo_ted/'
    model_y_name = 'BERT1E'
    #################################

    model_x_paths = glob.glob(model_x_dir + model_x_name + '*_ALIR_ALIP*.txt')
    model_y_paths = glob.glob(model_y_dir + model_y_name + '*_ALIR_ALIP*.txt')
    x_norm_paths, x_pp_paths = read_outputs(model_x_name, model_x_paths)
    y_norm_paths, y_pp_paths = read_outputs(model_y_name, model_y_paths)

    x_alirs, x_alips, x_alifs = load(x_norm_paths)
    y_alirs, y_alips, y_alifs = load(y_norm_paths)

    p_alir = randomized_sigtest(x_alirs[1], y_alirs[1])
    p_alip = randomized_sigtest(x_alips[1], y_alips[1])
    p_alif = randomized_sigtest(x_alifs[1], y_alifs[1])

    with codecs.open(out_path, 'w', encoding='utf-8') as f:
        f.write('\tSig-test against {0} vs {1}\n'.format(model_x_name, model_y_name))
        f.write('\tALIR\tALIP\tALIF\n')
        f.write('\t{0:.5f}\t{1:.5f}\t{2:.5f}\n'.format(p_alir, p_alip, p_alif))


def read_outputs(model, paths):
    postprocess_paths = []
    normal_paths = []
    if model.startswith('BERT1F'):
        if '(fixed)' in model:
            for p in [path for path in paths if
                      os.path.basename(path).startswith('BERT1F') and 'ft-' not in os.path.basename(path)]:
                postprocess_paths.append(p) if 'postprocess' in p else normal_paths.append(p)
        else:
            for p in [path for path in paths if
                      os.path.basename(path).startswith('BERT1F') and 'ft-' in os.path.basename(path)]:
                postprocess_paths.append(p) if 'postprocess' in p else normal_paths.append(p)
    elif model.startswith('BERT1E'):
        if 'fixed' in model:
            for p in [path for path in paths if
                      os.path.basename(path).startswith('BERT1E') and 'ft-' not in os.path.basename(path)]:
                postprocess_paths.append(p) if 'postprocess' in p else normal_paths.append(p)
        else:
            for p in [path for path in paths if
                      os.path.basename(path).startswith('BERT1E') and 'ft-' in os.path.basename(path)]:
                postprocess_paths.append(p) if 'postprocess' in p else normal_paths.append(p)
    else:
        for p in [path for path in paths if os.path.basename(path).startswith(model)]:
            postprocess_paths.append(p) if 'postprocess' in p else normal_paths.append(p)

    return normal_paths, postprocess_paths


def comp_all_models():
    dir = '../out_ave_wo_ted/'
    out_dir = dir + 'out/'
    permute = None
    sigtest_target = 'BERT1F'
    # sigtest_target = 'BERT_MeanPool'
    # sigtest_target = 'bert-base-uncased'
    # sigtest_target = 'ELMo_MeanPool'

    models = set()
    paths = glob.glob(dir + '*_ALIR_ALIP*.txt')
    for path in paths:
        basename = os.path.basename(path)
        parts = basename.split('_')
        if parts[0] == 'ELMo' or parts[0] == 'ELMo-1' or parts[0] == 'FastText' or parts[0] == 'BERT':
            models.add(parts[0] + '_' + parts[1])
        elif parts[0] in ['BERT1F', 'BERT1E']:
            if 'ft-' in basename:
                models.add(parts[0])
            else:
                models.add(parts[0] + '(fixed)')
        else:
            models.add(parts[0])

    model_stats, model_peformances, model_stats_pp, model_peformances_pp = {}, {}, {}, {}
    for model in models:
        normal_paths, postprocess_paths = read_outputs(model, paths)
        alirs, alips, alifs = load(normal_paths)
        model_stats[model] = (
            mean_confidence_interval(alirs[0]), mean_confidence_interval(alips[0]), mean_confidence_interval(alifs[0]))
        model_peformances[model] = (alirs[1], alips[1], alifs[1])

        if len(postprocess_paths) > 0:
            alirs_pp, alips_pp, alifs_pp = load(postprocess_paths)
            model_stats_pp[model] = (mean_confidence_interval(
                alirs_pp[0]), mean_confidence_interval(alips_pp[0]), mean_confidence_interval(alifs_pp[0]))
            model_peformances_pp[model] = (alirs_pp[1], alips_pp[1], alifs_pp[1])

    for model in models:
        with codecs.open(out_dir + model + '_mean_peformance.txt', 'w', encoding='utf-8') as f:
            f.write('CTED:\n')
            f.write('\tALIR\t(min)\t(max)\tALIP\t(min)\t(max)\tALIF\t(min)\t(max)\n')
            f.write('\t{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.2f}\t{5:.2f}\t{6:.2f}\t{7:.2f}\t{8:.2f}\n'.format(
                model_stats[model][0][0], model_stats[model][0][1], model_stats[model][0][2], model_stats[model][1][0],
                model_stats[model][1][1], model_stats[model][1][2], model_stats[model][2][0], model_stats[model][2][1],
                model_stats[model][2][2]))
            if model != sigtest_target:
                p_alir = randomized_sigtest(model_peformances[sigtest_target][0], model_peformances[model][0])
                p_alip = randomized_sigtest(model_peformances[sigtest_target][1], model_peformances[model][1])
                p_alif = randomized_sigtest(model_peformances[sigtest_target][2], model_peformances[model][2])
                f.write('\tSig-test against {0}\n'.format(sigtest_target))
                f.write('\tALIR\tALIP\tALIF\n')
                f.write('\t{0:.5f}\t{1:.5f}\t{2:.5f}\n'.format(p_alir, p_alip, p_alif))

            f.write('\tLatex source:\n')
            f.write('\tALIR\tconf_interval\tALIP\tconf_interval\tALIF\tconf_interval\n')
            f.write('\t{0:.1f} \pm {1:.1f} & {2:.1f} \pm {3:.1f} & {4:.1f} \pm {5:.1f}\n'.format(
                model_stats[model][0][0], model_stats[model][0][0] - model_stats[model][0][1],
                model_stats[model][1][0], model_stats[model][1][0] - model_stats[model][1][1],
                model_stats[model][2][0], model_stats[model][2][0] - model_stats[model][2][1]))

            if len(postprocess_paths) > 0:
                f.write('With postprocess:\n')
                f.write('\tALIR\t(min)\t(max)\tALIP\t(min)\t(max)\tALIF\t(min)\t(max)\n')
                f.write('\t{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.2f}\t{5:.2f}\t{6:.2f}\t{7:.2f}\t{8:.2f}\n'.format(
                    model_stats_pp[model][0][0], model_stats_pp[model][0][1], model_stats_pp[model][0][2],
                    model_stats_pp[model][1][0], model_stats_pp[model][1][1], model_stats_pp[model][1][2],
                    model_stats_pp[model][2][0], model_stats_pp[model][2][1], model_stats_pp[model][2][2]))

                if model != sigtest_target:
                    p_alir = randomized_sigtest(model_peformances_pp[sigtest_target][0], model_peformances_pp[model][0])
                    p_alip = randomized_sigtest(model_peformances_pp[sigtest_target][1], model_peformances_pp[model][1])
                    p_alif = randomized_sigtest(model_peformances_pp[sigtest_target][2], model_peformances_pp[model][2])
                    f.write('\tSig-test against {0}\n'.format(sigtest_target))
                    f.write('\tALIR\tALIP\tALIF\n')
                    f.write('\t{0:.5f}\t{1:.5f}\t{2:.5f}\n'.format(p_alir, p_alip, p_alif))

                f.write('\tLatex source:\n')
                f.write('\tALIR\tconf_interval\tALIP\tconf_interval\tALIF\tconf_interval\n')
                f.write('\t{0:.1f} \pm {1:.1f} & {2:.1f} \pm {3:.1f} & {4:.1f} \pm {5:.1f}\n'.format(
                    model_stats_pp[model][0][0], model_stats_pp[model][0][0] - model_stats_pp[model][0][1],
                    model_stats_pp[model][1][0], model_stats_pp[model][1][0] - model_stats_pp[model][1][1],
                    model_stats_pp[model][2][0], model_stats_pp[model][2][0] - model_stats_pp[model][2][1]))


if __name__ == "__main__":
    # comp_all_models()

    sigtest_two_models()
