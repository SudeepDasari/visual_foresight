import matplotlib; matplotlib.use('Agg')
from collections import OrderedDict
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import re
import pdb


def write_scores(conf, result_file, stat, i_traj=None):
    improvement = stat['improvement']

    final_dist = stat['final_dist']
    if 'initial_dist' in stat:
        initial_dist = stat['initial_dist']
    else: initial_dist = None

    if 'term_t' in stat:
        term_t = stat['term_t']

    sorted_ind = improvement.argsort()[::-1]

    if i_traj == None:
        i_traj = improvement.shape[0]

    mean_imp = np.mean(improvement)
    med_imp = np.median(improvement)
    mean_dist = np.mean(final_dist)
    med_dist = np.median(final_dist)

    if 'lifted' in stat:
        lifted = stat['lifted'].astype(np.int)
    else: lifted = np.zeros_like(improvement)

    print('mean imp, med imp, mean dist, med dist {}, {}, {}, {}\n'.format(mean_imp, med_imp, mean_dist, med_dist))

    f = open(result_file, 'w')
    if 'term_dist' in conf['agent']:
        tlen = conf['agent']['T']
        nsucc_frac = np.where(term_t != (tlen - 1))[0].shape[0]/ improvement.shape[0]
        f.write('percent success: {}%\n'.format(nsucc_frac * 100))
        f.write('---\n')
    if 'lifted' in stat:
        f.write('---\n')
        f.write('fraction of traj lifted: {0}\n'.format(np.mean(lifted)))
        f.write('---\n')
    f.write('standard error of the mean (SEM) {0}\n'.format(np.std(final_dist) / np.sqrt(final_dist.shape[0])))
    f.write('---\n')
    f.write('overall best pos improvement: {0} of traj {1}\n'.format(improvement[sorted_ind[0]], sorted_ind[0]))
    f.write('overall worst pos improvement: {0} of traj {1}\n'.format(improvement[sorted_ind[-1]], sorted_ind[-1]))
    f.write('average pos improvemnt: {0}\n'.format(mean_imp))
    f.write('median pos improvement {}'.format(med_imp))
    f.write('standard deviation of population {0}\n'.format(np.std(improvement)))
    f.write('standard error of the mean (SEM) {0}\n'.format(np.std(improvement) / np.sqrt(improvement.shape[0])))
    f.write('---\n')
    f.write('average pos score: {0}\n'.format(mean_dist))
    f.write('median pos score {}'.format(med_dist))
    f.write('standard deviation of population {0}\n'.format(np.std(final_dist)))
    f.write('standard error of the mean (SEM) {0}\n'.format(np.std(final_dist) / np.sqrt(final_dist.shape[0])))
    f.write('---\n')
    f.write('mean imp, med imp, mean dist, med dist {}, {}, {}, {}\n'.format(mean_imp, med_imp, mean_dist, med_dist))
    f.write('---\n')
    if initial_dist is not None:
        f.write('average initial dist: {0}\n'.format(np.mean(initial_dist)))
        f.write('median initial dist: {0}\n'.format(np.median(initial_dist)))
        f.write('----------------------\n')
    f.write('traj: improv, final_d, term_t, lifted, rank\n')
    f.write('----------------------\n')

    for n, t in enumerate(range(conf['start_index'], i_traj)):
        f.write('{}: {}, {}:{}\n'.format(t, improvement[n], final_dist[n], np.where(sorted_ind == n)[0][0]))
    f.close()


def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


def combine_scores(conf, dir, only_first_n=None):
    files = glob.glob(dir + '/scores_*')
    files = sorted_nicely(files)
    if len(files) == 0:
        raise ValueError

    stats_lists = OrderedDict()

    for f in files:
        print('load', f)
        dict_ = pickle.load(open(f, "rb"))
        for key in dict_.keys():
            if key not in stats_lists:
                stats_lists[key] = []
            stats_lists[key].append(dict_[key])

    pdb.set_trace()
    stat_array = OrderedDict()
    for key in dict_.keys():
        stat_array[key] = np.concatenate(stats_lists[key], axis=0)

    improvement = stat_array['improvement']
    final_dist = stat_array['final_dist']
    if only_first_n is not None:
        improvement = improvement[:only_first_n]
        final_dist = final_dist[:only_first_n]

    make_stats(dir, final_dist, 'finaldist', bounds=[0., 0.5])
    make_stats(dir, improvement, 'improvement', bounds=[-0.5, 0.5])
    make_imp_score(final_dist, improvement, dir)

    write_scores(conf, dir + '/results_all.txt', stat_array)
    print('writing {}'.format(dir))


def make_imp_score(score, imp, dir):
    plt.scatter(imp, score)
    plt.xlabel('improvement')
    plt.ylabel('final distance')
    plt.savefig(dir + '/imp_vs_dist.png')


def make_stats(dir, score, name, bounds):
    bin_edges = np.linspace(bounds[0], bounds[1], 11)
    binned_ind = np.digitize(score, bin_edges)
    occurrence, _ = np.histogram(score, bin_edges, density=False)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_mid = bin_edges + bin_width / 2
    plt.figure()
    plt.bar(bin_mid[:-1], occurrence, bin_width, facecolor='b', alpha=0.5)
    plt.title(name)
    plt.xlabel(name)
    plt.ylabel('occurences')
    plt.savefig(dir + '/' + name + '.png')
    plt.close()
    f = open(dir + '/{}_histo.txt'.format(name), 'w')
    for i in range(bin_edges.shape[0]-1):
        f.write('indices for bin {}, {} to {} : {} \n'.format(i, bin_edges[i], bin_edges[i+1], np.where(binned_ind == i+1)[0].tolist()))

