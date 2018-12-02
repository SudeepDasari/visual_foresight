import pickle
import matplotlib.pyplot as plt
import numpy as np

def make_plots(conf, dict=None, dir = None):
    if dict == None:
        dict = pickle.load(open(dir + '/data.pkl'))

    print('loaded')
    videos = dict['videos']

    I0_ts = videos['I0_ts']

    # num_exp = I0_t_reals[0].shape[0]
    num_ex = 4
    start_ex = 0
    num_rows = num_ex*len(list(videos.keys()))
    num_cols = len(I0_ts) + 1

    print('num_rows', num_rows)
    print('num_cols', num_cols)

    width_per_ex = 2.5

    standard_size = np.array([width_per_ex * num_cols, num_rows * 1.5])  ### 1.5
    figsize = (standard_size).astype(np.int)

    f, axarr = plt.subplots(num_rows, num_cols, figsize=figsize)

    print('start')
    for col in range(num_cols -1):
        row = 0
        for ex in range(start_ex, start_ex + num_ex, 1):
            for tag in list(videos.keys()):
                print('doing tag {}'.format(tag))
                if isinstance(videos[tag], tuple):
                    im = videos[tag][0][col]
                    score = videos[tag][1]
                    axarr[row, col].set_title('{:10.3f}'.format(score[col][ex]), fontsize=5)
                else:
                    im = videos[tag][col]

                h = axarr[row, col].imshow(np.squeeze(im[ex]), interpolation='none')

                if len(im.shape) == 3:
                    plt.colorbar(h, ax=axarr[row, col])
                axarr[row, col].axis('off')
                row += 1

    row = 0
    col = num_cols-1

    if 'I1' in dict:
        for ex in range(start_ex, start_ex + num_ex, 1):
            im = dict['I1'][ex]
            h = axarr[row, col].imshow(np.squeeze(im), interpolation='none')
            plt.colorbar(h, ax=axarr[row, col])
            axarr[row, col].axis('off')
            row += len(list(videos.keys()))

    # plt.axis('off')
    f.subplots_adjust(wspace=0, hspace=0.3)

    # f.subplots_adjust(vspace=0.1)
    # plt.show()
    plt.savefig(conf['output_dir']+'/warp_costs_{}.png'.format(dict['name']))