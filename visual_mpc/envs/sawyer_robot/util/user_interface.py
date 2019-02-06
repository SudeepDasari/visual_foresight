import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle


class Getdesig(object):
    def __init__(self, img, basedir, img_namesuffix = '', n_desig=1, only_desig = False,
                 im_shape = None, clicks_per_desig=2):
        plt.switch_backend('qt5agg')
        self.im_shape = im_shape

        self.only_desig = only_desig
        self.n_desig = n_desig
        self.suf = img_namesuffix
        self.basedir = basedir
        self.img = img
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        self.ax.set_xlim(0, self.im_shape[1])
        self.ax.set_ylim(self.im_shape[0], 0)
        plt.imshow(img)

        self.goal = None
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.i_click = 0

        self.desig = np.zeros((n_desig,2))  #idesig, (r,c)
        if clicks_per_desig == 2:
            self.goal = np.zeros((n_desig, 2))  # idesig, (r,c)
        else: self.goal = None

        self.i_click_max = n_desig * clicks_per_desig
        self.clicks_per_desig = clicks_per_desig

        self.i_desig = 0
        self.i_goal = 0
        self.marker_list = ['o',"D","v","^"]

        plt.show()

    def onclick(self, event):
        print(('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata)))
        import matplotlib.pyplot as plt
        self.ax.set_xlim(0, self.im_shape[1])
        self.ax.set_ylim(self.im_shape[0], 0)

        print('iclick', self.i_click)

        i_task = self.i_click//self.clicks_per_desig
        print('i_task', i_task)

        if self.i_click == self.i_click_max:
            print('saving desig-goal picture')

            with open(self.basedir +'/{}_pix.pkl'.format(self.suf), 'wb') as f:
                dict= {'desig_pix': self.desig,
                       'goal_pix': self.goal}
                pickle.dump(dict, f)

            plt.savefig(self.basedir + '/img_' + self.suf)
            print('closing')
            plt.close()
            return

        rc_coord = np.array([event.ydata, event.xdata])

        if self.i_click % self.clicks_per_desig == 0:
            self.desig[i_task, :] = rc_coord
            color = "r"
        else:
            self.goal[i_task, :] = rc_coord
            color = "g"
        marker = self.marker_list[i_task]
        self.ax.scatter(rc_coord[1], rc_coord[0], s=100, marker=marker, facecolors=color)

        plt.draw()

        self.i_click += 1


def select_points(cam_images, cam_names, fig_name, fig_save_dir, clicks_per_desig = 2, n_desig = 1):
    assert clicks_per_desig == 1 or clicks_per_desig == 2, "CLICKS_PER_DESIG SHOULD BE 1 OR 2"

    start_pix = []
    if clicks_per_desig == 2:
        goal_pix = []
    for img, cam in zip(cam_images, cam_names):
        img_height, img_width = img.shape[:2]
        c_main = Getdesig(img, fig_save_dir, '{}{}'.format(fig_name, cam), n_desig=n_desig,
                          im_shape=[img_height, img_width], clicks_per_desig=clicks_per_desig)

        start_pos = c_main.desig.astype(np.int64)
        start_pix.append(start_pos.reshape(1, n_desig, 2))

        if clicks_per_desig == 2:
            goal_pos = c_main.goal.astype(np.int64)
            goal_pix.append(goal_pos.reshape(1, n_desig, 2))

    start_pix = np.concatenate(start_pix, 0)
    if clicks_per_desig == 2:
        goal_pix = np.concatenate(goal_pix, 0)
        return start_pix, goal_pix
    return start_pix
