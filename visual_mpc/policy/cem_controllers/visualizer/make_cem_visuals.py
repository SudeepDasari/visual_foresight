import numpy as np
import os
import collections
import pickle
from .render_utils import add_crosshairs, resize_image, draw_text_onimage, get_score_images, make_direct_vid
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt


def image_addgoalpix(bsize, seqlen, image_l, goal_pix):
    goal_pix_ob = np.tile(goal_pix[None, None, :], [bsize, seqlen, 1])
    return add_crosshairs(image_l, goal_pix_ob)


def write_tradeoff_onimage(image, tradeoff_percam, ntask, startgoal):
    """
    :param tradeoff_percam:
    :param ntask:
    :param startgoal:  0 or 1; 0 stands for startimage
    :return:
    """
    tradeoff_percam = tradeoff_percam.reshape([ntask,2])
    string = ','.join(['%.2f' %  tr for  tr in list(tradeoff_percam[:,startgoal])])
    return draw_text_onimage(string, image)


def upsample_if_necessary(dict, target_shape):
    for k in dict.keys():
        if dict[k].shape[3] != target_shape[0]:
            dict[k] = resize_image(dict[k][:,:,None], target_shape)[:,:,0]
    return dict


def annotate_tracks(vd, current_image, icam, len_pred, num_ex):
    ipix = 0
    for p in range(vd.ntask):
        if 'start' in vd.hp.register_gtruth:
            desig_pix_start = np.tile(vd.desig_pix[icam, ipix][None, None, :], [num_ex, len_pred, 1])
            current_image = add_crosshairs(current_image, desig_pix_start, color=[1., 0., 0])
            ipix += 1
        if 'goal' in vd.hp.register_gtruth:
            desig_pix_goal = np.tile(vd.desig_pix[icam, ipix][None, None, :], [num_ex, len_pred, 1])
            current_image = add_crosshairs(current_image, desig_pix_goal, color=[0, 0, 1.])
            ipix += 1
    return current_image


def compute_overlay(images, color_coded_dist):
    alpha = .6
    return color_coded_dist*alpha + (1-alpha)*images


def color_code(inp, num_ex, renormalize=False):
    out_distrib = []
    for t in range(inp.shape[1]):
        distrib = inp[:, t]
        out_t = []

        for b in range(num_ex):
            # cmap = plt.cm.get_cmap('jet')
            cmap = plt.cm.get_cmap('viridis')
            if renormalize:
                distrib[b] /= (np.max(distrib[b])+1e-6)
            colored_distrib = cmap(np.squeeze(distrib[b]))[:, :, :3]
            out_t.append(colored_distrib)

        out_t = np.stack(out_t, 0)
        out_distrib.append(out_t)
    return np.stack(out_distrib, 1)


class CEM_Visual_Preparation(object):
    def __init__(self, **kwargs):
        pass

    def visualize(self, vd):
        """
        :param vd:  visualization data
        :return:
        """

        bestindices = vd.scores.argsort()[:vd.K]
        self.ncam = vd.netconf['ncam']
        self.ndesig = vd.netconf['ndesig']
        self.agentparams = vd.agentparams
        self.hp = vd.hp
        print('in make_cem_visuals')
        plt.switch_backend('agg')

        if self.hp.visualize_best:
            selindices = bestindices
        else:
            selindices = list(range(vd.K))

        self.selindices = selindices
        gen_distrib = vd.gen_distrib[selindices]
        gen_images = vd.gen_images[selindices]
        print('selected distributions')
        if vd.agentparams['image_height'] != vd.image_height:
            gen_distrib = resize_image(gen_distrib, vd.goal_image.shape[1:3])
            gen_images = resize_image(gen_images, vd.goal_image.shape[1:3])
            print('resized images')

        self.num_ex = selindices.shape[0]
        self.len_pred = vd.netconf['sequence_length'] - vd.netconf['context_frames']

        print('made directories')
        self._t_dict = collections.OrderedDict()

        self.annontate_images(vd, vd.last_frames)

        self.visualize_goal_pixdistrib(vd, gen_distrib)

        for icam in range(self.ncam):
            print('putting cam: {} res into dict'.format(icam))

            gen_images_ = gen_images[:, :, icam]
            for p in range(self.ndesig):
                if vd.agentparams['image_height'] != vd.image_height:
                    goal_pix = vd.goal_pix_med[icam, p]
                else:
                    goal_pix = vd.goal_pix[icam, p]
                
                # gen_images_= image_addgoalpix(self.num_ex, self.len_pred, gen_images_, goal_pix)

            self._t_dict['gen_images_icam{}_t{}'.format(icam, vd.t)] = gen_images_

        print('itr{} best scores: {}'.format(vd.cem_itr, [vd.scores[selindices[ind]] for ind in range(self.num_ex)]))
        self._t_dict['scores'] = get_score_images(vd.scores[selindices], vd.last_frames.shape[3], vd.last_frames.shape[4], self.len_pred, self.num_ex)

        if 'no_instant_gif' not in vd.agentparams:

            self._t_dict = upsample_if_necessary(self._t_dict,  [vd.agentparams['image_height'], vd.agentparams['image_width']])
            make_direct_vid(self._t_dict, self.num_ex, vd.agentparams['record'] + '/plan/',
                                suf='t{}iter{}'.format(vd.t, vd.cem_itr))

        # make_action_summary(self.num_ex, actions, agentparams, selindices, cem_itr, netconf['sequence_length'], t)

        if 'save_pkl' in vd.agentparams:
            dir = vd.agentparams['record'] + '/plan'
            if not os.path.exists(dir):
                os.makedirs(dir)
            pickle.dump(self._t_dict, open(dir + '/pred_t{}iter{}.pkl'.format(vd.t, vd.cem_itr), 'wb'))
            print('written files to:', dir)

    def visualize_goal_pixdistrib(self, vd, gen_distrib):
        for icam in range(self.ncam):
            for p in range(self.ndesig):
                gen_distrib_ann = color_code(gen_distrib[:, :, icam, :, :, p], self.num_ex, renormalize=True)
                gen_distrib_ann = image_addgoalpix(self.num_ex, self.len_pred, gen_distrib_ann,
                                                   vd.goal_pix[icam, p])
                self._t_dict['gen_distrib_cam{}_p{}'.format(icam, p)] = gen_distrib_ann

    def annontate_images(self, vd, last_frames):
        for icam in range(self.ncam):
            current_image = np.tile(last_frames[0, 1, icam][None, None], [self.num_ex, self.len_pred, 1, 1, 1, 1])
            self._t_dict['curr_img_cam{}'.format(icam)] = current_image.squeeze()


class CEM_Visual_Preparation_Registration(CEM_Visual_Preparation):
    def annontate_images(self, vd, last_frames):
        for icam in range(self.ncam):
            print('annotating tracks for cam: {}'.format(icam))
            current_image = np.tile(last_frames[0, 1, icam][None, None], [self.num_ex, self.len_pred, 1, 1, 1, 1])
            current_image = annotate_tracks(vd, current_image.squeeze(), icam, self.len_pred, self.num_ex)
            self._t_dict['curr_img_cam{}'.format(icam)] = current_image.squeeze()

        self.visualize_registration(vd)

    def visualize_goal_pixdistrib(self, vd, gen_distrib):
        
        for icam in range(self.ncam):
            for p in range(self.ndesig):
                sel_gen_distrib_p = gen_distrib[:, :, icam, :, :, p]
                color_coded_dist = color_code(sel_gen_distrib_p, self.num_ex, renormalize=True)

                if vd.agentparams['image_height'] != vd.image_height:
                    goal_pix = vd.goal_pix_med[icam, p]
                else:
                    goal_pix = vd.goal_pix[icam, p]
                color_coded_dist = image_addgoalpix(self.num_ex, self.len_pred, color_coded_dist,
                                                   goal_pix)

                # plt.switch_backend('TkAgg')
                # plt.imshow()
                # plt.show()

                self._t_dict['gen_distrib_cam{}_p{}'.format(icam, p)] = color_coded_dist
                self._t_dict['gen_dist_goalim_overlay_cam{}_p{}'.format(icam, p)] = \
                compute_overlay(self.gl_im_ann_per_tsk[p, :, :, icam], color_coded_dist)

    def visualize_registration(self, vd):
        pix_mult = self.agentparams['image_height']/vd.image_height

        for icam in range(self.ncam):
            print("on cam: {}".format(icam))
            if 'start' in self.hp.register_gtruth:
                print('on start case')
                if self.hp.trade_off_reg:
                    warped_img_start_cam = write_tradeoff_onimage(vd.warped_image_start[icam].squeeze(), vd.reg_tradeoff[icam],
                                                                  vd.ntask, 0)
                else:
                    warped_img_start_cam = vd.warped_image_start[icam].squeeze()
                self._t_dict['warp_start_cam{}'.format(icam)] = np.repeat(np.repeat(warped_img_start_cam[None], self.len_pred, axis=0)[None], self.num_ex, axis=0)

            startimages = np.tile(vd.start_image[icam][None, None], [self.num_ex, self.len_pred, 1, 1, 1])
            for p in range(vd.ntask):
                print('on task {}'.format(p))
                if vd.agentparams['image_height'] != vd.image_height:
                    desig_pix_t0 = vd.desig_pix_t0_med[icam, p][None]
                else:
                    desig_pix_t0 = vd.desig_pix_t0[icam, p][None]
                
                desig_pix_t0 = np.tile(desig_pix_t0, [self.num_ex, self.len_pred, 1])

                startimages = add_crosshairs(startimages, desig_pix_t0)
            self._t_dict['start_img_cam{}'.format(icam)] = startimages

            for p in range(vd.ntask):
                if 'goal' in self.hp.register_gtruth:
                    print('on goal case cam: {}'.format(p))
                    if self.hp.trade_off_reg:
                        warped_img_goal_cam = write_tradeoff_onimage(vd.warped_image_goal[icam].squeeze(), vd.reg_tradeoff[icam], vd.ntask, 1)
                    else:
                        warped_img_goal_cam = vd.warped_image_goal[icam].squeeze()
                    self._t_dict['warp_goal_cam{}'.format(icam)] = np.repeat(np.repeat(warped_img_goal_cam[None], self.len_pred, axis=0)[None], self.num_ex, axis=0)

        if vd.agentparams['image_height'] != vd.image_height:
            goal_pix = vd.goal_pix_med
        else:
            goal_pix = vd.goal_pix

        gl_im_shape = [self.num_ex, self.len_pred, vd.ncam] + list(vd.goal_image.shape[1:])
        gl_im_ann = np.zeros(gl_im_shape)  # b, t, n, r, c, 3
        self.gl_im_ann_per_tsk = np.zeros([vd.ndesig] + gl_im_shape)  # p, b, t, n, r, c, 3
        for icam in range(vd.ncam):
            print('adding goal pixes {}'.format(icam))
            gl_im_ann[:, :, icam] = np.tile(vd.goal_image[icam][None, None], [self.num_ex, self.len_pred, 1, 1, 1])
            self.gl_im_ann_per_tsk[:, :, :, icam] = np.tile(vd.goal_image[icam][None, None, None],
                                                       [vd.ndesig, self.num_ex, self.len_pred, 1, 1, 1])
            for p in range(vd.ndesig):
                gl_im_ann[:, :, icam] = image_addgoalpix(self.num_ex, self.len_pred, gl_im_ann[:, :, icam],
                                                         goal_pix[icam, p] * pix_mult)
                self.gl_im_ann_per_tsk[p, :, :, icam] = image_addgoalpix(self.num_ex, self.len_pred, self.gl_im_ann_per_tsk[p][:, :, icam],
                                                                    goal_pix[icam, p])
            self._t_dict['goal_image{}'.format(icam)] = gl_im_ann[:, :, icam]


class CEM_Visual_Preparation_FullImageReg(CEM_Visual_Preparation):
    def annontate_images(self, vd, last_frames):
        self.visualize_registration(vd)

    def visualize_registration(self, vd):
        for icam in range(self.ncam):
            self._t_dict['reg_cam{}'.format(icam)] = vd.warped_images[self.selindices,:,icam]

        for icam in range(vd.ncam):
            if vd.goal_image.shape[0] != 1:  # if the complete trajectory is registered
                goal_image = np.tile(vd.goal_image[:,icam][None, :], [self.num_ex,  1, 1, 1, 1])
            else:
                goal_image = np.tile(vd.goal_image[:,icam][None, :], [self.num_ex, self.len_pred, 1, 1, 1])
            self._t_dict['goal_image_cam{}'.format(icam)] = goal_image
            self._t_dict['flow_mags_cam{}'.format(icam)] = color_code(vd.flow_mags[self.selindices,:,icam], self.num_ex, renormalize=True)

    def visualize_goal_pixdistrib(self, vd, gen_distrib):
        pass


class CEM_Visual_Preparation_FullImage(CEM_Visual_Preparation):
    def annontate_images(self, vd, last_frames):
        for icam in range(vd.ncam):
            if vd.goal_image.shape[0] != 1:  # if the complete trajectory is registered
                goal_image = np.tile(vd.goal_image[:,icam][None, :], [self.num_ex,  1, 1, 1, 1])
            else:
                goal_image = np.tile(vd.goal_image[:,icam][None, :], [self.num_ex, self.len_pred, 1, 1, 1])
            self._t_dict['goal_image_cam{}'.format(icam)] = goal_image

    def visualize_goal_pixdistrib(self, vd, gen_distrib):
        pass
