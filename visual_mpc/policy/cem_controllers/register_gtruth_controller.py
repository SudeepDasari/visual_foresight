from .pixel_cost_controller import PixelCostController
import copy
import numpy as np
from .visualizer.render_utils import resize_image
from .visualizer.make_cem_visuals import CEM_Visual_Preparation_Registration
import imp
from visual_mpc.registration_network.setup_registration import setup_gdn


class Register_Gtruth_Controller(PixelCostController):
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        super(Register_Gtruth_Controller, self).__init__(ag_params, policyparams, gpu_id, ngpu)
        
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)

        self.reg_tradeoff = np.ones([self._n_cam, self._n_desig]) / self._n_cam / self._n_desig

        params = imp.load_source('params', ag_params['current_dir'] + '/gdnconf.py')
        self.gdnconf = params.configuration
        self.goal_image_warper = setup_gdn(self.gdnconf, gpu_id)

        num_reg_images = len(self._hp.register_gtruth)

        self.ntask = self._n_desig // num_reg_images

        self.visualizer = CEM_Visual_Preparation_Registration()

    def _default_hparams(self):
        default_dict = {
            'register_gtruth':['start','goal'],
            'register_region':False,
            # 'trade_off_reg':True

        }
        parent_params = super(Register_Gtruth_Controller, self)._default_hparams()

        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def _prep_vidpred_inp(self, actions, cem_itr):
        actions, last_frames, last_states, t_0 = super(Register_Gtruth_Controller, self)._prep_vidpred_inp(actions, cem_itr)
        if 'image_medium' in self.agentparams:  # downsample to video-pred reslution
            last_frames = resize_image(last_frames, (self._img_height, self._img_width))
        if self._hp.register_gtruth and cem_itr == 0:
            self.start_image = copy.deepcopy(self.images[0]).astype(np.float32) / 255.
            self.warped_image_start, self.warped_image_goal, self.reg_tradeoff = self.register_gtruth(self.start_image,
                                                                                                      last_frames)
        if self.agentparams['image_height'] != self._img_height:  # downsample to video-pred reslution
            last_frames = resize_image(last_frames, (self._img_height, self._img_width))
        return actions, last_frames, last_states, t_0

    def register_gtruth(self,start_image, last_frames):
        """
        :param start_image:
        :param last_frames:
        :param goal_image:
        :return:  returns tradeoff with shape: ncam, ndesig
        """
        last_frames = last_frames[0, self._net_context - 1]

        desig_pix_l, warperrs_l = [], []
        warped_image_start, _, start_warp_pts = self.goal_image_warper(last_frames[None], start_image[None])
        if 'goal' in self._hp.register_gtruth:
            warped_image_goal, _, goal_warp_pts = self.goal_image_warper(last_frames[None], self.goal_image[None])

        imheight, imwidth = self.goal_image.shape[1:3]
        for n in range(self._n_cam):
            start_warp_pts = start_warp_pts.reshape(self._n_cam, imheight, imwidth, 2)
            warped_image_start = warped_image_start.reshape(self._n_cam, imheight, imwidth, 3)
            if 'goal' in self._hp.register_gtruth:
                goal_warp_pts = goal_warp_pts.reshape(self._n_cam, imheight, imwidth, 2)
                warped_image_goal = warped_image_goal.reshape(self._n_cam, imheight, imwidth, 3)
            else:
                goal_warp_pts = None
                warped_image_goal = None

            
            warperr, desig_pix = self.get_warp_err(n, start_image, self.goal_image, start_warp_pts, goal_warp_pts, warped_image_start, warped_image_goal)
            warperrs_l.append(warperr)
            desig_pix_l.append(desig_pix)

        self.desig_pix = np.stack(desig_pix_l, axis=0).reshape(self._n_cam, self._n_desig, 2)

        warperrs = np.stack(warperrs_l, 0)    # shape: ncam, ntask, r

        tradeoff = (1 / warperrs)
        normalizers = np.sum(np.sum(tradeoff, 0, keepdims=True), 2, keepdims=True)
        tradeoff = tradeoff / normalizers
        tradeoff = tradeoff.reshape(self._n_cam, self._n_desig)

        self.plan_stat['tradeoff'] = tradeoff
        self.plan_stat['warperrs'] = warperrs.reshape(self._n_cam, self._n_desig)

        # fill visualzation data object:
        self.vd.reg_tradeoff = tradeoff
        self.vd.ntask = self.ntask
        self.vd.warped_image_start = warped_image_start
        self.vd.warped_image_goal = warped_image_goal

        
        self.vd.desig_pix_t0_med = self.desig_pix_t0_med
        self.vd.goal_pix_med = self.goal_pix_med
        self.vd.desig_pix_t0 = self.desig_pix_t0
        self.vd.desig_pix = self.desig_pix
        self.vd.start_image = self.start_image
        self.vd.goal_image = self.goal_image
        self.vd.image_height = self._img_height

        return warped_image_start, warped_image_goal, tradeoff

    def get_warp_err(self, icam, start_image, goal_image, start_warp_pts, goal_warp_pts, warped_image_start, warped_image_goal):
        nreg = len(self._hp.register_gtruth)
        warperrs = np.zeros((self.ntask, nreg))
        desig = np.zeros((self.ntask, nreg, 2))

        region_tradeoff = True

        for p in range(self.ntask):
            if self.agentparams['image_height'] != self._img_height:
                
                pix_t0 = self.desig_pix_t0_med[icam, p]
                goal_pix = self.goal_pix_med[icam, p]
                print('using desig goal pix medium')
            else:
                pix_t0 = self.desig_pix_t0[icam, p]     # desig_pix_t0 shape: icam, ndesig, 2
                goal_pix = self.goal_pix_sel[icam, p]

            if not self._hp.register_region:
                if 'start' in self._hp.register_gtruth:
                    desig[p, 0] = np.flip(start_warp_pts[icam][pix_t0[0], pix_t0[1]], 0)

                if 'goal' in self._hp.register_gtruth:
                    desig[p, 1] = np.flip(goal_warp_pts[icam][goal_pix[0], goal_pix[1]], 0)
            else:
                # taking region of 2*width+1 around the designated pixel and computing the median flow vector for x and y

                if self.agentparams['image_height'] >= 96:
                    width = 5
                else: width = 2


                r_range = np.clip(np.array((pix_t0[0]-width,pix_t0[0]+width+1)), 0, self.agentparams['image_height']-1)
                c_range = np.clip(np.array((pix_t0[1]-width,pix_t0[1]+width+1)), 0, self.agentparams['image_width']-1)

                if region_tradeoff:
                    warperrs[p, 0] = np.mean(np.square(start_image[icam][r_range[0]:r_range[1], c_range[0]:c_range[1]] - warped_image_start[icam][r_range[0]:r_range[1], c_range[0]:c_range[1]]))

                point_field = start_warp_pts[icam][r_range[0]:r_range[1], c_range[0]:c_range[1]]
                desig[p, 0] = np.flip(np.array([np.median(point_field[:,:,0]), np.median(point_field[:,:,1])]), axis=0)


                r_range = np.clip(np.array((goal_pix[0]-width,goal_pix[0]+width+1)), 0, self.agentparams['image_height'])
                c_range = np.clip(np.array((goal_pix[1]-width,goal_pix[1]+width+1)), 0, self.agentparams['image_width'])

                if region_tradeoff:
                    warperrs[p, 1] = np.mean(np.square(goal_image[icam][r_range[0]:r_range[1], c_range[0]:c_range[1]] - warped_image_goal[icam][r_range[0]:r_range[1], c_range[0]:c_range[1]]))

                point_field = goal_warp_pts[icam][r_range[0]:r_range[1], c_range[0]:c_range[1]]
                desig[p, 1] = np.flip(np.array([np.median(point_field[:,:,0]), np.median(point_field[:,:,1])]), axis=0)

            if not region_tradeoff:
                if 'start' in self._hp.register_gtruth:
                    warperrs[p, 0] = np.linalg.norm(start_image[icam][pix_t0[0], pix_t0[1]] -
                                                    warped_image_start[icam][pix_t0[0], pix_t0[1]])

                if 'goal' in self._hp.register_gtruth:
                    warperrs[p, 1] = np.linalg.norm(goal_image[icam][goal_pix[0], goal_pix[1]] -
                                                    warped_image_goal[icam][goal_pix[0], goal_pix[1]])

        desig = desig * self._img_height / self.agentparams['image_height']
        return warperrs, desig

    def act(self,goal_image=None, t=None, i_tr=None, desig_pix=None, goal_pix=None, images=None, state=None):

        num_reg_images = len(self._hp.register_gtruth)

        self.goal_pix_sel = np.array(goal_pix).reshape((self._n_cam, self.ntask, 2))
        self.goal_pix = np.tile(self.goal_pix_sel[:,:,None,:], [1,1,num_reg_images,1])  # copy along r: shape: ncam, ntask, r
        self.goal_pix = self.goal_pix.reshape(self._n_cam, self._n_desig, 2)
        print('regvidpred received goalpix', self.goal_pix)
        self.goal_pix_med = (self.goal_pix * self.agentparams['image_height'] / self._img_height).astype(np.int)

        

        self.goal_image = goal_image[-1]

        if t == 0:
            self.desig_pix_t0 = np.array(desig_pix).reshape((self._n_cam, self.ntask, 2))   # 1,1,2
            self.desig_pix_t0_med = (self.desig_pix_t0 * self.agentparams['image_height'] / self._img_height).astype(np.int)

        self.images = images
        self.state = state
        return super(PixelCostController, self).act(t, i_tr)
