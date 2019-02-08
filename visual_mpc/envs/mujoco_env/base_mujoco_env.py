from mujoco_py import load_model_from_path, MjSim
import numpy as np
from visual_mpc.envs.base_env import BaseEnv


class BaseMujocoEnv(BaseEnv):
    def __init__(self, model_path, _hp):
        self._frame_height = _hp.viewer_image_height
        self._frame_width = _hp.viewer_image_width

        self._model_path = model_path
        self.sim = MjSim(load_model_from_path(self._model_path))

        self._base_adim, self._base_sdim = None, None                 #state/action dimension of Mujoco control
        self._adim, self._sdim = None, None   #state/action dimension presented to agent
        self.num_objects, self._n_joints = None, None
        self._goal_obj_pose = None
        self._goaldistances = []

        self._ncam = _hp.ncam
        self.cameras = ['cam{}'.format(i) for i in range(self._ncam)]

        self._last_obs = None
        self._hp = _hp

        self._save_buffer = []
    
    def _default_hparams(self):
        parent_params = super()._default_hparams()
        parent_params.add_hparam('viewer_image_height', 480)
        parent_params.add_hparam('viewer_image_width', 640)
        parent_params.add_hparam('ncam', 1)

        return parent_params

    def set_goal_obj_pose(self, pose):
        self._goal_obj_pose = pose

    def _reset_eval(self):
        if self._goal_obj_pose is not None:
            self._goaldistances = [self.get_distance_score()]

    def reset(self):
        self._save_buffer = []

    def render(self):
        """ Renders the enviornment.
        Implements custom rendering support. If mode is:

        - dual: renders both left and main cameras
        - left: renders only left camera
        - main: renders only main (front) camera
        :param mode: Mode to render with (dual by default)
        :return: uint8 numpy array with rendering from sim
        """
        images = np.zeros((self._ncam, self._frame_height, self._frame_width, 3), dtype=np.uint8)
        for i, cam in enumerate(self.cameras):
            images[i] = self.sim.render(self._frame_width, self._frame_height, camera_name=cam)
        self._append_save_buffer(images[0])
        return images

    def _append_save_buffer(self, img):
        self._save_buffer.append(img.copy())

    def project_point(self, point, camera):
        model_matrix = np.zeros((4, 4))
        model_matrix[:3, :3] = self.sim.data.get_camera_xmat(camera).T
        model_matrix[-1, -1] = 1

        fovy_radians = np.deg2rad(self.sim.model.cam_fovy[self.sim.model.camera_name2id(camera)])
        uh = 1. / np.tan(fovy_radians / 2)
        uw = uh / (self._frame_width / self._frame_height)
        extent = self.sim.model.stat.extent
        far, near = self.sim.model.vis.map.zfar * extent, self.sim.model.vis.map.znear * extent
        view_matrix = np.array([[uw, 0., 0., 0.],                        # matrix definition from
                                [0., uh, 0., 0.],                        # https://stackoverflow.com/questions/18404890/how-to-build-perspective-projection-matrix-no-api
                                [0., 0., far / (far - near), -1.],
                                [0., 0., -2*far*near/(far - near), 0.]]) # Note Mujoco doubles this quantity

        MVP_matrix = view_matrix.dot(model_matrix)
        world_coord = np.ones((4, 1))
        world_coord[:3, 0] = point - self.sim.data.get_camera_xpos(camera)

        clip = MVP_matrix.dot(world_coord)
        ndc = clip[:3] / clip[3]  # everything should now be in -1 to 1!!
        col, row = (ndc[0] + 1) * self._frame_width / 2, (-ndc[1] + 1) * self._frame_height / 2

        return self._frame_height - row, col                 # rendering flipped around in height

    def get_desig_pix(self, target_width, round=True, obj_poses = None):
        qpos_dim = self._n_joints      # the states contains pos and vel
        assert self.sim.data.qpos.shape[0] == qpos_dim + 7 * self.num_objects
        desig_pix = np.zeros([self._ncam, self.num_objects, 2], dtype=np.int)
        ratio = self._frame_width / target_width
        for icam, cam in enumerate(self.cameras):
            for i in range(self.num_objects):
                if obj_poses is None:
                    fullpose = self.sim.data.qpos[i * 7 + qpos_dim:(i + 1) * 7 + qpos_dim].squeeze()
                    chosen_point = fullpose[:3]
                else:
                    chosen_point = obj_poses[i, :3]
                d = self.project_point(chosen_point, cam)
                d = np.stack(d) / ratio
                if round:
                    d = np.around(d).astype(np.int)
                desig_pix[icam, i] = d.squeeze()
        return desig_pix

    def get_goal_pix(self, target_width, round=True):
        goal_pix = np.zeros([self._ncam, self.num_objects, 2], dtype=np.int)
        ratio = self._frame_width / target_width
        for icam, cam in enumerate(self.cameras):
            for i in range(self.num_objects):
                g = self.project_point(self._goal_obj_pose[i, :3], cam)
                g = np.stack(g) / ratio
                if round:
                    g= np.around(g).astype(np.int)
                goal_pix[icam, i] = g.squeeze()
        return goal_pix

    def eval(self, target_width=None, save_dir=None, ntasks=None):
        self._goaldistances.append(self.get_distance_score())
        stats = {}
        stats['improvement'] = self._goaldistances[0] - self._goaldistances[-1]
        stats['initial_dist'] = self._goaldistances[0]
        stats['final_dist'] = self._goaldistances[-1]
        return stats

    def get_distance_score(self):
        """
        :return:  mean of the distances between all objects and goals
        """
        abs_distances = []
        for i_ob in range(self.num_objects):
            goal_pos = self._goal_obj_pose[i_ob, :3]
            curr_pos = self.sim.data.qpos[self._n_joints + i_ob * 7: self._n_joints + 3 + i_ob * 7]
            abs_distances.append(np.linalg.norm(goal_pos - curr_pos))
        return np.mean(np.array(abs_distances))

    @property
    def adim(self):
        return self._adim

    @property
    def sdim(self):
        return self._sdim

    @property
    def ncam(self):
        return self._ncam

    def generate_task(self):
        raise NotImplementedError

    def save_recording(self, save_worker, i_traj):
        if len(self._save_buffer):
            save_worker.put(('mov', 'traj_{}.gif'.format(i_traj), self._save_buffer))
