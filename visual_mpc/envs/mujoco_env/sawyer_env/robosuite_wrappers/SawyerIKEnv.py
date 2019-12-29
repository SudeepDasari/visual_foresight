from collections import OrderedDict
import random
import numpy as np
try:
    from robosuite.utils.transform_utils import convert_quat
    from robosuite.environments.sawyer import SawyerEnv

    from .BinArena import BinArena
    from robosuite.models.objects import BreadObject, MilkObject, LemonObject, CanObject, BottleObject, CerealObject
    from robosuite.models.robots import Sawyer
    from robosuite.models.tasks import TableTopTask, UniformRandomSampler
    from robosuite.wrappers import IKWrapper
except ImportError:
    print("Robosuite is required to use this module. Please try ' pip install robosuite' ")
    raise ImportError()


def make_sawyer_env(hparams_dict):
    return IKWrapper(SawyerMultiObjEnv(**hparams_dict))


OBJECTS = [BreadObject, MilkObject, LemonObject, CanObject, BottleObject, CerealObject]


class SawyerMultiObjEnv(SawyerEnv):
    """
    This class corresponds to the lifting task for the Sawyer robot arm.
    """

    def __init__(
        self,
        gripper_type="TwoFingerGripper",
        table_full_size=(0.8, 0.8, 0.8),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=False,
        placement_initializer=None,
        gripper_visualization=False,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=True,
        camera_name=("frontview", "leftview"),
        camera_height=192,
        camera_width=256,
        camera_depth=False,
        num_objects=1
    ):
        """
        Args:
            gripper_type (str): type of gripper, used to instantiate
                gripper models from gripper factory.
            table_full_size (3-tuple): x, y, and z dimensions of the table.
            table_friction (3-tuple): the three mujoco friction parameters for
                the table.
            use_camera_obs (bool): if True, every observation includes a
                rendered image.
            use_object_obs (bool): if True, include object (cube) information in
                the observation.
            reward_shaping (bool): if True, use dense rewards.
            placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomSampler
                is used by default.
            gripper_visualization (bool): True if using gripper visualization.
                Useful for teleoperation.
            use_indicator_object (bool): if True, sets up an indicator object that
                is useful for debugging.
            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.
            has_offscreen_renderer (bool): True if using off-screen rendering.
            render_collision_mesh (bool): True if rendering collision meshes
                in camera. False otherwise.
            render_visual_mesh (bool): True if rendering visual meshes
                in camera. False otherwise.
            control_freq (float): how many control signals to receive
                in every second. This sets the amount of simulation time
                that passes between every action input.
            horizon (int): Every episode lasts for exactly @horizon timesteps.
            ignore_done (bool): True if never terminating the environment (ignore @horizon).
            camera_name (str): name of camera to be rendered. Must be
                set if @use_camera_obs is True.
            camera_height (int): height of camera frame.
            camera_width (int): width of camera frame.
            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.
        """

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # number of objects to initialize on table
        self._num_objects = num_objects

        # reward configuration
        self.reward_shaping = reward_shaping

        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomSampler(
                x_range=[-0.3, 0.3],
                y_range=[-0.3, 0.3],
                ensure_object_boundary_in_range=False,
                z_rotation=True,
            )

        super().__init__(
            gripper_type=gripper_type,
            gripper_visualization=gripper_visualization,
            use_indicator_object=use_indicator_object,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            use_camera_obs=use_camera_obs,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
            camera_depth=camera_depth,
        )
        self.camera_width = camera_width

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = BinArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])

        # initialize objects of interest
        self.mujoco_objects = OrderedDict()
        for i in range(self._num_objects):
            self.mujoco_objects['obj{}'.format(i)] = random.choice(OBJECTS)()

        # task includes arena, robot, and objects of interest
        self.model = TableTopTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            initializer=self.placement_initializer,
        )
        self.model.place_objects()

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()
        self.body_ids = [self.sim.model.body_name2id("obj{}".format(i)) for i in range(self._num_objects)]
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # reset positions of objects
        self.model.place_objects()

        # reset joint positions
        init_pos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        init_pos += np.random.randn(init_pos.shape[0]) * 0.02
        self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(init_pos)

    def reward(self, action=None):
        """
        Visual mpc does not use a reward function
        """
        return 0.

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].
        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()
        # camera observations
        if self.use_camera_obs:
            if self.camera_depth:
                raise NotImplementedError

            camera_obs = np.zeros((len(self.camera_name),
                                   self.camera_height,
                                   self.camera_width,
                                   3), dtype=np.uint8)

            for i, n in enumerate(self.camera_name):
                camera_obs[i] = self.sim.render(
                    camera_name=n,
                    width=self.camera_width,
                    height=self.camera_height,
                    depth=self.camera_depth,
                )[::-1, ::-1]
            di["images"] = camera_obs

        # low-level object information
        if self.use_object_obs:
            # position and rotation of object
            for i, id in enumerate(self.body_ids):
                obj_pos = np.array(self.sim.data.body_xpos[id])
                obj_quat = convert_quat(
                    np.array(self.sim.data.body_xquat[id]), to="xyzw"
                )
                di["obj{}_pos".format(i)] = obj_pos
                di["obj{}_quat".format(i)] = obj_quat

        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if (
                self.sim.model.geom_id2name(contact.geom1)
                in self.gripper.contact_geoms()
                or self.sim.model.geom_id2name(contact.geom2)
                in self.gripper.contact_geoms()
            ):
                collision = True
                break
        return collision

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.table_full_size[2]

        # cube is higher than the table top above a margin
        return cube_height > table_height + 0.04

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """

        # color the gripper site appropriately based on distance to cube
        return
