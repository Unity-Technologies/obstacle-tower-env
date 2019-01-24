import logging
from PIL import Image
import itertools
import gym
import numpy as np
from mlagents.envs import UnityEnvironment
from gym import error, spaces
import os


class UnityGymException(error.Error):
    """
    Any error related to the gym wrapper of ml-agents.
    """
    pass


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gym_unity")


class ObstacleTowerEnv(gym.Env):
    def __init__(self, environment_filename=None, docker_training=False, worker_id=0, retro=True):
        """
        WARNING: Copied from gym-unity / UnityEnv wholesale.  Duplicates initialization logic since 
        gym-unity doesn't support docker training.  Rather than updating this, it would be better to fix 
        compatibility with gym-unity.

        Arguments:
          environment_filename: The file path to the Unity executable.  Does not require the extension.
          docker_training: Whether this is running within a docker environment and should use a virtual 
            frame buffer (xvfb).
          worker_id: The index of the worker in the case where multiple environments are running.  Each 
            environment reserves port (5005 + worker_id) for communication with the Unity executable.
          retro: Resize visual observation to 84x84 (int8) and flattens action space.
        """
        if self.is_grading():
            environment_filename = None

        self._env = UnityEnvironment(environment_filename, worker_id,
                                     docker_training=docker_training)
        self.name = self._env.academy_name
        self.visual_obs = None
        self._current_state = None
        self._n_agents = None
        self._multiagent = False
        self._done_grading = False
        self._flattener = None
        self.game_over = False  # Hidden flag used by Atari environments to determine if the game is over
        self.retro = retro

        use_visual = True
        flatten_branched = self.retro
        uint8_visual = self.retro

        # Check brain configuration
        if len(self._env.brains) != 1:
            raise UnityGymException(
                "There can only be one brain in a UnityEnvironment "
                "if it is wrapped in a gym.")
        self.brain_name = self._env.external_brain_names[0]
        brain = self._env.brains[self.brain_name]

        if use_visual and brain.number_visual_observations == 0:
            raise UnityGymException("`use_visual` was set to True, however there are no"
                                    " visual observations as part of this environment.")
        self.use_visual = brain.number_visual_observations >= 1 and use_visual

        if not use_visual and uint8_visual:
            logger.warning("`uint8_visual was set to true, but visual observations are not in use. "
                           "This setting will not have any effect.")
        else:
            self.uint8_visual = uint8_visual

        if brain.number_visual_observations > 1:
            logger.warning("The environment contains more than one visual observation. "
                           "Please note that only the first will be provided in the observation.")

        if brain.num_stacked_vector_observations != 1:
            raise UnityGymException(
                "There can only be one stacked vector observation in a UnityEnvironment "
                "if it is wrapped in a gym.")

        # Check for number of agents in scene.
        initial_info = self._env.reset()[self.brain_name]
        self._check_agents(len(initial_info.agents))

        # Set observation and action spaces
        if brain.vector_action_space_type == "discrete":
            if len(brain.vector_action_space_size) == 1:
                self._action_space = spaces.Discrete(brain.vector_action_space_size[0])
            else:
                if flatten_branched:
                    self._flattener = ActionFlattener(brain.vector_action_space_size)
                    self._action_space = self._flattener.action_space
                else:
                    self._action_space = spaces.MultiDiscrete(brain.vector_action_space_size)

        else:
            if flatten_branched:
                logger.warning("The environment has a non-discrete action space. It will "
                               "not be flattened.")
            high = np.array([1] * brain.vector_action_space_size[0])
            self._action_space = spaces.Box(-high, high, dtype=np.float32)
        high = np.array([np.inf] * brain.vector_observation_space_size)
        self.action_meanings = brain.vector_action_descriptions
        if self.use_visual:
            if brain.camera_resolutions[0]["blackAndWhite"]:
                depth = 1
            else:
                depth = 3
            self._observation_space = spaces.Box(0, 1, dtype=np.float32,
                                                 shape=(brain.camera_resolutions[0]["height"],
                                                        brain.camera_resolutions[0]["width"],
                                                        depth))
        else:
            self._observation_space = spaces.Box(-high, high, dtype=np.float32)

    def done_grading(self):
        return self._done_grading

    def is_grading(self):
        return os.getenv('OTC_EVALUATION_ENABLED', False)

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        In the case of multi-agent environments, this is a list.
        Returns: observation (object/list): the initial observation of the
            space.
        """
        info = self._env.reset()[self.brain_name]
        n_agents = len(info.agents)
        self._check_agents(n_agents)
        self.game_over = False

        if not self._multiagent:
            obs, reward, done, info = self._single_step(info)
        else:
            obs, reward, done, info = self._multi_step(info)
        return obs

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        In the case of multi-agent environments, these are lists.
        Args:
            action (object/list): an action provided by the environment
        Returns:
            observation (object/list): agent's observation of the current environment
            reward (float/list) : amount of reward returned after previous action
            done (boolean/list): whether the episode has ended.
            info (dict): contains auxiliary diagnostic information, including BrainInfo.
        """

        # Use random actions for all other agents in environment.
        if self._multiagent:
            if not isinstance(action, list):
                raise UnityGymException("The environment was expecting `action` to be a list.")
            if len(action) != self._n_agents:
                raise UnityGymException(
                    "The environment was expecting a list of {} actions.".format(self._n_agents))
            else:
                if self._flattener is not None:
                    # Action space is discrete and flattened - we expect a list of scalars
                    action = [self._flattener.lookup_action(_act) for _act in action]
                action = np.array(action)
        else:
            if self._flattener is not None:
                # Translate action into list
                action = self._flattener.lookup_action(action)

        info = self._env.step(action)[self.brain_name]
        n_agents = len(info.agents)
        self._check_agents(n_agents)
        self._current_state = info

        if not self._multiagent:
            obs, reward, done, info = self._single_step(info)
            self.game_over = done
        else:
            obs, reward, done, info = self._multi_step(info)
            self.game_over = all(done)

        if info.get('text_observation') == 'evaluation_complete':
            done = True
            self._done_grading = True
        return obs, reward, done, info

    def _single_step(self, info):
        if self.use_visual:
            self.visual_obs = self._preprocess_single(info.visual_observations[0][0, :, :, :])
            if self.retro:
                self.visual_obs = self._resize_observation(self.visual_obs)
                self.visual_obs = self._add_stats_to_image(
                    self.visual_obs, info.vector_observations[0])
                default_observation = self.visual_obs
            else:
                default_observation = self._prepare_tuple_observation(
                    self.visual_obs, info.vector_observations[0])
        else:
            default_observation = info.vector_observations[0, :]

        return default_observation, info.rewards[0], info.local_done[0], {
            "text_observation": info.text_observations[0],
            "brain_info": info}

    def _preprocess_single(self, single_visual_obs):
        if self.uint8_visual:
            return (255.0 * single_visual_obs).astype(np.uint8)
        else:
            return single_visual_obs

    def _multi_step(self, info):
        if self.use_visual:
            self.visual_obs = self._preprocess_multi(info.visual_observations)
            default_observation = self.visual_obs
        else:
            default_observation = info.vector_observations
        return list(default_observation), info.rewards, info.local_done, {
            "text_observation": info.text_observations,
            "brain_info": info}

    def _preprocess_multi(self, multiple_visual_obs):
        if self.uint8_visual:
            return [(255.0 * _visual_obs).astype(np.uint8) for _visual_obs in multiple_visual_obs]
        else:
            return multiple_visual_obs

    def render(self, mode='rgb_array'):
        return self.visual_obs

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self._env.close()

    def get_action_meanings(self):
        return self.action_meanings

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Currently not implemented.
        """
        logger.warn("Could not seed environment %s", self.name)
        return

    @staticmethod
    def _resize_observation(observation):
        """
        Re-sizes visual observation to 84x84
        """
        retro_height = 84
        retro_width = 84
        obs_image = Image.fromarray(observation)
        obs_image = obs_image.resize((retro_height, retro_width), Image.NEAREST)
        return np.array(obs_image)

    @staticmethod
    def _prepare_tuple_observation(vis_obs, vector_obs):
        """
        Converts separate visual and vector observation into prepared tuple
        """
        key = vector_obs[0:6]
        time = vector_obs[6]
        key_num = np.argmax(key, axis=0)
        return vis_obs, key_num, time

    @staticmethod
    def _add_stats_to_image(vis_obs, vector_obs):
        """
        Displays time left and number of keys on visual observation
        """
        key = vector_obs[0:6]
        time = vector_obs[6]
        key_num = np.argmax(key, axis=0)
        time_num = min(time, 10000) / 10000

        vis_obs[0:10, :, :] = 0
        for i in range(key_num):
            start = int(i * 16.8) + 4
            end = start + 10
            vis_obs[1:5, start:end, 0:2] = 255
        vis_obs[6:10, 0:int(time_num * 84), 1] = 255
        return vis_obs

    def _check_agents(self, n_agents):
        if not self._multiagent and n_agents > 1:
            raise UnityGymException(
                "The environment was launched as a single-agent environment, however"
                "there is more than one agent in the scene.")
        elif self._multiagent and n_agents <= 1:
            raise UnityGymException(
                "The environment was launched as a mutli-agent environment, however"
                "there is only one agent in the scene.")
        if self._n_agents is None:
            self._n_agents = n_agents
            logger.info("{} agents within environment.".format(n_agents))
        elif self._n_agents != n_agents:
            raise UnityGymException("The number of agents in the environment has changed since "
                                    "initialization. This is not supported.")

    @property
    def metadata(self):
        return {'render.modes': ['rgb_array']}

    @property
    def reward_range(self):
        return -float('inf'), float('inf')

    @property
    def spec(self):
        return None

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def number_agents(self):
        return self._n_agents


class ActionFlattener():
    """
    Flattens branched discrete action spaces into single-branch discrete action spaces.
    """

    def __init__(self, branched_action_space):
        """
        Initialize the flattener.
        :param branched_action_space: A List containing the sizes of each branch of the action
        space, e.g. [2,3,3] for three branches with size 2, 3, and 3 respectively.
        """
        self._action_shape = branched_action_space
        self.action_lookup = self._create_lookup(self._action_shape)
        self.action_space = spaces.Discrete(len(self.action_lookup))

    @classmethod
    def _create_lookup(self, branched_action_space):
        """
        Creates a Dict that maps discrete actions (scalars) to branched actions (lists).
        Each key in the Dict maps to one unique set of branched actions, and each value
        contains the List of branched actions.
        """
        possible_vals = [range(_num) for _num in branched_action_space]
        all_actions = [list(_action) for _action in itertools.product(*possible_vals)]
        # Dict should be faster than List for large action spaces
        action_lookup = {_scalar: _action for (_scalar, _action) in enumerate(all_actions)}
        return action_lookup

    def lookup_action(self, action):
        """
        Convert a scalar discrete action into a unique set of branched actions.
        :param: action: A scalar value representing one of the discrete actions.
        :return: The List containing the branched actions.
        """
        return self.action_lookup[action]
