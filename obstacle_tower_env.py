from gym_unity.envs import UnityEnv
from mlagents.envs import UnityEnvironment
from gym import spaces
import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlagents.envs")

class ObstacleTowerEnv(UnityEnv):
    def __init__(self, environment_filename=None, docker_training=False, worker_id=0, use_visual=True, multiagent=False):
        """
        WARNING: Copied from gym-unity / UnityEnv wholesale.  Duplicates initialization logic since 
        gym-unity doesn't support docker training.  Rather than updating this, it would be better to fix 
        compatibility with gym-unity.

        Environment initialization
        :param environment_filename: The UnityEnvironment path or file to be wrapped in the gym.
        :param worker_id: Worker number for environment.
        :param use_visual: Whether to use visual observation or vector observation.
        :param multiagent: Whether to run in multi-agent mode (lists of obs, reward, done).
        """
        if self.is_grading():
            environment_filename = None

        self._env = UnityEnvironment(environment_filename, worker_id, docker_training=docker_training)
        self.name = self._env.academy_name
        self.visual_obs = None
        self._current_state = None
        self._n_agents = None
        self._multiagent = multiagent
        self._done_grading = False

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
                self._action_space = spaces.MultiDiscrete(brain.vector_action_space_size)
        else:
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

    def step(self, action):
        obs, rew, done, info = super().step(action)
        if info.get('text_observation') == 'evaluation_complete':
            done = True
            self._done_grading = True
        return obs, rew, done, info

    def done_grading(self):
        return self._done_grading

    def is_grading(self):
        return os.getenv('OTC_EVALUATION_ENABLED', False)

