import logging
import time
from collections import deque

from PIL import Image
import itertools
import gym
import numpy as np
import time
from collections import deque
from gym import error, spaces
from mlagents.envs.environment import UnityEnvironment


class UnityGymException(error.Error):
    """
    Any error related to the gym wrapper of ml-agents.
    """
    pass


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gym_unity")
    

class ObstacleTowerEnv(gym.Env):
    ALLOWED_VERSIONS = ['3.1']

    def __init__(self, environment_filename=None, docker_training=False, worker_id=0, retro=True,
                 timeout_wait=30, realtime_mode=False, config=None, greyscale=False):
        """
        Arguments:
          environment_filename: The file path to the Unity executable.  Does not require the extension.
          docker_training: Whether this is running within a docker environment and should use a virtual 
            frame buffer (xvfb).
          worker_id: The index of the worker in the case where multiple environments are running.  Each 
            environment reserves port (5005 + worker_id) for communication with the Unity executable.
          retro: Resize visual observation to 84x84 (int8) and flattens action space.
          timeout_wait: Time for python interface to wait for environment to connect.
          realtime_mode: Whether to render the environment window image and run environment at realtime.
        """
        self._env = UnityEnvironment(environment_filename,
                                     worker_id,
                                     docker_training=docker_training,
                                     timeout_wait=timeout_wait)

        split_name = self._env.academy_name.split('-v')
        if len(split_name) == 2 and split_name[0] == "ObstacleTower":
            self.name, self.version = split_name
        else:
            raise UnityGymException(
                "Attempting to launch non-Obstacle Tower environment"
            )

        if self.version not in self.ALLOWED_VERSIONS:
            raise UnityGymException(
                "Invalid Obstacle Tower version.  Your build is v" + self.version +
                " but only the following versions are compatible with this gym: " +
                str(self.ALLOWED_VERSIONS)
            )

        self.visual_obs = None
        self._current_state = None
        self._n_agents = None
        self._flattener = None
        self._greyscale = greyscale

        # Environment reset parameters
        self._seed = None
        self._floor = None

        self.realtime_mode = realtime_mode
        self.game_over = False  # Hidden flag used by Atari environments to determine if the game is over
        self.retro = retro
        if config != None:
            self.config = config
        else:
            self.config = None

        flatten_branched = self.retro
        uint8_visual = self.retro

        # Check brain configuration
        if len(self._env.brains) != 1:
            raise UnityGymException(
                "There can only be one brain in a UnityEnvironment "
                "if it is wrapped in a gym.")
        self.brain_name = self._env.external_brain_names[0]
        brain = self._env.brains[self.brain_name]

        if brain.number_visual_observations == 0:
            raise UnityGymException("Environment provides no visual observations.")

        self.uint8_visual = uint8_visual

        if brain.number_visual_observations > 1:
            logger.warning("The environment contains more than one visual observation. "
                           "Please note that only the first will be provided in the observation.")

        # Check for number of agents in scene.
        initial_info = self._env.reset(train_mode=not self.realtime_mode)[self.brain_name]
        self._check_agents(len(initial_info.agents))

        # Set observation and action spaces
        if len(brain.vector_action_space_size) == 1:
            self._action_space = spaces.Discrete(brain.vector_action_space_size[0])
        else:
            if flatten_branched:
                self._flattener = ActionFlattener(brain.vector_action_space_size)
                self._action_space = self._flattener.action_space
            else:
                self._action_space = spaces.MultiDiscrete(brain.vector_action_space_size)

        high = np.array([np.inf] * brain.vector_observation_space_size)
        self.action_meanings = brain.vector_action_descriptions

        if self._greyscale:
            depth = 1
        else:
            depth = 3
        image_space_max = 1.0
        image_space_dtype = np.float32
        camera_height = brain.camera_resolutions[0]["height"]
        camera_width = brain.camera_resolutions[0]["width"]
        if self.retro:
            image_space_max = 255
            image_space_dtype = np.uint8
            camera_height = 84
            camera_width = 84

        image_space = spaces.Box(
            0, image_space_max,
            dtype=image_space_dtype,
            shape=(camera_height, camera_width, depth)
        )
        if self.retro:
            self._observation_space = image_space
        else:
            max_float = np.finfo(np.float32).max
            keys_space = spaces.Discrete(5)
            time_remaining_space = spaces.Box(low=0.0, high=max_float, shape=(1,), dtype=np.float32)
            floor_space = spaces.Discrete(9999)
            self._observation_space = spaces.Tuple(
                (image_space, keys_space, time_remaining_space, floor_space)
            )

    def reset(self, config=None):
        """Resets the state of the environment and returns an initial observation.
        In the case of multi-agent environments, this is a list.
        Returns: observation (object/list): the initial observation of the
            space.
        """
        if config is None:
            reset_params = {}
            if self.config is not None:
                reset_params = self.config
        else:
            reset_params = config
        if self._floor is not None:
            reset_params['starting-floor'] = self._floor
        if self._seed is not None:
            reset_params['tower-seed'] = self._seed

        self.reset_params = self._env.reset_parameters
        info = self._env.reset(config=reset_params,
                               train_mode=not self.realtime_mode)[self.brain_name]
        n_agents = len(info.agents)
        self._check_agents(n_agents)
        self.game_over = False

        obs, reward, done, info = self._single_step(info)
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
        if self._flattener is not None:
            # Translate action into list
            action = self._flattener.lookup_action(action)

        info = self._env.step(action)[self.brain_name]
        n_agents = len(info.agents)
        self._check_agents(n_agents)
        self._current_state = info

        obs, reward, done, info = self._single_step(info)
        self.game_over = done

        return obs, reward, done, info

    def _single_step(self, info):
        self.visual_obs = self._preprocess_single(info.visual_observations[0][0][:, :, :])

        self.visual_obs, keys, time, current_floor = self._prepare_tuple_observation(
            self.visual_obs, info.vector_observations[0])

        if self.retro:
            self.visual_obs = self._resize_observation(self.visual_obs)
            self.visual_obs = self._add_stats_to_image(
                self.visual_obs, info.vector_observations[0])
            default_observation = self.visual_obs
        else:
            default_observation = self.visual_obs, keys, time, current_floor

        if self._greyscale:
            default_observation = self._greyscale_obs(default_observation)

        return default_observation, info.rewards[0], info.local_done[0], {
            "text_observation": info.text_observations[0],
            "brain_info": info,
            "total_keys": keys,
            "time_remaining": time,
            "current_floor": current_floor
        }

    def _greyscale_obs(self, obs):
        new_obs = np.floor(np.expand_dims(np.mean(obs, axis=2), axis=2)).astype(np.uint8)
        return new_obs

    def _preprocess_single(self, single_visual_obs):
        if self.uint8_visual:
            return (255.0 * single_visual_obs).astype(np.uint8)
        else:
            return single_visual_obs

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
        """Sets a fixed seed for this env's random number generator(s).
        The valid range for seeds is [0, 99999). By default a random seed
        will be chosen.
        """
        if seed is None:
            self._seed = seed
            return

        seed = int(seed)
        if seed < 0 or seed >= 99999:
            logger.warning(
                "Seed outside of valid range [0, 99999). A random seed "
                "within the valid range will be used on next reset."
            )
        logger.warning("New seed " + str(seed) + " will apply on next reset.")
        self._seed = seed

    def floor(self, floor=None):
        """Sets the starting floor to a fixed floor number on subsequent environment
        resets."""
        if floor is None:
            self._floor = floor
            return

        floor = int(floor)
        if floor < 0 or floor > 99:
            logger.warning(
                "Starting floor outside of valid range [0, 99]. Floor 0 will be used"
                "on next reset."
            )
        logger.warning("New starting floor " + str(floor) + " will apply on next reset.")
        self._floor = floor

    @staticmethod
    def _resize_observation(observation):
        """
        Re-sizes visual observation to 84x84
        """
        obs_image = Image.fromarray(observation)
        obs_image = obs_image.resize((84, 84), Image.NEAREST)
        return np.array(obs_image)

    @staticmethod
    def _prepare_tuple_observation(vis_obs, vector_obs):
        """
        Converts separate visual and vector observation into prepared tuple
        """
        key = vector_obs[0:6]
        time = vector_obs[6]
        floor_number = vector_obs[7]
        key_num = np.argmax(key, axis=0)
        return vis_obs, key_num, time, floor_number

    @staticmethod
    def _add_stats_to_image(vis_obs, vector_obs):
        """
        Displays time left and number of keys on visual observation
        """
        key = vector_obs[0:6]
        time = vector_obs[6]
        key_num = int(np.argmax(key, axis=0))
        time_num = min(time, 10000) / 10000

        vis_obs[0:10, :, :] = 0
        for i in range(key_num):
            start = int(i * 16.8) + 4
            end = start + 10
            vis_obs[1:5, start:end, 0:2] = 255
        vis_obs[6:10, 0:int(time_num * 84), 1] = 255
        return vis_obs

    def _check_agents(self, n_agents):
        if n_agents > 1:
            raise UnityGymException(
                "The environment was launched as a single-agent environment, however"
                "there is more than one agent in the scene.")
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


class EpisodeResults:
    def __init__(self, seed, reset_params):
        self.seed = seed
        self.start_time = time.time()
        self.time_elapsed = None
        self.total_steps = 0
        self.reward = 0.0
        self.max_floor_reached = 0
        self.reset_params = reset_params

    def complete(self, reward, floor, total_steps):
        curr_time = time.time()
        self.time_elapsed = curr_time - self.start_time
        self.reward = reward
        self.max_floor_reached = floor
        self.total_steps = total_steps

    def as_dict(self):
        return {
            'seed': self.seed,
            'time_elapsed': self.time_elapsed,
            'episode_reward': self.reward,
            'max_floor_reached': self.max_floor_reached,
            'total_steps': self.total_steps,
            'reset_params': self.reset_params
        }


class ObstacleTowerEvaluation(gym.Wrapper):
    """
    Environment wrapper for performing evaluation. Accepts an ObstacleTowerEnv and a list 
    of seeds and will collect resulting rewards and floors reached for each seed.  This wrapper 
    automatically resets the environment, so an external environment reset is not necessary.
    """

    def __init__(self, env, seeds):
        """
        Arguments:
        env: ObstacleTowerEnv object created externally.
        """
        super().__init__(env)

        if not isinstance(seeds, list):
            raise UnityGymException("Invalid seeds list for evaluation.")
        if len(seeds) < 1:
            raise UnityGymException("No seeds provided for evaluation.")
        self.episode_results = {}
        self.episodic_return = 0.0
        self.episodic_steps = 0
        self.current_floor = 0
        self.seeds = deque(seeds)
        self.current_seed = self.seeds.popleft()
        self.env.seed(self.current_seed)
        self.reset()

    def reset(self):
        if self.current_seed is None:
            raise UnityGymException("Attempting to reset but evaluation has completed.")

        obs = self.env.reset()
        self.episodic_return = 0.0
        self.episodic_steps = 0
        self.current_floor = 0
        self.episode_results[self.current_seed] = EpisodeResults(
            self.current_seed,
            self.env.reset_params
        )
        return obs

    def step(self, action):
        if self.current_seed is None:
            raise UnityGymException("Attempting to step but evaluation has completed.")

        observation, reward, done, info = self.env.step(action)
        self.episodic_return += reward
        self.episodic_steps += 1
        if info['current_floor'] > self.current_floor:
            self.current_floor = info['current_floor']
        if done:
            self.episode_results[self.current_seed].complete(
                self.episodic_return,
                self.current_floor,
                self.episodic_steps)
            if len(self.seeds) > 0:
                self.current_seed = self.seeds.popleft()
                self.env.seed(self.current_seed)
                self.reset()
            else:
                self.current_seed = None
        return observation, reward, done, info

    @property
    def evaluation_complete(self):
        return self.current_seed is None

    @property
    def results(self):
        """
        Returns the evaluation results in a dictionary.  Results include the average reward and floor 
        reached for each seed and the list of rewards / floors reached for each seed.
        """
        total_reward = 0.0
        total_floors = 0.0
        total_steps = 0.0
        num_episodes = len(self.episode_results.values())
        for result in self.episode_results.values():
            total_reward += result.reward
            total_floors += result.max_floor_reached
            total_steps += result.total_steps
        return {
            'average_reward': total_reward / num_episodes,
            'average_floor_reached': total_floors / num_episodes,
            'average_episode_steps': total_steps / num_episodes,
            'episode_count': num_episodes,
            'episodes': list(map(lambda es: es.as_dict(), self.episode_results.values()))
        }
