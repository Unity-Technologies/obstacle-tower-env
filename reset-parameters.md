# Obstacle Tower Reset Parameters

Obstacle Tower can be configured in a variety of ways both when launching the environment and on episode reset. Below are a list of parameters, along with the ranges of values and what they correspond to. Pass these as part of a `config` dictionary when calling `env.reset(config=config)`, or pass them as part of a dicitonary when launching the environment in `ObstacleTowerEnv('path_to_binary', config=config)`. 

*Note: The config passed on environment launch will be the default used when starting a new episode if there is no config passed during `env.reset()`.*

| *Parameter*  | *Value range* | *Effect* |                                                                  
| --- | --- | --- |
| `tower-seed` | (-1 - 99999)| Sets the seed used to generate the tower. -1 corresponds to a random tower on every `reset()` call. 
| `starting-floor` | (0, 99)| Sets the starting floor for the agent on `reset()`. 
| `total-floors` | (1, 100) | Sets the maximum number of possible floors in the tower.
| `dense-reward` | (0, 1) | Whether to use the sparse (0) or dense (1) reward function.
| `lighting-type` | (0, 1, 2) | Whether to use no realtime light (0), a single realtime light with minimal color variations (1), or a realtime light with large color variations (2). 
| `visual-theme` | (0, 1, 2) | Whether to use only the `default-theme` (0), the normal ordering or themes (1), or a random theme every floor (2).
| `agent-perspective` | (0, 1) | Whether to use first-person (0) or third-person (1) perspective for the agent.
| `allowed-rooms` | (0, 1, 2) | Whether to use only normal rooms (0), normal and key rooms (1), or normal, key, and puzzle rooms (2). 
| `allowed-modules` | (0, 1, 2) | Whether to fill rooms with no modules (0), only easy modules (1), or the full range of modules (2). 
| `allowed-floors` | (0, 1, 2) | Whether to include only straightforward floor layouts (0), layouts that include branching (1), or layouts that include branching and circling (2).
| `default-theme` | (0, 1, 2, 3, 4) | Whether to set the default theme to `Ancient` (0), `Moorish` (1), `Industrial` (2), `Modern` (3), or `Future` (4). 
