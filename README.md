# Obstacle Tower Environment

![alt text](banner.png "Obstacle Tower")

## About

The Obstacle Tower is a procedurally generated environment consisting of multiple floors to be solved by a learning agent. It is designed to test learning agents abilities in computer vision, locomotion skills, high-level planning, and generalization. It combines platforming-style gameplay with puzzles and planning problems, and critically, increases in difficulty as the agent progresses.

Within each floor, the goal of the agent is to arrive at the set of stairs leading to the next level of the tower. These floors are composed of multiple rooms, each which can contain their own unique challenges. Furthermore, each floor contains a number of procedurally generated elements, such as visual appearance, puzzle configuration, and floor layout. This ensures that in order for an agent to be successful at the Obstacle Tower task, they must be able to generalize to new and unseen combinations of conditions.

### Reference Paper

To learn more, please read our AAAI Workshop paper:

[**Obstacle Tower: A Generalization Challenge in Vision, Control, and Planning**](https://arxiv.org/abs/1902.01378).

### Version History

* v1.0 - Initial Release.
* v1.1 - Obstacle Tower Challenge Round 1 Release.
   * Improved determinism between resets.
   * Fixed bug in room template leading to un-beatable floors.
   * Improved rendering & communication speed.
* v1.2 - Hotfix release.
	* Adds timeout_wait parameter to extend python wait time for Unity environment handshake.
	* Adds realtime_mode parameter to launch Unity environment from API at real-time speed and render to the window.
	* Updates Windows and Linux binaries to address launching issues.
	* Updated v1.2 binary includes fixes for agent collision detection issues.
* v1.3 Hotfix release.
   * Resolves memory leak when running in Docker.
   * Fixes issue where environment could freeze on certain higher floors.
* v2.0 Obstacle Tower Challenge Round 2 Release.
   * Towers can now be generated with up to 100 floors.
   * Additional visual themes, obstacles, enemy types, and floor difficulties added.
   * Additional reset parameters added to customize generated towers. Go [here](./reset-parameters.md) for details on the parameters and their values.
   * Various bugs fixed and performance improvements.
* v2.1 Hotfix release.
   * Resolves issue with new seeds not being applied on `env.reset`.
   * Resolves issue with underspecified observation space.
* v2.2 Hotfix release.
   * Resolves issue with reset parameters sometimes not being updated during `env.reset`.
   * Resolves issue where agents could possibly skip levels.
* v3.0 Open Source release available [here](https://github.com/Unity-Technologies/obstacle-tower-source).
   * Added evaluation script for benchmarking, along with official benchmarking guidelines.
* v3.1 Minor release.
   * Updates Unity project to 2019.2 and ML-Agents version to 0.10.
* 4.0 Major project updates
   * Unity project updated to 2019.4 LTS.
   * ML-Agents package updated to 1.1.0 (Release 3).
   * Gym wrapper now uses registry by default to auto-download binary.
* 4.1 Minor release.
   * Adds additional reset parameters for more fine-grained control of visual themes.
   * Fixes issue in gym wrapper related to end of episode information.
   * Fixes issue with environment not being responsive to some reset parameters.
  

## Installation

### Requirements

The Obstacle Tower environment runs on Mac OS X, Windows, or Linux.

Python dependencies (also in [setup.py](https://github.com/Unity-Technologies/obstacle-tower-env/blob/master/setup.py)):

* Python 3.6+
* Unity ML-Agents 1.x
* OpenAI Gym

### Download the environment (optional)

By default, the binary will be automatically downloaded when the Obstacle Tower gym is first instantiated. The binaries for each platform can be separately downloaded at the following links.

| *Platform*     | *Download Link*                                                                     |
| --- | --- |
| Linux (x86_64) | https://storage.googleapis.com/obstacle-tower-build/v4.1/obstacletower_v4.1_linux.zip   |
| Mac OS X       | https://storage.googleapis.com/obstacle-tower-build/v4.1/obstacletower_v4.1_osx.zip     |
| Windows        | https://storage.googleapis.com/obstacle-tower-build/v4.1/obstacletower_v4.1_windows.zip |

For checksums on these files, see [here](https://storage.googleapis.com/obstacle-tower-build/v4.1/ote-v4.1-checksums.txt).

### Install the Gym interface

```bash
$ git clone git@github.com:Unity-Technologies/obstacle-tower-env.git
$ cd obstacle-tower-env
$ pip install -e .
```

## Getting Started

### Using the Gym Interface

To see an example of how to interact with the environment using the gym interface, see our [Basic Usage Jupyter Notebook](examples/basic_usage.ipynb).

### Customizing the environment

Obstacle Tower can be configured in a number of different ways to adjust the difficulty and content of the environment. This is done through the use of reset parameters, which can be set when calling `env.reset()`. See [here](./reset-parameters.md) for a list of the available parameters to adjust. 

### Player Control

It is also possible to launch the environment in "Player Mode," and directly control the agent using a keyboard. This can be done by double-clicking on the binary file. The keyboard controls are as follows:

| *Keyboard Key* | *Action* |
| --- | --- |
| W | Move character forward. |
| S | Move character backwards. |
| A | Move character left. |
| D | Move character right. |
| K | Rotate camera left. |
| L | Rotate camera right. |
| Space | Character jump. |

### Performance evaluation

We provide an environment wrapper for evaluating performance of a player or agent across multiple pre-defined seeds.  We provide [an example implementation](examples/evaluation.py) of evaluation on a random policy.

### Training a Dopamine Rainbow agent on GCP

If you are interested in training an agent using Google's Dopamine framework and/or Google Cloud Platform, see our guide [here](./examples/gcp_training.md).
