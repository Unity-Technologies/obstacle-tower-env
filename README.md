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

## Installation

### Requirements

The Obstacle Tower environment runs on Mac OS X, Windows, or Linux.

Python dependencies (also in [setup.py](https://github.com/Unity-Technologies/obstacle-tower-env/blob/master/setup.py)):

* Unity ML-Agents v0.6
* OpenAI Gym
* Pillow

### Download the environment

| *Platform*     | *Download Link*                                                                     |
| --- | --- |
| Linux (x86_64) | https://storage.googleapis.com/obstacle-tower-build/v1.2/obstacletower_v1.2_linux.zip   |
| Mac OS X       | https://storage.googleapis.com/obstacle-tower-build/v1.2/obstacletower_v1.2_osx.zip     |
| Windows        | https://storage.googleapis.com/obstacle-tower-build/v1.2/obstacletower_v1.2_windows.zip |

For checksums on these files, see [here](https://storage.googleapis.com/obstacle-tower-build/v1.2/ote-v1.2-checksums.txt).

### Install the Gym interface

```bash
$ git clone git@github.com:Unity-Technologies/obstacle-tower-env.git
$ cd obstacle-tower-env
$ pip install -e .
```

## Getting Started

### Using the Gym Interface

To see an example of how to interact with the environment using the gym interface, see our [Basic Usage Jupyter Notebook](examples/basic_usage.ipynb).

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

### Training a Dopamine Rainbow agent on GCP

If you are interested in training an agent using Google's Doapmine framework and/or Google Cloud Platform, see our guide [here](./examples/gcp_training.md).