# Obstacle Tower Environment

![alt text](banner.png "Obstacle Tower")

## About

The Obstacle Tower is a procedurally generated environment consisting of multiple floors to be solved by a learning agent. It is designed to test learning agents abilities in computer vision, locomotion skills, high-level planning, and generalization. It combines platforming-style gameplay with puzzles and planning problems, and critically, increases in difficult as the agent progresses.

Within each floor, the goal of the agent is to arrive at the set of stairs leading to the next level of the tower. These floors are composed of multiple rooms, each which can contain their own unique challenges. Furthermore, each floor contains a number of procedurally generated elements, such as visual appearance, puzzle configuration, and floor layout. This ensures that in order for an agent to be successful at the Obstacle Tower task, they must be able to generalize to new and unseen combinations of conditions.

### Reference Paper

To learn more, please read our AAAI Workshop paper:

[**The Obstacle Tower: A Generalization Challenge in Vision, Control, and Planning**](https://storage.googleapis.com/obstacle-tower-build/Obstacle_Tower_Paper_Final.pdf).

### Version History

* v1.0 - Initial Release.

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
| Linux (x86_64) | https://storage.googleapis.com/obstacle-tower-build/v1/obstacletower_v1_linux.zip   |
| Mac OS X       | https://storage.googleapis.com/obstacle-tower-build/v1/obstacletower_v1_osx.zip     |
| Windows        | https://storage.googleapis.com/obstacle-tower-build/v1/obstacletower_v1_windows.zip |

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
