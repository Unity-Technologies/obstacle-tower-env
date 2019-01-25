# Obstacle Tower Environment

![alt text](banner.png "Obstacle Tower")

## About

The Obstacle Tower is a procedurally generated environment consisting of multiple floors to be solved by a learning agent. It is designed to test learning agents abilities in computer vision, locomotion skills, high-level planning, and generalization. It combines platforming-style gameplay with puzzles and planning problems, and critically, increases in difficult as the agent progresses.

Within each floor, the goal of the agent is to arrive at the set of stairs leading to the next level of the tower. These floors are composed of multiple rooms, each which can contain their own unique challenges. Furthermore, each floor contains a number of procedurally generated elements, such as visual appearance, puzzle configuration, and floor layout. This ensures that in order for an agent to be successful at the Obstacle Tower task, they must be able to generalize to new and unseen combinations of conditions.

### Reference Paper

To learn more, read our AAAI Workshop paper on the Obstacle Tower [here]().

It can be cited as:

[to be added]

### Version History

* v1.0 - Initial Release.

## Installation

### Requirements
* ML-Agents v0.6
* OpenAI Gym
* Mac, Windows, or Linux

### Download the environment

* Windows:
* Mac:
* Linux:

### Install the Gym interface

```bash
$ pip install -e obstacle-tower-env
```

## Getting Started

Below is a simple python snippet which describes how to launch and interact with the Obstacle Tower.

[Insert code snippet]