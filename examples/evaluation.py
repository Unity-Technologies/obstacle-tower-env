from obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation


def run_episode(env):
    done = False
    episode_return = 0.0
    
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        episode_return += reward
    return episode_return


if __name__ == '__main__':
    # In this example we use the seeds used for evaluating submissions 
    # to the Obstacle Tower Challenge.
    eval_seeds = [1001, 1002, 1003, 1004, 1005]

    # Create the ObstacleTowerEnv gym and launch ObstacleTower
    env = ObstacleTowerEnv('./ObstacleTower/obstacletower')

    # Wrap the environment with the ObstacleTowerEvaluation wrapper
    # and provide evaluation seeds.
    env = ObstacleTowerEvaluation(env, eval_seeds)

    # We can run episodes (in this case with a random policy) until 
    # the "evaluation_complete" flag is True.  Attempting to step or reset after
    # all of the evaluation seeds have completed will result in an exception.
    while not env.evaluation_complete:
        episode_rew = run_episode(env)

    # Finally the evaluation results can be fetched as a dictionary from the 
    # environment wrapper.
    print(env.results)

    env.close()
