# Libraries
import settings
import utils
import importlib
import sys
import torch as T
import numpy as np
from gym import wrappers

# Paths to saved agents and nets
sys.path.append('./agents')
sys.path.append('./nets')

if __name__ == '__main__':

    # Getting Agent-Environment Configuration
    config = settings.config

    # Creating Environment
    env = utils.create_environment(config)
    # env = wrappers.Monitor(env, "\Recordings", video_callable=False, force=True)
    # Creating Agent
    Agent = getattr(importlib.import_module("agents." + config.agent_name), config.agent_name)
    agent = Agent(config)
    agent.update_target()

    # Creating Buffer
    Buffer = getattr(importlib.import_module("Buffers"), config.buffer_name)
    buffer = Buffer(config)

    Rewards = []

    for game in range(1, config.total_games + 1):  # Games or Experiments
        print("Playing Game: {}".format(game))

        # Reset Results, flags and state
        all_rewards = []
        all_losses = []
        state = env.reset()
        done = False
        episode_reward = 0
        frame = 0
        loss = 0
        update_flag=0

        for episodes in range(1, config.total_episodes + 1 ):  # Total Episodes for a game
            print("Playing Episode: {}".format(episodes))

            while not done:  # Until Episode ends with failure or max frames playing
                epsilon = utils.epsilon_by_frame(config, episodes)  # Epsilon-Greedy strategy to take an action
                action = agent.act(T.FloatTensor(np.float32(state)), epsilon)
                frame += 1
                update_flag += 1
                next_state, reward, done, _ = env.step(action)  # Step in environment
                episode_reward += reward  # Accumulate reward
                buffer.push(state, action, reward, next_state, done)  # Store transition in Buffer
                state = next_state

                # Wait to learn until buffer have at least batch size elements
                if len(buffer) > config.batch_size:
                    loss = agent.train(config, buffer)
                    all_losses.append(loss.item())

                # Copy parameters between models
                if update_flag % 100 == 0:
                    agent.update_target()

            # If episode ends due to failure or success, reset flags and store reward.
            print("Episode Done after:{} frames".format(frame))
            print("Episode Reward:{}".format(episode_reward))
            done = False
            state = env.reset()
            frame = 0
            all_rewards.append(episode_reward)
            episode_reward = 0
            print("\n")

        Rewards.append(all_rewards)

        # Store results in a file
        # with open('Results.txt', 'w') as writefile:
            # writefile.write(str(all_rewards) + '\n')

    # Statistics
    Rewards = np.array(Rewards)
    game_mean_reward = []
    for i in range(0, config.total_episodes):
        print(i)
        game_mean_reward.append(np.mean(Rewards[:, i]))
    # Plotting Utility
    utils.plot(game_mean_reward, "Mean Reward")
    # utils.plot(all_losses, "Loss")
