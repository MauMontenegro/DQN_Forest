# Libraries
import settings
import utils
import importlib
import sys
import torch as T
import numpy as np
from tqdm import tqdm
from gym import wrappers
import Logger

# Paths to saved agents and nets
sys.path.append('./agents')
sys.path.append('./nets')

if __name__ == '__main__':

    # Getting Agent-Environment Configuration
    config = settings.config

    # Creating Environment
    env = utils.create_environment(config)
    # Creating Agent
    Agent = getattr(importlib.import_module("agents." + config.agent_name), config.agent_name)
    agent = Agent(config)
    agent.update_target()

    # Create File Path
    file_path = 'Experiments' + '/' + str(config.agent_name) + '/' + str(config.env_name)
    file = '/File'
    # Create Logger
    logger = Logger.Experiment_Log(file_path, 'File')

    # Creating Buffer
    Buffer = getattr(importlib.import_module("Buffers"), config.buffer_name)
    buffer = Buffer(config)

    # Save all rewards of all runs
    Rewards = []
    Sample_efficiency = []
    Losses = []
    frames = []

    for game in range(1, config.total_games + 1):  # Games or Experiments
        print("\n Playing Game: {}".format(game))
        # Reset Buffer and Agent
        buffer.reset_buffer()
        agent.reset_agent()

        # Reset Results, flags and state
        all_rewards = []
        all_losses = []
        env_interactions = []
        update_flag = 0
        interaction = 0
        for episodes in tqdm(range(1, config.total_episodes + 1)):  # Total Episodes for a game
            # print("Playing Episode: {}".format(episodes))
            done = False
            state = env.reset()
            state = utils.pre_processing_img(state, config.state_size_h, config.state_size_w, config.env_type)
            # Stack 4 frames if Env is Atari
            if config.env_type == "Atari":
                state = np.stack((state, state, state, state))
            frame = 0
            loss = 0
            episode_reward = 0
            while not done:  # Until Episode ends with failure or max frames playing
                if config.save_render and (game == config.total_games) and (episodes == config.total_episodes):
                    frames.append(env.render(mode="rgb_array"))
                epsilon = utils.epsilon_by_frame(config, episodes)  # Epsilon-Greedy strategy to take an action
                action = agent.act(T.FloatTensor(np.float32(state)), epsilon)
                interaction += 1
                frame += 1
                next_state, reward, done, _ = env.step(action)  # Step in environment
                next_state = utils.pre_processing_img(next_state, config.state_size_h, config.state_size_w,
                                                      config.env_type)
                episode_reward += reward  # Accumulate reward
                if config.env_type == "Atari":
                    next_state = np.stack((next_state, state[0], state[1], state[2]))
                buffer.push(state, action, reward, next_state, done)  # Store transition in Buffer
                state = next_state

                # Wait to learn until buffer have at least batch size elements
                if len(buffer) > config.batch_size:
                    beta = utils.beta_by_frame(config, episodes)
                    loss = agent.train(config, buffer, beta)
                    update_flag += 1
                    all_losses.append(loss.item())

                # Copy parameters between models
                if update_flag % 100 == 0:
                    agent.update_target()
                    update_flag = 0

            # If episode ends due to failure or success, reset flags and store reward.
            all_rewards.append(episode_reward)
            env_interactions.append(interaction)

            # print("Episode Done after:{} frames".format(frame))
            # print("Episode Reward:{}".format(episode_reward))

            # print("\n")

        Rewards.append(all_rewards)
        Sample_efficiency.append(env_interactions)
        Losses.append(all_losses)
        # Store results in a file
        # with open('Results.txt', 'w') as writefile:
        # writefile.write(str(all_rewards) + '\n')

    env.close()
    utils.save_frames_as_gif(frames)
    # Statistics
    Rewards = np.array(Rewards)
    Sample_efficiency = np.array(Sample_efficiency)
    #Losses = np.array(Losses)

    game_mean_reward = []
    game_mean_sample_efficiency = []
    game_std_dev = []
    game_mean_losses = []

    # Calculate Std Deviation for rewards and sample efficiency
    for i in range(0, config.total_episodes):
        game_mean_reward.append(np.mean(Rewards[:, i]))
        game_mean_sample_efficiency.append(np.mean(Sample_efficiency[:, i]))
        # game_mean_losses.append(np.mean(Losses[:, i]))
        game_std_dev.append(np.std(Rewards[:, i]))
        # game_mean_losses.append(np.mean(Losses[:, i]))
    # Plotting Utility
    # utils.plot(game_mean_reward, game_std_dev, "Mean Reward")
    logger.log_save(game_mean_reward, game_mean_sample_efficiency, game_std_dev)
    #utils.plot(game_mean_reward, game_mean_sample_efficiency, game_std_dev, "Sample Efficiency")
    # utils.plot(all_losses, "Loss")

    # Plotting Experiments
    utils.plot(file_path+file)
