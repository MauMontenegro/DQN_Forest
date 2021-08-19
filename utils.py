import gym
import math
import numpy as np
import matplotlib.pyplot as plt


def create_environment(config):
    env = gym.make(config.env_name)
    config.action_num = env.action_space.n
    config.state_space_input = env.observation_space.shape[0]

    return env


def epsilon_by_frame(config, frame):
    epsilon = config.final_epsilon + (config.initial_epsilon - config.final_epsilon) * math.exp(
        -1. * frame / config.decay_epsilon)

    return epsilon


def plot(rewards,message):
    plt.figure(figsize=(20, 5))
    plt.title(message)
    plt.plot(rewards)
    plt.show()
