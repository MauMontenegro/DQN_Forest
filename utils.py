import gym
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import animation


def create_environment(config):
    env = gym.make(config.env_name)
    config.action_num = env.action_space.n
    if config.env_type == "Vector":
        config.state_space_input = env.observation_space.shape[0]
    elif config.env_type == "Atari":
        config.state_size_h = env.observation_space.shape[0]  # Height
        config.state_size_w = env.observation_space.shape[1]  # Width
        config.state_size_c = env.observation_space.shape[2]  # Channels

    return env


def epsilon_by_frame(config, frame):
    epsilon = config.final_epsilon + (config.initial_epsilon - config.final_epsilon) * math.exp(
        -1. * frame / config.decay_epsilon)

    return epsilon


def beta_by_frame(config, frame):
    beta = min(1.0, config.beta_start + ((frame * (1.0 - config.beta_start)) / config.beta_frames))
    return beta


def plot(file_path):
    experiment_file = np.loadtxt(file_path, comments="#", delimiter=",", unpack=False)
    m_rewards = experiment_file[:, 0]
    m_samples = experiment_file[:, 1]
    std_dev = experiment_file[:, 2]

    x_range = range(0, len(m_rewards))
    plt.figure(figsize=(20, 5))
    plt.title("Average Reward")
    plt.ylabel('Mean Reward')
    plt.xlabel('Episode')
    plt.errorbar(x_range, m_rewards, std_dev, color='gray', ecolor='lightgray', elinewidth=1, capsize=0)
    plt.show()


def save_frames_as_gif(frames, path='Recordings/', filename='gym_animation.gif'):
    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


def pre_processing_img(image, h_img, w_img, env_type, crop_top_image=20):
    if env_type == "Atari":
        # To grayscale
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Cut 20 px from top
        frame = frame[crop_top_image:h_img, 0:w_img]
        # Resize
        frame = cv2.resize(frame, (80, 64))
        # Normalize
        frame = frame.reshape(80, 64) / 255
    elif env_type == "Vector":
        frame = image

    return frame


if __name__ == '__main__':
    file = np.loadtxt("Experiments/DQN/CartPole-v0/File", comments="#", delimiter=",", unpack=False)
    m_rewards = file[:, 0]
    m_samples = file[:, 1]
    std_dev = file[:, 2]
    # plot(m_rewards, m_samples, std_dev, "Hello")
