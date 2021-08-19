# Settings File
"""
    Contains Main Parameters
        -Environment
        -Network
        -Agent
        -Training Parameters
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-env_name", default="LunarLander-v2")
parser.add_argument("-agent_name", default="DQN")
parser.add_argument("-replay_memory_capacity", type=int, default=10000)  # Buffer Capacity
parser.add_argument("-buffer_name", default="Simple_Buffer")  # Buffer Capacity
parser.add_argument("-batch_size", type=int, default=32)  # Batch Size
parser.add_argument("-net", default="Simple_Net")  # Network Model
parser.add_argument("-gamma", type=float, default=0.99)  # Importance of future rewards
parser.add_argument("-learning_rate", type=float, default=0.00025)  # Learning Step
parser.add_argument("-initial_epsilon", type=float, default=0.99)  # Epsilon Greedy Strategy
parser.add_argument("-final_epsilon", type=float, default=0.01)
parser.add_argument("-decay_epsilon", type=int, default=100)

# Loop Arguments
parser.add_argument("-total_games", type=int, default=20)  # Total numbers of Plays (Games)
parser.add_argument("-total_episodes", type=int, default=250)  # Total of Movements
parser.add_argument("-max_frames", type=int, default=200)  # Max number of frames in episode

parser.add_argument("-optimizer_name", default="Adam")  # Optimizer
parser.add_argument("-logging", default="")

config = parser.parse_args()
config.logging = config.logging not in ["0", "false", "False"]


