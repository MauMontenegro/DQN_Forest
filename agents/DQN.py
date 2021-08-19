# DQN Agent
import torch
import torch as T
import numpy as np
import random as rnd
import importlib


class DQN:
    def __init__(self, config):
        Net = getattr(importlib.import_module("nets." + config.net), config.net)
        self.current_model = Net(config)
        self.target_model = Net(config)

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.current_model.to(self.device)
        self.target_model.to(self.device)

        self.action_space_num = config.action_num
        self.optimizer = T.optim.Adam(self.current_model.parameters())

    def act(self, state, epsilon):
        if rnd.random() > epsilon:
            with torch.no_grad():
                q_value = self.current_model.forward(state.to(self.device))
            action = T.argmax(q_value).item()
        else:
            action = rnd.randrange(self.action_space_num)
        return action

    def train(self, config, buffer):
        state, action, reward, next_state, done = buffer.sample(config.batch_size)

        state = T.FloatTensor(np.float32(state)).to(self.device)
        next_state = T.FloatTensor(np.float32(next_state)).to(self.device)
        action = T.LongTensor(action).to(self.device)
        reward = T.FloatTensor(reward).to(self.device)
        done = T.FloatTensor(done).to(self.device)

        # Calculate Q_Values (Prediction) with Online Network
        q_values = self.current_model(state)

        # Calculate q values of next state with online network (To evaluate greedy Policy)(arg max)
        next_q_values = self.current_model(next_state).detach()

        # Estimate value with target network (Evaluation)
        next_q_state_values = self.target_model(next_state)

        # Evaluating next state with the arg max from target network
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        expected_q_value = reward + config.gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def update_target(self):
        print("Updating Parameters")
        self.target_model.load_state_dict(self.current_model.state_dict())
