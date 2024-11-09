import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

import torch.nn.functional as F
import torch.nn.init as init

class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
        self.apply(self.init_weights)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

    @staticmethod
    def init_weights(layer):
        if isinstance(layer, nn.Linear):
            init.xavier_uniform_(layer.weight)

class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-5, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.999):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = DQNetwork(state_size, action_size)
        self.target_model = DQNetwork(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)  # Learning rate scheduler

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, valid_moves, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.rand() <= epsilon:
            valid_actions = np.where(valid_moves == 1)[0]
            action = np.random.choice(valid_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
                masked_q_values = torch.where(torch.tensor(valid_moves, dtype=torch.bool), q_values, torch.tensor(-float('inf')))
                action = torch.argmax(masked_q_values).item()

        row, col = divmod(action, int(np.sqrt(self.action_size)))
        return row, col

    def memorize(self, state, action, reward, next_state, done):
        reward = max(min(reward, 1.0), -1.0)
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=256):
        if len(self.memory) < batch_size:
            return None  
        
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1).unsqueeze(1)
            next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze()

        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.smooth_l1_loss(q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step() 

        return loss  