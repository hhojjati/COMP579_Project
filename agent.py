import numpy as np
import torch
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, reward, next_state, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.done[self.ptr] = done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		idx = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[idx]).to(self.device),
			torch.FloatTensor(self.action[idx]).to(self.device),
			torch.FloatTensor(self.reward[idx]).to(self.device),
			torch.FloatTensor(self.next_state[idx]).to(self.device),
			torch.FloatTensor(self.done[idx]).to(self.device)
		)
  
# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
  def __init__(self, state_dim, action_dim, max_action):
    super(Actor, self).__init__()

    self.fc1 = nn.Linear(state_dim, 256)
    self.fc2 = nn.Linear(256, 256)
    self.fc3 = nn.Linear(256, action_dim)
    
    self.max_action = max_action
    

  def forward(self, state):
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    x = torch.tanh(self.fc3(x))
    return self.max_action * x


class Critic(nn.Module):
  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()

    # Q1:
    self.fc1 = nn.Linear(state_dim + action_dim, 256)
    self.fc2 = nn.Linear(256, 256)
    self.fc3 = nn.Linear(256, 1)

    # Q2:
    self.fc4 = nn.Linear(state_dim + action_dim, 256)
    self.fc5 = nn.Linear(256, 256)
    self.fc6 = nn.Linear(256, 1)


  def forward(self, state, action):
    x = torch.cat([state, action], 1)

    q1 = F.relu(self.fc1(x))
    q1 = F.relu(self.fc2(q1))
    q1 = self.fc3(q1)

    q2 = F.relu(self.fc4(x))
    q2 = F.relu(self.fc5(q2))
    q2 = self.fc6(q2)
    return q1, q2


  def Q1(self, state, action):
    x = torch.cat([state, action], 1)

    q1 = F.relu(self.fc1(x))
    q1 = F.relu(self.fc2(q1))
    q1 = self.fc3(q1)
    return q1



class Agent():
  def __init__(
    self,
    env_specs,
    gamma=0.99,
    tau=0.005,
    policy_noise=0.2,
    noise_clip=0.5,
    policy_freq=2,
    expl_noise = 0.1,
    T_train=25e3,
    batch_size=256
  ):

    self.action_space = env_specs['action_space']
    self.observation_space = env_specs['observation_space']
    self.state_dim = self.observation_space.shape[0]
    self.action_dim = self.action_space.shape[0] 
    self.max_action = float(self.action_space.high[0])
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
    self.actor_target = copy.deepcopy(self.actor)
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

    self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
    self.critic_target = copy.deepcopy(self.critic)
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
    
    self.gamma = gamma
    self.tau = tau
    self.policy_noise = policy_noise*self.max_action
    self.noise_clip = noise_clip*self.max_action
    self.policy_freq = policy_freq
    self.expl_noise = expl_noise
    self.T_train=T_train
    self.batch_size = batch_size

    self.act_ts = 0
    self.iters = 0

    self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim)
    
  def load_weights(self,root_path): ###
    filename = root_path + f"./{'Hopper'}" ###
    self.critic.load_state_dict(torch.load(filename + "_critic"))
    self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
    self.critic_target = copy.deepcopy(self.critic)

    self.actor.load_state_dict(torch.load(filename + "_actor"))
    self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
    self.actor_target = copy.deepcopy(self.actor)
    pass

  def act(self, curr_obs, mode='eval'):
    self.act_ts += 1
    if mode == 'eval':
      self.actor.eval()
      state = torch.FloatTensor(curr_obs.reshape(1, -1)).to(self.device)
      # if self.act_ts < self.T_train:
      #   action = self.action_space.sample()
      # else:
      action = (self.actor(state).cpu().data.numpy().flatten() + np.random.normal(0,self.max_action*self.expl_noise,size=self.action_dim)).clip(-self.max_action,self.max_action)
    else:
      state = torch.FloatTensor(curr_obs.reshape(1, -1)).to(self.device)
      if self.act_ts < self.T_train:
        action = self.action_space.sample()
      else:
        action = (self.actor(state).cpu().data.numpy().flatten() + np.random.normal(0,self.max_action*self.expl_noise,size=self.action_dim)).clip(-self.max_action,self.max_action)
    
    return action


  def update(self, curr_obs, action, reward1, next_obs, done,t):

    self.replay_buffer.add(curr_obs, action, reward1, next_obs, done)

    if t>self.T_train:
    
      self.iters += 1

      # Sample replay buffer 
      state, actions,rewards, next_state, dones = self.replay_buffer.sample(self.batch_size)

      with torch.no_grad():

        # Take Action
        N = (torch.randn(actions.shape) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_state) + N.to(self.device))
        next_action = next_action.clamp(-self.max_action, self.max_action)

        # Compute the target Q value
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = rewards + (1-dones) * self.gamma * target_Q

      Q1_Val, Q2_Val = self.critic(state, actions)

      # Critic loss and optimization
      critic_loss = F.mse_loss(Q1_Val, target_Q) + F.mse_loss(Q2_Val, target_Q)
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()

      
      if self.iters % self.policy_freq == 0: # Delayed policy updates

        # Actor Loss and optimization:
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks:
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


  def save(self, filename= f"./{'Hopper'}"):
    torch.save(self.critic.state_dict(), filename + "_critic")
    torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
    
    torch.save(self.actor.state_dict(), filename + "_actor")
    torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    
