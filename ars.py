#!/usr/bin/env python3
# Augmented Random Search (ARS) V2

# Import the Libraries

import os
import numpy as np
import gym
from gym import wrappers
import pybullet_envs

# Set the Hyper Parameters

class HyperParameters():

    def __init__(self):
        self.number_steps = 500
        self.episode_length = 1000
        self.learning_rate = 0.02
        self.number_directions = 16
        self.number_best_directions = 16
        assert self.number_best_directions <= self.number_directions
        self.noise = 0.03
        self.seed = 1
        self.env_name = 'HalfCheetahBulletEnv-v0'

# Normalize the states

class Normalizer():

    def __init__(self, number_inputs):
        self.n = np.zeros(number_inputs)
        self.mean = np.zeros(number_inputs)
        self.mean_diff = np.zeros(number_inputs)
        self.var = np.zeros(number_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std

# Build the AI

class Policy():

    def __init__(self, input_size, output_size):
        self.theta = np.zeros((output_size, input_size))

    def evaluate(self, input, delta = None, direction = None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == "positive":
            return (self.theta + hp.noise * delta).dot(input)
        else:
            return (self.theta - hp.noise * delta).dot(input)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(hp.number_directions)]

    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += hp.learning_rate / (hp.number_best_directions * sigma_r) * step

# Explore the policy on one specific direction and over one episode

def explore(env, normalizer, policy, direction = None, delta = None):
    state = env.reset()
    done = False
    num_plays = 0.
    sum_rewards = 0
    while not done and num_plays < hp.episode_length:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        reward = max(min(reward, 5), -1)
        sum_rewards += reward
        num_plays += 1
    return sum_rewards

# Train the AI

def train(env, policy, normalizer, hp):
    for step in range(hp.number_steps):
        # Initialize the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.number_directions
        negative_rewards = [0] * hp.number_directions

        # Get the positive rewards in the positive directions
        for k in range(hp.number_directions):
            positive_rewards[k] = explore(env, normalizer, policy, direction = "positive", delta = deltas[k])

        # Get the negative rewards in the negative/opposite directions
        for k in range(hp.number_directions):
            negative_rewards[k] = explore(env, normalizer, policy, direction = "negative", delta = deltas[k])

        # Gather all positive/negative rewards to compute the standard deviation
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()

        # Sort the rollouts by the max(r_pos, r_neg) and select best directions
        scores = {k:max(r_pos, r_neg) for k,(r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(), key = lambda x:scores[x])[:hp.number_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

        # Update the policy
        policy.update(rollouts, sigma_r)

        # Print the final reward of the policy after the update
        reward_evaluation = explore(env, normalizer, policy)
        print ('Step: ', step, 'Reward: ', reward_evaluation)

# Run the main code
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')

hp = HyperParameters()
np.random.seed(hp.seed)
env = gym.make(hp.env_name)
env = wrappers.Monitor(env, monitor_dir, force = True)
nbr_inputs = env.observation_space.shape[0]
nbr_outputs = env.action_space.shape[0]
policy = Policy(nbr_inputs, nbr_outputs)
normalizer = Normalizer(nbr_inputs)
train(env, policy, normalizer, hp)
