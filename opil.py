#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class of reward agent
"""

import tensorflow.compat.v2 as tf

class opil(object):
    def __init__(self, state_dim, action_dim):
        self.reward_function = tf.keras.Sequential([
            tf.keras.layers.Dense(
                256, input_shape=(state_dim + action_dim,), activation=tf.nn.tanh),
            tf.keras.layers.Dense(256, activation=tf.nn.tanh),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self.optimizer = tf.keras.optimizers.Adam()
        
    def get_reward(self, states, actions):
        inputs = tf.concat([states, actions], -1)
        return self.reward_function(inputs)
        
    def update(self, expert_dataset_iter, reward_replay_buffer_iter):
        expert_states, expert_actions, _ = next(expert_dataset_iter)
        policy_states, policy_actions, _, _, _ = next(reward_replay_buffer_iter)[0]
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.reward_function.variables)
            expert_inputs = tf.concat([expert_states, expert_actions], -1)
            expert_rewards = tf.reduce_mean(self.reward_function(expert_inputs))
            agent_inputs = tf.concat([policy_states, policy_actions], -1)
            agent_rewards = tf.reduce_mean(self.reward_function(agent_inputs))

            # project
            loss = agent_rewards - expert_rewards 

        grads = tape.gradient(loss, self.reward_function.variables)

        self.optimizer.apply_gradients(zip(grads, self.reward_function.variables))
        