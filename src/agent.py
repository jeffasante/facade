import numpy as np
import random
from collections import deque

import jax
import jax.numpy as jnp
from jax import random as jax_random
import flax.linen as nn
from flax.training import train_state
import optax

'''
Deep Q-Network agent using JAX/Flax with experience replay
'''

class QNetwork(nn.Module):
    """Deep Q-Network using JAX/Flax"""
    action_size: int

    @nn.compact
    def __call__(self, x, training=False):
        # Input layer
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.1)(x, deterministic=not training)

        # Hidden layers with residual connections for better training
        residual = x
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.1)(x, deterministic=not training)
        x = nn.Dense(128)(x)
        x = nn.relu(x + residual)  # residual connection

        # Output layer - Q-values for each action
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        q_values = nn.Dense(self.action_size)(x)

        return q_values


class DQNAgent:
    """JAX-based Deep Q-Network agent with experience replay"""

    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001, seed: int = 42):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.gamma = 0.99  # discount factor [High gamma = the agent avoids short-term gains that risk long-term pain.]
        self.memory = deque(maxlen=10000)

        # JAX setup
        self.key = jax_random.PRNGKey(seed)
        self.key, init_key = jax_random.split(self.key)

        # Initialize network
        self.q_network = QNetwork(action_size=action_size)

        # Initialize parameters with dummy input
        dummy_input = jnp.ones((1, state_size))
        self.params = self.q_network.init(init_key, dummy_input, training=False)

        # Create optimizer
        self.optimizer = optax.adam(learning_rate)
        self.train_state = train_state.TrainState.create(
            apply_fn=self.q_network.apply,
            params=self.params,
            tx=self.optimizer
        )

        # Target network for stable training
        self.target_params = self.params
        self.update_target_every = 100
        self.update_counter = 0

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        # Convert to JAX array and add batch dimension
        state_batch = jnp.expand_dims(jnp.array(state), 0)

        # Get Q-values from network (no dropout during inference)
        q_values = self.train_state.apply_fn(self.train_state.params, state_batch, training=False)

        return int(jnp.argmax(q_values[0]))

    def _update_step(self, params, target_params, states, actions, rewards, next_states, dones):
        """Update step for training"""

        def loss_fn(params):
            # Get new PRNG key for dropout
            self.key, dropout_key = jax_random.split(self.key)

            # Current Q-values (with dropout during training)
            current_q = self.train_state.apply_fn(params, states, training=True, rngs={'dropout': dropout_key})
            current_q = current_q[jnp.arange(len(actions)), actions]

            # Next Q-values from target network (no dropout)
            next_q = self.train_state.apply_fn(target_params, next_states, training=False)
            next_q_max = jnp.max(next_q, axis=1)

            # Target values
            targets = rewards + self.gamma * next_q_max * (1 - dones)

            # MSE loss
            loss = jnp.mean((current_q - targets) ** 2)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        new_state = self.train_state.apply_gradients(grads=grads)

        return new_state, loss

    def replay(self, batch_size=32):
        """Train the agent on a batch of experiences"""
        if len(self.memory) < batch_size:
            return None

        # Sample batch
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to JAX arrays
        states = jnp.array(states)
        actions = jnp.array(actions)
        rewards = jnp.array(rewards, dtype=jnp.float32)
        next_states = jnp.array(next_states)
        dones = jnp.array(dones, dtype=jnp.float32)

        # Update network
        self.train_state, loss = self._update_step(
           self.train_state.params, self.target_params,
            states, actions, rewards, next_states, dones,
        )

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.update_target_every == 0:
            self.target_params = self.train_state.params

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(loss)

    def save_model(self, filepath: str):
        """Save model parameters"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'params': self.train_state.params,
                'target_params': self.target_params,
                'epsilon': self.epsilon
            }, f)

    def load_model(self, filepath: str):
        """Load model parameters"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.train_state = self.train_state.replace(params=data['params'])
            self.target_params = data['target_params']
            self.epsilon = data['epsilon']