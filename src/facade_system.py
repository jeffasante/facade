from .environment import AdEnvironment
from .agent import DQNAgent
import numpy as np
import random
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd

from .user import Ad, User

import jax.numpy as jnp


class FacadeSystem:
    """Main system orchestrating the RL ad recommendation"""

    def __init__(self):
        self.ads = self._create_sample_ads()
        self.users = self._create_sample_users()
        self.env = AdEnvironment(self.ads)
        self.agent = DQNAgent(state_size=14, action_size=len(self.ads) + 1)  # +1 for skip
        self.training_history = []

    def _create_sample_ads(self) -> List[Ad]:
        """Create sample ad database"""
        categories = ['sports', 'tech', 'fashion', 'food', 'travel']
        ads = []

        for i in range(20):  # 20 different ads
            ads.append(Ad(
                id=i,
                category=random.choice(categories),
                relevance_score=random.uniform(0.3, 0.9),
                click_value=random.uniform(1.0, 5.0),
                show_cost=0.1
            ))

        return ads

    def _create_sample_users(self) -> List[User]:
        """Create diverse user profiles"""
        users = []

        for i in range(100):
            # Random interest distribution
            interests = {}
            categories = ['sports', 'tech', 'fashion', 'food', 'travel']
            for cat in categories:
                interests[cat] = random.uniform(0.0, 1.0)

            users.append(User(
                id=f"user_{i}",
                interests=interests,
                patience_level=random.uniform(0.2, 0.9),
                session_history=[]
            ))

        return users

    def train(self, episodes: int = 1000, verbose: bool = True):
        """Train the RL agent with JAX acceleration"""
        episode_rewards = []
        losses = []

        for episode in range(episodes):
            # Random user for this episode
            user = random.choice(self.users)
            state = self.env.reset(user)
            total_reward = 0
            episode_loss = []

            while not self.env.done:
                action = self.agent.act(state)
                next_state, reward, done, info = self.env.step(action)

                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                # Train agent every few steps
                if len(self.agent.memory) > 100:
                    loss = self.agent.replay(batch_size=64)
                    if loss is not None:
                        episode_loss.append(loss)

                if done:
                    break

            episode_rewards.append(total_reward)
            if episode_loss:
                losses.append(np.mean(episode_loss))

            # Log progress
            if verbose and episode % 100 == 0:
                recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
                avg_reward = np.mean(recent_rewards)
                recent_loss = losses[-10:] if len(losses) >= 10 else losses
                avg_loss = np.mean(recent_loss) if recent_loss else 0
                print(f"Episode {episode:4d} | Avg Reward: {avg_reward:6.2f} | Loss: {avg_loss:.4f} | ε: {self.agent.epsilon:.3f}")

        self.training_history = episode_rewards
        return episode_rewards, losses

    def evaluate_policy(self, episodes: int = 100) -> Dict:
        """Evaluate trained policy vs baseline with detailed metrics"""
        results = {
            'rl_rewards': [],
            'baseline_rewards': [],
            'random_rewards': [],
            'rl_dropouts': 0,
            'baseline_dropouts': 0,
            'random_dropouts': 0,
            'rl_session_lengths': [],
            'baseline_session_lengths': [],
            'random_session_lengths': [],
            'rl_ad_counts': [],
            'baseline_ad_counts': [],
            'random_ad_counts': [],
            'rl_click_rates': [],
            'baseline_click_rates': [],
            'random_click_rates': []
        }

        for episode in range(episodes):
            user = random.choice(self.users)

            # Test RL policy
            state = self.env.reset(user)
            total_reward = 0
            clicks = 0
            ads_shown = 0
            steps = 0
            while not self.env.done:
                action = self.agent.act(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                total_reward += reward
                steps += 1
                if action < len(self.ads):  # ad was shown
                    ads_shown += 1
                    if info.get('action') == 'click':
                        clicks += 1
                if info.get('dropout'):
                    results['rl_dropouts'] += 1

            results['rl_rewards'].append(total_reward)
            results['rl_session_lengths'].append(steps)
            results['rl_ad_counts'].append(ads_shown)
            results['rl_click_rates'].append(clicks / max(1, ads_shown))

            # Test baseline (aggressive) policy - always show highest value ads
            state = self.env.reset(user)
            total_reward = 0
            clicks = 0
            ads_shown = 0
            steps = 0
            while not self.env.done:
                # Pick highest value ad
                action = max(range(len(self.ads)), key=lambda i: self.ads[i].click_value)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                total_reward += reward
                steps += 1
                ads_shown += 1
                if info.get('action') == 'click':
                    clicks += 1
                if info.get('dropout'):
                    results['baseline_dropouts'] += 1

            results['baseline_rewards'].append(total_reward)
            results['baseline_session_lengths'].append(steps)
            results['baseline_ad_counts'].append(ads_shown)
            results['baseline_click_rates'].append(clicks / max(1, ads_shown))

            # Test random policy
            state = self.env.reset(user)
            total_reward = 0
            clicks = 0
            ads_shown = 0
            steps = 0
            while not self.env.done:
                action = random.randrange(len(self.ads) + 1)  # include skip action
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                total_reward += reward
                steps += 1
                if action < len(self.ads):
                    ads_shown += 1
                    if info.get('action') == 'click':
                        clicks += 1
                if info.get('dropout'):
                    results['random_dropouts'] += 1

            results['random_rewards'].append(total_reward)
            results['random_session_lengths'].append(steps)
            results['random_ad_counts'].append(ads_shown)
            results['random_click_rates'].append(clicks / max(1, ads_shown))

        return results

    def plot_training_progress(self, rewards: List[float], losses: List[float], save_path: str = None):
        """Plot training metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Facade Training Progress', fontsize=16, fontweight='bold')

        # Reward progression
        episodes = range(len(rewards))
        window = min(50, len(rewards) // 10)
        smoothed_rewards = pd.Series(rewards).rolling(window, center=True).mean()

        ax1.plot(episodes, rewards, alpha=0.3, color='lightblue', label='Raw')
        ax1.plot(episodes, smoothed_rewards, color='darkblue', linewidth=2, label=f'MA({window})')
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Loss progression
        if losses:
            ax2.plot(losses, color='red', linewidth=1.5)
            ax2.set_title('Training Loss')
            ax2.set_xlabel('Training Steps')
            ax2.set_ylabel('MSE Loss')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)

        # Reward distribution
        ax3.hist(rewards[-200:], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(np.mean(rewards[-200:]), color='red', linestyle='--',
                   label=f'Mean: {np.mean(rewards[-200:]):.2f}')
        ax3.set_title('Recent Reward Distribution')
        ax3.set_xlabel('Reward')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Epsilon decay
        epsilons = [1.0 * (0.995 ** i) for i in range(len(rewards))]
        epsilons = [max(0.01, e) for e in epsilons]  # clip to min
        ax4.plot(epsilons, color='orange', linewidth=2)
        ax4.set_title('Exploration Rate (ε)')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_policy_comparison(self, results: Dict, save_path: str = None):
        """Compare RL policy against baselines"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Facade Policy Comparison: RL vs Baselines', fontsize=16, fontweight='bold')

        policies = ['RL Agent', 'Aggressive', 'Random']
        colors = ['#2E86AB', '#A23B72', '#F18F01']

        # Reward comparison
        ax = axes[0, 0]
        rewards_data = [results['rl_rewards'], results['baseline_rewards'], results['random_rewards']]
        bp = ax.boxplot(rewards_data, labels=policies, patch_artist=True, showmeans=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title('Total Reward per Session')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)

        # Session length comparison
        ax = axes[0, 1]
        length_data = [results['rl_session_lengths'], results['baseline_session_lengths'], results['random_session_lengths']]
        bp = ax.boxplot(length_data, labels=policies, patch_artist=True, showmeans=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title('Session Length (Steps)')
        ax.set_ylabel('Steps')
        ax.grid(True, alpha=0.3)

        # Click rate comparison
        ax = axes[0, 2]
        click_data = [results['rl_click_rates'], results['baseline_click_rates'], results['random_click_rates']]
        bp = ax.boxplot(click_data, labels=policies, patch_artist=True, showmeans=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title('Click Rate per Session')
        ax.set_ylabel('Click Rate')
        ax.grid(True, alpha=0.3)

        # Dropout comparison
        ax = axes[1, 0]
        dropouts = [results['rl_dropouts'], results['baseline_dropouts'], results['random_dropouts']]
        bars = ax.bar(policies, dropouts, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('Total Dropouts')
        ax.set_ylabel('Dropouts')
        # Add value labels on bars
        for bar, value in zip(bars, dropouts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(value), ha='center', va='bottom', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Performance metrics
        ax = axes[1, 1]
        metrics = ['Avg Reward', 'Avg Session Length', 'Avg Click Rate']
        rl_metrics = [np.mean(results['rl_rewards']),
                     np.mean(results['rl_session_lengths']),
                     np.mean(results['rl_click_rates'])]
        baseline_metrics = [np.mean(results['baseline_rewards']),
                          np.mean(results['baseline_session_lengths']),
                          np.mean(results['baseline_click_rates'])]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax.bar(x - width/2, rl_metrics, width, label='RL Agent', color=colors[0], alpha=0.7)
        bars2 = ax.bar(x + width/2, baseline_metrics, width, label='Aggressive', color=colors[1], alpha=0.7)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Revenue vs Retention scatter
        ax = axes[1, 2]
        ax.scatter(results['rl_session_lengths'], results['rl_rewards'],
                  color=colors[0], alpha=0.6, label='RL Agent', s=30)
        ax.scatter(results['baseline_session_lengths'], results['baseline_rewards'],
                  color=colors[1], alpha=0.6, label='Aggressive', s=30)
        ax.scatter(results['random_session_lengths'], results['random_rewards'],
                  color=colors[2], alpha=0.6, label='Random', s=30)
        ax.set_xlabel('Session Length')
        ax.set_ylabel('Total Reward')
        ax.set_title('Revenue vs Retention Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_user_journey_analysis(self, save_path: str = None):
        """Analyze user journey patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Facade User Journey Analysis', fontsize=16, fontweight='bold')

        # Sample user journeys
        sample_users = random.sample(self.users, 5)

        # User interest heatmap
        ax = axes[0, 0]
        categories = ['sports', 'tech', 'fashion', 'food', 'travel']
        interest_matrix = []
        for user in sample_users[:10]:  # First 10 users
            user_interests = [user.interests.get(cat, 0) for cat in categories]
            interest_matrix.append(user_interests)

        im = ax.imshow(interest_matrix, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45)
        ax.set_ylabel('User ID')
        ax.set_title('User Interest Profiles')
        plt.colorbar(im, ax=ax)

        # Ad category distribution
        ax = axes[0, 1]
        ad_categories = [ad.category for ad in self.ads]
        category_counts = {cat: ad_categories.count(cat) for cat in categories}
        ax.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
        ax.set_title('Ad Inventory Distribution')

        # Simulate fatigue progression
        ax = axes[1, 0]
        user = random.choice(sample_users)
        state = self.env.reset(user)
        fatigue_levels = [user.current_fatigue]
        rewards = []

        for _ in range(20):  # Simulate 20 actions
            if self.env.done:
                break
            action = self.agent.act(state, training=False)
            next_state, reward, done, info = self.env.step(action)
            fatigue_levels.append(user.current_fatigue)
            rewards.append(reward)
            state = next_state

        ax.plot(fatigue_levels, 'r-', linewidth=2, label='Fatigue Level')
        ax.set_ylabel('Fatigue Level', color='r')
        ax.tick_params(axis='y', labelcolor='r')
        ax.set_title('User Fatigue Progression')

        # Add reward on secondary y-axis
        ax2 = ax.twinx()
        ax2.bar(range(1, len(rewards) + 1), rewards, alpha=0.6, color='blue', label='Rewards')
        ax2.set_ylabel('Reward', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        ax.set_xlabel('Action Step')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # Q-value visualization for sample state
        ax = axes[1, 1]
        sample_state = self._get_sample_state()
        q_values = self.agent.train_state.apply_fn(
            self.agent.train_state.params,
            jnp.expand_dims(sample_state, 0)
        )[0]

        ad_labels = [f"Ad{i}" for i in range(len(self.ads))] + ["Skip"]
        bars = ax.bar(range(len(q_values)), q_values, color='skyblue', edgecolor='navy')
        ax.set_title('Q-Values for Sample State')
        ax.set_xlabel('Action')
        ax.set_ylabel('Q-Value')
        ax.set_xticks(range(0, len(q_values), max(1, len(q_values)//10)))

        # Highlight best action
        best_action = int(jnp.argmax(q_values))
        bars[best_action].set_color('gold')
        bars[best_action].set_edgecolor('orange')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def _get_sample_state(self):
        """Get a sample state for Q-value visualization"""
        user = random.choice(self.users)
        return self.env.reset(user)

    def generate_comprehensive_report(self, rewards: List[float], losses: List[float],
                                    results: Dict, save_dir: str = "./facade_results/"):
        """Generate all plots and save them"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        print("Generating comprehensive analysis plots...")

        # Training progress
        self.plot_training_progress(rewards, losses, f"{save_dir}/training_progress.png")

        # Policy comparison
        self.plot_policy_comparison(results, f"{save_dir}/policy_comparison.png")

        # User journey analysis
        self.plot_user_journey_analysis(f"{save_dir}/user_journey.png")

        # Generate summary stats
        summary = {
            "training_episodes": len(rewards),
            "final_avg_reward": float(np.mean(rewards[-100:])),
            "rl_vs_aggressive_improvement": float((np.mean(results['rl_rewards']) - np.mean(results['baseline_rewards'])) / abs(np.mean(results['baseline_rewards'])) * 100),
            "dropout_reduction": results['baseline_dropouts'] - results['rl_dropouts'],
            "final_epsilon": float(self.agent.epsilon)
        }

        with open(f"{save_dir}/summary_stats.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"All plots saved to {save_dir}")
        print(f"Summary stats: {summary}")

        return summary
