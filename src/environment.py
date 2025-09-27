'''
Facade - JAX-Powered Adaptive Ad Recommendation System

Custom environment for ad recommendation RL
'''

from .user import Ad, User, UserSimulator
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


class AdEnvironment:
    """Custom environment for ad recommendation RL"""

    def __init__(self, ads: List[Ad], max_ads_per_session: int = 10):
        self.ads = ads
        self.max_ads_per_session = max_ads_per_session
        self.current_user = None
        self.current_simulator = None
        self.ads_shown_this_session = 0
        self.session_reward = 0
        self.done = False

    def reset(self, user: User) -> np.ndarray:
        """Reset environment for new user session"""
        self.current_user = user
        self.current_simulator = UserSimulator(user)
        self.current_simulator.reset_session()
        self.ads_shown_this_session = 0
        self.session_reward = 0
        self.done = False
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action in environment
        action: ad_id to show, or len(ads) for "skip"
        """
        if self.done:
            return self._get_state(), 0, True, {}

        step_penalty = -0.6 # Add a small cost for every step to encourage efficiency
        reward = step_penalty
        info = {}

        if action < len(self.ads):  # Show ad
            ad = self.ads[action]

            # Check if showing duplicate recent ad
            if action in self.current_user.session_history[-3:]:
                reward = -5  # penalty for repetitive ads
                info['action'] = 'duplicate_penalty'
            else:
                # Simulate user interaction
                user_action, ad_reward = self.current_simulator.interact_with_ad(ad)
                reward = ad_reward
                info['action'] = user_action
                self.current_user.session_history.append(action)

                # Add a penalty for increasing fatigue
                '''
                If the ad is likely to be relevant → show it.
                If not → skip it to avoid racking up penalties.
                '''
                fatigue_penalty = self.current_user.current_fatigue * 0.5 # The 0.5 is a tunable factor
                reward -= fatigue_penalty

            self.ads_shown_this_session += 1

        else:  # Skip showing ad
            '''
            Why it works: This makes the Q-value for the "skip" action higher, encouraging the agent to choose it more often,
            especially when the user's fatigue is rising and the chance of a click is low.
            '''
            reward += 0.0
            info['action'] = 'skip'

        # Check if user drops out
        if self.current_simulator.should_dropout():
            reward -= 20  # large penalty for dropout
            self.done = True
            info['dropout'] = True

        # Check session limits
        if self.ads_shown_this_session >= self.max_ads_per_session:
            self.done = True
            info['session_complete'] = True

        self.session_reward += reward
        next_state = self._get_state()

        return next_state, reward, self.done, info

    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        if self.current_user is None:
            return np.zeros(10)  # placeholder

        # Create state vector
        state = []

        # User interests (top 5 categories)
        categories = ['sports', 'tech', 'fashion', 'food', 'travel']
        for cat in categories:
            state.append(self.current_user.interests.get(cat, 0.0))

        # User fatigue and session info
        state.extend([
            self.current_user.current_fatigue,
            self.ads_shown_this_session / self.max_ads_per_session,
            len(self.current_user.session_history) / 20,  # normalized history length
        ])

        # Tell the agent about the user's personality
        '''
        This allows the agent to learn different strategies for different types of users.
        It might learn to show more ads to patient users and to be more cautious with impatient ones.
        '''
        state.append(self.current_user.patience_level)

        # Recent ad categories (one-hot encoding of last 2 ads)
        recent_ads = self.current_user.session_history[-2:]
        recent_categories = [0] * len(categories)
        for ad_id in recent_ads:
            if ad_id < len(self.ads):
                ad_category = self.ads[ad_id].category
                if ad_category in categories:
                    recent_categories[categories.index(ad_category)] = 1

        state.extend(recent_categories)

        return np.array(state, dtype=np.float32)