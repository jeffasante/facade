'''
Facade - JAX-Powered Adaptive Ad Recommendation System 

User profile and ad definitions, along with user behavior simulation.
'''
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class User:
    """User profile with interests and fatigue tracking"""
    id: str
    interests: Dict[str, float]  # category -> interest_score (0-1)
    patience_level: float  # base tolerance for ads (0-1)
    current_fatigue: float = 0.0
    session_history: List[int] = None
    clicks_today: int = 0
    sessions_today: int = 0

    def __post_init__(self):
        if self.session_history is None:
            self.session_history = []

@dataclass
class Ad:
    """Advertisement with targeting info"""
    id: int
    category: str
    relevance_score: float  # base quality score (0-1)
    click_value: float  # revenue per click
    show_cost: float = 0.1  # cost to show (impressions)

class UserSimulator:
    """Simulates realistic user behavior and fatigue"""

    def __init__(self, user: User):
        self.user = user
        self.base_dropout_prob = 0.05

    def interact_with_ad(self, ad: Ad) -> Tuple[str, float]:
        """
        Simulate user interaction with ad
        Returns: (action, reward_contribution)
        """
        # Calculate relevance based on user interests
        interest_match = self.user.interests.get(ad.category, 0.0)
        relevance = (ad.relevance_score + interest_match) / 2

        # Fatigue affects interaction probability
        fatigue_penalty = self.user.current_fatigue * 0.3
        click_prob = max(0, relevance - fatigue_penalty)

        # Determine user action
        if random.random() < click_prob:
            action = "click"
            self.user.clicks_today += 1
            reward = ad.click_value
            self.user.current_fatigue = max(0, self.user.current_fatigue - 0.05)  # clicking reduces fatigue slightly
        elif random.random() < 0.3:  # 30% chance of ignoring
            action = "ignore"
            reward = -ad.show_cost
            self.user.current_fatigue += 0.1
        else:
            action = "view"
            reward = -ad.show_cost * 0.5
            self.user.current_fatigue += 0.05

        return action, reward

    def should_dropout(self) -> bool:
        """Determine if user drops out of session"""
        dropout_prob = self.base_dropout_prob + (self.user.current_fatigue * 0.4)
        return random.random() < dropout_prob

    def reset_session(self):
        """Reset for new session"""
        self.user.current_fatigue *= 0.7  # carry over some fatigue
        self.user.sessions_today += 1
