'''
Facade - JAX-Powered Adaptive Ad Recommendation System
Comprehensive demo script to initialize, train, evaluate, and analyze the system.
'''

from facade.facade_system import FacadeSystem

import numpy as np

def main():
    """Demo the Facade system with comprehensive analysis"""
    print("Initializing Facade - JAX-Powered Adaptive Ad Recommendation System")

    system = FacadeSystem()

    print(f"System Overview:")
    print(f"  • {len(system.ads)} ads across {len(set(ad.category for ad in system.ads))} categories")
    print(f"  • {len(system.users)} diverse user profiles")
    print(f"  • State space: {system.agent.state_size} dimensions")
    print(f"  • Action space: {system.agent.action_size} actions (ads + skip)")

    print(f"\nNeural Network Architecture:")
    print(f"  • Input: {system.agent.state_size} → Dense(128) → Dense(128) → Dense(64) → Output({system.agent.action_size})")
    print(f"  • Optimizer: Adam (lr={system.agent.learning_rate})")
    print(f"  • Experience replay buffer: {system.agent.memory.maxlen} experiences")

    print(f"\nTraining RL agent...")

    # Train the agent
    rewards, losses = system.train(episodes=1000, verbose=True)
    print(f"\nTraining completed!")
    print(f"  • Final epsilon: {system.agent.epsilon:.4f}")
    print(f"  • Avg reward (last 100): {np.mean(rewards[-100:]):.2f}")
    if losses:
        print(f"  • Avg loss (last 10): {np.mean(losses[-10:]):.4f}")

    print(f"\nEvaluating performance against baselines...")
    results = system.evaluate_policy(episodes=200)

    rl_avg = np.mean(results['rl_rewards'])
    baseline_avg = np.mean(results['baseline_rewards'])
    random_avg = np.mean(results['random_rewards'])
    improvement_vs_aggressive = ((rl_avg - baseline_avg) / abs(baseline_avg)) * 100
    improvement_vs_random = ((rl_avg - random_avg) / abs(random_avg)) * 100

    print(f"\nResults Summary:")
    print(f"  RL Agent:      Reward: {rl_avg:6.2f} | Dropouts: {results['rl_dropouts']:2d} | Avg Session: {np.mean(results['rl_session_lengths']):.1f}")
    print(f"  Aggressive:    Reward: {baseline_avg:6.2f} | Dropouts: {results['baseline_dropouts']:2d} | Avg Session: {np.mean(results['baseline_session_lengths']):.1f}")
    print(f"  Random:        Reward: {random_avg:6.2f} | Dropouts: {results['random_dropouts']:2d} | Avg Session: {np.mean(results['random_session_lengths']):.1f}")
    print(f"\nImprovements:")
    print(f"  vs Aggressive: {improvement_vs_aggressive:+6.1f}% reward | {results['baseline_dropouts'] - results['rl_dropouts']:+2d} fewer dropouts")
    print(f"  vs Random:     {improvement_vs_random:+6.1f}% reward | {results['random_dropouts'] - results['rl_dropouts']:+2d} fewer dropouts")

    # Generate comprehensive analysis
    print(f"\nGenerating visual analysis...")
    summary_stats = system.generate_comprehensive_report(rewards, losses, results)

    # Save the trained model
    print(f"\nSaving trained model...")
    system.agent.save_model("facade_model.pkl")
    print(f"  Model saved to 'facade_model.pkl'")

    print(f"\nFacade analysis complete! Check the generated plots to verify performance.")
    print(f"All results saved in './facade_results/' directory")

    return system, rewards, losses, results, summary_stats

if __name__ == "__main__":
    facade_system, training_rewards, training_losses, eval_results, stats = main()