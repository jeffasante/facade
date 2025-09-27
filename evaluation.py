import numpy as np
import argparse
import os
from src.facade_system import FacadeSystem

def display_evaluation_results(results: dict):
    """Helper function to print the evaluation summary."""
    rl_avg = np.mean(results['rl_rewards'])
    baseline_avg = np.mean(results['baseline_rewards'])
    random_avg = np.mean(results['random_rewards'])
    
    improvement_vs_aggressive = ((rl_avg - baseline_avg) / abs(baseline_avg) * 100) if baseline_avg != 0 else float('inf')
    improvement_vs_random = ((rl_avg - random_avg) / abs(random_avg) * 100) if random_avg != 0 else float('inf')

    print("\n" + "="*20 + " Evaluation Results " + "="*20)
    print(f"  RL Agent:      Reward: {rl_avg:6.2f} | Dropouts: {results['rl_dropouts']:3d} | Avg Session: {np.mean(results['rl_session_lengths']):.1f}")
    print(f"  Aggressive:    Reward: {baseline_avg:6.2f} | Dropouts: {results['baseline_dropouts']:3d} | Avg Session: {np.mean(results['baseline_session_lengths']):.1f}")
    print(f"  Random:        Reward: {random_avg:6.2f} | Dropouts: {results['random_dropouts']:3d} | Avg Session: {np.mean(results['random_session_lengths']):.1f}")
    print("="*60)


def main():
    """Loads a pre-trained agent and evaluates its performance."""
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained Facade RL agent.")
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        default='facade_model.pkl', 
        help='Path to the model file to load for evaluation.'
    )
    args = parser.parse_args()

    print("--- Running in EVALUATION mode ---")
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at '{args.model_path}'")
        return

    system = FacadeSystem()
    
    print(f"\nLoading trained model from '{args.model_path}'...")
    system.agent.load_model(args.model_path)
    print("Model loaded successfully.")
    
    print(f"\nEvaluating performance against baselines over 200 episodes...")
    results = system.evaluate_policy(episodes=200)
    
    display_evaluation_results(results)

    print(f"\nGenerating visual analysis...")
    os.makedirs("./facade_results_main", exist_ok=True)
    system.plot_policy_comparison(results, save_path="./facade_results_main/evaluation_policy_comparison.png")
    system.plot_user_journey_analysis(save_path="./facade_results_main/evaluation_user_journey.png")
    print(f"\nEvaluation complete! Check the './facade_results_main/' directory for plots.")

if __name__ == "__main__":
    main()