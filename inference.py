import argparse
import os
import numpy as np

# Import necessary classes from the src directory
from src.facade_system import FacadeSystem
from src.user import User

def main():
    """Loads a pre-trained agent and performs a single inference for a sample user."""
    parser = argparse.ArgumentParser(description="Perform a single inference with a pre-trained Facade agent.")
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        default='facade_model.pkl', 
        help='Path to the model file to load for inference.'
    )
    args = parser.parse_args()

    print("--- Running in INFERENCE demo mode ---")
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at '{args.model_path}'")
        return

    # Initialize the system, which contains the agent and environment
    system = FacadeSystem()
    
    print(f"\nLoading trained model from '{args.model_path}'...")
    system.agent.load_model(args.model_path)
    print("Model loaded successfully.")

    # --- Create a Sample User for the Demo ---
    # Randomize interests for varied demos
    categories = ['sports', 'tech', 'fashion', 'food', 'travel']
    interests = {cat: np.random.random() for cat in categories}
    sample_user_profile = {
        "id": "demo_user_123",
        "interests": interests,
        "patience_level": np.random.uniform(0.3, 0.9)
    }
    sample_user = User(**sample_user_profile)
    # Find top 2 interests for printing
    top_interests = sorted(interests.items(), key=lambda x: x[1], reverse=True)[:2]
    interest_names = [name for name, _ in top_interests]
    print(f"\nCreated a sample user with interests in '{interest_names[0]}' and '{interest_names[1]}'.")

    # --- Simulate the Start of a Session to Get a State ---
    # The 'reset' method prepares the environment for the new user and returns the initial state
    state = system.env.reset(sample_user)
    print(f"\nSimulating start of session. Initial state for the agent:")
    print(state)

    # --- Perform Inference ---
    # Use the agent's 'act' method to get the best action for the current state
    # training=False ensures the agent only uses its learned knowledge
    action_id = system.agent.act(state, training=False)
    
    # --- Translate the Action ID into a Human-Readable Decision ---
    print("\n" + "="*20 + " Agent's Decision " + "="*20)
    
    if action_id < len(system.ads):
        # The action is to show an ad
        chosen_ad = system.ads[action_id]
        print(f"Recommended Action: Show Ad #{chosen_ad.id}")
        print(f"  - Category: {chosen_ad.category}")
        print(f"  - Relevance Score: {chosen_ad.relevance_score}")
        print(f"  - Click Value: ${chosen_ad.click_value:.2f}")
    else:
        # The action is to skip
        print("Recommended Action: Skip (Do not show an ad)")
        print("  - Reason: The agent determined that showing an ad now is suboptimal,")
        print("    likely to avoid increasing user fatigue or because no ad is a good match.")
        
    print("="*58)

if __name__ == "__main__":
    main()