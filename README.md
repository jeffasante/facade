# Facade

An adaptive ad recommendation system using Deep Reinforcement Learning.

## Description

This project simulates an ad environment where a Deep Q-Network (DQN) agent learns to make optimal decisions about ad placements, balancing immediate revenue with long-term user engagement.

![Policy Comparison](results/facade_results/policy_comparison.png)

## Installation

1. Clone the repository.
2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training a Model

Run the training script:

```bash
python main.py
```

This trains the DQN agent and saves the model to `facade_model.pkl`.

### Running Inference

Use the inference script:

```bash
python inference.py
```

Example outputs:

**Demo 1:**

```text
--- Running in INFERENCE demo mode ---

Loading trained model from 'facade_model.pkl'...
Model loaded successfully.

Created a sample user with interests in 'sports' and 'travel'.

Simulating start of session. Initial state for the agent:
[0.8 0.1 0.2 0.4 0.7 0.  0.  0.  0.6 0.  0.  0.  0.  0. ]

==================== Agent's Decision ====================
Recommended Action: Show Ad #8
  - Category: sports
  - Relevance Score: 0.30278881708125654
  - Click Value: $2.81
==========================================================
```

**Demo 2:**

```text
--- Running in INFERENCE demo mode ---

Loading trained model from 'facade_model.pkl'...
Model loaded successfully.

Created a sample user with interests in 'sports' and 'travel'.

Simulating start of session. Initial state for the agent:
[0.8 0.1 0.2 0.4 0.7 0.  0.  0.  0.6 0.  0.  0.  0.  0. ]

==================== Agent's Decision ====================
Recommended Action: Show Ad #8
  - Category: food
  - Relevance Score: 0.35637223436385024
  - Click Value: $1.78
==========================================================
```

**Demo 3:**

```text
--- Running in INFERENCE demo mode ---

Loading trained model from 'facade_model.pkl'...
Model loaded successfully.

Created a sample user with interests in 'sports' and 'travel'.

Simulating start of session. Initial state for the agent:
[0.8 0.1 0.2 0.4 0.7 0.  0.  0.  0.6 0.  0.  0.  0.  0. ]

==================== Agent's Decision ====================
Recommended Action: Show Ad #8
  - Category: travel
  - Relevance Score: 0.34556114008612815
  - Click Value: $1.97
==========================================================
```

## Features

- Custom simulated environment for ad-user interactions
- DQN agent implemented with JAX/Flax
- User simulation with interest profiles and fatigue
- Evaluation against baseline policies
- Analysis and visualization of results

## Project Structure

- `src/`: Core source code (agent, environment, user, system)
- `results/`: Output plots and statistics
- `docs/`: Research documentation

## Documentation

For more detailed information:

- [Research](docs/RESEARCH.md)
- [Summary of Experimental Phases](docs/summary-of-experimental-phases.md)

## Results

Training results, including plots and statistics, are saved in `results/facade_results/`. Key outputs include policy comparisons, training progress, and user journey analysis.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.