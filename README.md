# Facade

An adaptive ad recommendation system using Deep Reinforcement Learning.

## Description

This project simulates an ad environment where a Deep Q-Network (DQN) agent learns to make optimal decisions about ad placements, balancing immediate revenue with long-term user engagement.

![Policy Comparison](results/facade_results/policy_comparison.png)

## Try it in Google Colab

For a quick demo without installation, try the interactive notebook:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15yyhranLEL_CsIT3H2HamYBimuqqKSk6?usp=sharing)

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

Created a sample user with interests in 'tech' and 'food'.

Simulating start of session. Initial state for the agent:
[0.0721851  0.873756   0.36416262 0.7099551  0.48997286 0.
 0.         0.         0.3608316  0.         0.         0.
 0.         0.        ]

==================== Agent's Decision ====================
Recommended Action: Show Ad #18
  - Category: fashion
  - Relevance Score: 0.5925505232093682
  - Click Value: $1.48
==========================================================
```

**Demo 2:**

```text
--- Running in INFERENCE demo mode ---

Loading trained model from 'facade_model.pkl'...
Model loaded successfully.

Created a sample user with interests in 'food' and 'travel'.

Simulating start of session. Initial state for the agent:
[0.26137498 0.19076613 0.14005695 0.9419397  0.6099581  0.
 0.         0.         0.70459545 0.         0.         0.
 0.         0.        ]

==================== Agent's Decision ====================
Recommended Action: Show Ad #5
  - Category: travel
  - Relevance Score: 0.7119103721059065
  - Click Value: $3.10
==========================================================
```

**Demo 3:**

```text
--- Running in INFERENCE demo mode ---

Loading trained model from 'facade_model.pkl'...
Model loaded successfully.

Created a sample user with interests in 'food' and 'tech'.

Simulating start of session. Initial state for the agent:
[0.48964655 0.82860184 0.36085954 0.8294869  0.5173254  0.
 0.         0.         0.31208152 0.         0.         0.
 0.         0.        ]

==================== Agent's Decision ====================
Recommended Action: Show Ad #9
  - Category: sports
  - Relevance Score: 0.38044689483262445
  - Click Value: $3.52
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