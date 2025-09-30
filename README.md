# Reinforcement Learning - QLearning: Fronzen Lake ⛄

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/envs.gif" alt="Environments" width="600"/>

## Overview

This project provides a comprehensive, step-by-step implementation of the Q-Learning algorithm to solve the `FrozenLake-v1` environment from the [Gymnasium library](https://gymnasium.farama.org/environments/toy_text/frozen_lake/). The goal is to train an agent to navigate a frozen lake from a starting position (S) to a goal (G) without falling into holes (H).

This repository is designed to be highly educational, with a detailed Jupyter Notebook that explains each concept, from the Q-table to the exploration-exploitation trade-off.

## Key Concepts Covered

- **Reinforcement Learning (RL)**: The fundamentals of agents, environments, states, actions, and rewards.
- **Q-Learning**: A model-free, off-policy RL algorithm.
- **Q-Table**: The data structure used to store and update state-action values.
- **Epsilon-Greedy Policy**: A strategy for balancing exploration and exploitation.
- **Gymnasium API**: Interacting with RL environments.
- **Model Evaluation and Visualization**: Assessing the agent's performance and creating video replays.
- **Hugging Face Hub**: Pushing the trained model to the Hub for sharing and collaboration.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/<your-username>/q-learning-frozenlake.git
    cd q-learning-frozenlake
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For use in environments like Google Colab that require a virtual display for rendering, additional system packages might be needed. See the notebook for details.*

## How to Run

Simply open and run the `frozen_lake_q_learning.ipynb` notebook in a Jupyter environment. The notebook will guide you through the entire process of training, evaluating, and even publishing your agent.

## Results

After training, the agent successfully learns the optimal policy to navigate the lake. The final model achieves a high mean reward, and a video replay of its performance is generated and can be viewed on the [Hugging Face Hub](https://huggingface.co/models?other=q-learning).
