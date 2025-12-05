# import numpy as np
# import torch
#
# from src.backtester import print_backtesting_results
# from src.strategy.environment import Environment
# from src.strategy.model import Model
# from src.strategy.agent import Agent
# from src.strategy.buffer import Buffer
# from src.utils import get_config, read_file
# from src.position_sizing import fiducia_calculator
#
# config = get_config.read_yaml()
#
# def train():
#     print("Starting Training...")
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     MODEL_PATH = config['paths']['model_directory']
#
#     hp = config['hyperparameters']
#     NUM_ASSETS = hp['num_assets']
#     INPUT_DIM = hp['input_dim']
#     NUM_LSTM_LAYERS = hp['num_lstm_layers']
#     HIDDEN_STATE_DIM = hp['hidden_state_dim']
#     ACTOR_HIDDEN_DIM = hp['actor_hidden_dim']
#     CRITIC_HIDDEN_DIM = hp['critic_hidden_dim']
#     GAMMA = hp['gamma']
#     GAE_LAMBDA = hp['gae_lambda']
#     CLIP_EPSILON = hp['clip_epsilon']
#     VALUE_LOSS_COEF = hp['value_loss_coef']
#     ENTROPY_LOSS_COEF = hp['entropy_loss_coef']
#     LEARNING_RATE = hp['learning_rate']
#     NUM_EPOCHS = hp['num_epochs']
#     ROLLOUT_SIZE = hp['rollout_size']
#     BATCH_SIZE = hp['mini_batch_size']
#     SEQUENCE_LENGTH = hp['seq_len']
#     train_data = read_file.read_merged_training_data()
#
#     model = Model(INPUT_DIM,
#                   HIDDEN_STATE_DIM,
#                   NUM_ASSETS,
#                   NUM_LSTM_LAYERS,
#                   ACTOR_HIDDEN_DIM,
#                   CRITIC_HIDDEN_DIM,)
#
#     agent = Agent(model,
#                   GAMMA,
#                   GAE_LAMBDA,
#                   CLIP_EPSILON,
#                   VALUE_LOSS_COEF,
#                   ENTROPY_LOSS_COEF,
#                   LEARNING_RATE,
#                   device,
#                   MODEL_PATH)
#
#     env = Environment(train_data,
#                       SEQUENCE_LENGTH)
#
#     buffer = Buffer()
#     buffer.states.append(train_data[0:SEQUENCE_LENGTH])
#
#     env.reset()
#
#     for rollout in range((len(train_data)) // ROLLOUT_SIZE):
#         for step in range(ROLLOUT_SIZE):
#             buffer = agent.get_action_and_value(buffer)
#             fiduciae = fiducia_calculator.calculate(buffer.actions[-1])
#             buffer, timestep = env.step(fiduciae, buffer)
#             buffer = agent.compute_gae(buffer)      # next value, next done ?
#
#         indices = np.arange(ROLLOUT_SIZE)  # Create indices 0...ROLLOUT_SIZE-1
#
#         for epoch in range(NUM_EPOCHS):
#             np.random.shuffle(indices)  # Shuffle indices
#
#             for start in range(0, ROLLOUT_SIZE, BATCH_SIZE):
#                 end = start + BATCH_SIZE
#                 mb_indices = indices[start:end]  # Get mini-batch indices
#
#                 # Get the mini-batch data using shuffled indices
#                 agent.update(
#                     buffer, start
#                 )
#
#
#     agent.save()
#     print_backtesting_results.print_results()

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.strategy.environment import FastEnvironment
from src.strategy.model import Model
from src.strategy.agent import Agent
from src.strategy.buffer import Buffer
from src.utils import get_config, read_file, get_absolute_path

config = get_config.read_yaml()


def train():
    print("--- Starting Refined Algorithm II Training (M=T) ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data (Merged Training Data)
    print("Loading Data...")
    train_data = read_file.read_merged_training_data()
    # Normalize or clean data here if needed, or assume it's done in ETL

    # 2. Initialize Components
    hp = config['hyperparameters']

    model = Model(
        input_dim=hp['input_dim'],
        lstm_hidden_dim=hp['hidden_state_dim'],
        n_assets=hp['num_assets'],
        n_lstm_layers=hp['num_lstm_layers'],
        actor_hidden_dim=hp['actor_hidden_dim'],
        critic_hidden_dim=hp['critic_hidden_dim']
    )

    agent = Agent(model, config, device)
    buffer = Buffer(device)

    # Using the Fast In-Memory Environment
    env = FastEnvironment(train_data, config)

    # 3. Training Loop Variables
    rollout_size = hp['rollout_size']  # 1024
    total_steps = len(train_data) - hp['seq_len'] - 1
    num_rollouts = total_steps // rollout_size

    state = env.reset()

    # 4. Main Loop
    for rollout in tqdm(range(num_rollouts), desc="Training Progress"):

        # --- A. ROLLOUT PHASE (Data Collection) ---
        for t in range(rollout_size):
            # 1. Get Action from Agent
            action, log_prob, value = agent.get_action(state)

            # 2. Execute in Env
            next_state, reward, done, _ = env.step(action)

            # 3. Store in Buffer
            buffer.store(state, action, log_prob, value, reward, done)

            state = next_state

            if done:
                state = env.reset()
                break

        # --- B. GAE PHASE (Advantage Calculation) ---
        # We need value of the *next* state to calculate advantage for the last step
        last_value = agent.get_value(state)
        buffer.finish_path(last_value, hp['gamma'], hp['gae_lambda'])

        # --- C. UPDATE PHASE (Learning) ---
        # Update on the full 1024 steps at once (M=T)
        loss = agent.update(buffer)

        # --- D. CLEANUP ---
        buffer.clear()

        if rollout % 10 == 0:
            print(f" Rollout {rollout}/{num_rollouts} | Loss: {loss:.2f}")

    # 5. Save Model
    agent.save(get_absolute_path.absolute(config['paths']['model_directory']) / 'ppo_agent.pth')
    print("Training Complete. Model Saved.")