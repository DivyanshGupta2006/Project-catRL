import torch

from src.backtester import print_backtesting_results
from src.strategy.environment import Environment
from src.strategy.model import Model
from src.strategy.agent import Agent
from src.strategy.buffer import Buffer
from src.utils import get_config, read_file

config = get_config.read_yaml()

def train():
    print("Starting Training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    MODEL_PATH = config['paths']['model_directory']

    hp = config['hyperparameters']
    NUM_ASSETS = hp['num_assets']
    INPUT_DIM = hp['input_dim']
    NUM_LSTM_LAYERS = hp['num_lstm_layers']
    HIDDEN_STATE_DIM = hp['hidden_state_dim']
    ACTOR_HIDDEN_DIM = hp['actor_hidden_dim']
    CRITIC_HIDDEN_DIM = hp['critic_hidden_dim']
    GAMMA = hp['gamma']
    GAE_LAMBDA = hp['gae_lambda']
    CLIP_EPSILON = hp['clip_epsilon']
    VALUE_LOSS_COEF = hp['value_loss_coef']
    ENTROPY_LOSS_COEF = hp['entropy_loss_coef']
    LEARNING_RATE = hp['learning_rate']
    NUM_EPOCHS = hp['num_epochs']
    ROLLOUT_SIZE = hp['rollout_size']
    BATCH_SIZE = hp['mini_batch_size']
    SEQUENCE_LENGTH = hp['seq_len']
    train_data = read_file.read_merged_training_data()

    model = Model(INPUT_DIM,
                  HIDDEN_STATE_DIM,
                  NUM_ASSETS,
                  NUM_LSTM_LAYERS,
                  ACTOR_HIDDEN_DIM,
                  CRITIC_HIDDEN_DIM,)

    agent = Agent(model,
                  GAMMA,
                  GAE_LAMBDA,
                  CLIP_EPSILON,
                  VALUE_LOSS_COEF,
                  ENTROPY_LOSS_COEF,
                  LEARNING_RATE,
                  device,
                  MODEL_PATH)

    env = Environment(train_data,
                      SEQUENCE_LENGTH)

    buffer = Buffer()

    for rollout in range((len(train_data)) / ROLLOUT_SIZE):
        for mini_batch in range(ROLLOUT_SIZE / BATCH_SIZE):
            for sequence in range(BATCH_SIZE):
                for epoch in range(NUM_EPOCHS):
                    buffer = agent.get_action_and_value(buffer)
                    env.step(buffer)
                agent.update(buffer)
                buffer.clear()

    agent.save()
    print_backtesting_results.print_results()