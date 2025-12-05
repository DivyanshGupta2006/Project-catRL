import torch
from tqdm import tqdm

from src.strategy.model import Model
from src.strategy.agent import Agent
from src.strategy.environment import Environment
from src.strategy.buffer import Buffer
from src.utils import get_config, get_absolute_path, read_file

config = get_config.read_yaml()

# load hyperparameters
hp = config['hyperparameters']

NUM_ASSETS = hp['num_assets']
INPUT_DIM = hp['input_dim']
ACTION_DIM = hp['action_dim']
NUM_LSTM_LAYERS = hp['num_lstm_layers']
HIDDEN_STATE_DIM = hp['hidden_state_dim']
ACTOR_HIDDEN_DIM = hp['actor_hidden_dim']
CRITIC_HIDDEN_DIM = hp['critic_hidden_dim']

GAMMA = hp['gamma']
GAE_LAMBDA = hp['gae_lambda']
CLIP_EPSILON = hp['clip_epsilon']
VALUE_LOSS_COEF = hp['value_loss_coef']
ENTROPY_LOSS_COEF = hp['entropy_loss_coef']

SEQUENCE_LENGTH = hp['seq_len']
MINI_BATCH_SIZE = hp['mini_batch_size']
ROLLOUT_SIZE = hp['rollout_size']
NUM_EPOCHS = hp['num_epochs']
LEARNING_RATE = hp['learning_rate']

SYMBOLS = config['data']['symbols']
CAPITAL = config['strategy']['capital']
MODEL_PATH = get_absolute_path.absolute(config['paths']['model_directory'] + "model.pth")

def train():
    print('Starting Training...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print('Loading data...')
    train_data_norm = read_file.read_merged_training_data(True)
    train_data_unnorm = read_file.read_merged_training_data(False)

    print('Instantiating entities...')

    model = Model(n_assets=NUM_ASSETS,
                  input_dim=INPUT_DIM,
                  lstm_hidden_dim=HIDDEN_STATE_DIM,
                  n_lstm_layers=NUM_LSTM_LAYERS,
                  actor_hidden_dim=ACTOR_HIDDEN_DIM,
                  critic_hidden_dim=CRITIC_HIDDEN_DIM)

    agent = Agent(model=model,
                  gamma=GAMMA,
                  gae_lambda=GAE_LAMBDA,
                  clip_epsilon=CLIP_EPSILON,
                  num_epochs=NUM_EPOCHS,
                  mini_batch_size=MINI_BATCH_SIZE,
                  learning_rate=LEARNING_RATE,
                  value_loss_coef=VALUE_LOSS_COEF,
                  entropy_loss_coef=ENTROPY_LOSS_COEF,
                  model_path=MODEL_PATH,
                  device=device)

    env = Environment(data=train_data_unnorm,
                      seq_len=SEQUENCE_LENGTH,
                      capital=CAPITAL,
                      symbols=SYMBOLS)

    buffer = Buffer(total_rollout_size=ROLLOUT_SIZE,
                    device=device)

    temp_buffer = Buffer(total_rollout_size=1,
                         device=device)

    num_rollouts = (int)(len(train_data_norm) / ROLLOUT_SIZE)
    num_rollouts = min(20, num_rollouts)
    state = env.reset(train_data_norm)

    for rollout in tqdm(range(num_rollouts), desc='Training Rollouts'):
        buffer.clear()
        buffer.store_state(state)

        # rollout
        for i in range(ROLLOUT_SIZE):
            buffer = agent.get_action_and_value(buffer)
            state, reward, done = env.step(buffer.actions[-1], train_data_norm)
            buffer.store_state(state)
            buffer.store_rewards(reward)
            buffer.store_dones(done)
            if done == 1:
                state = env.reset(train_data_norm)
                # break

        next_value = 0
        if done != 1:
            if state is not None:
                temp_buffer.store_state(state)
                temp_buffer = agent.get_action_and_value(temp_buffer)
                next_value = temp_buffer.values[-1]
            temp_buffer.clear()

        loss = agent.update(buffer, next_value)
        print(f"\nRollout {rollout}/{num_rollouts} | Loss: {loss:.2f}")
        # print(f"Trajectory length: {len(buffer.rewards)}")

    agent.save()
    print('Training complete.')