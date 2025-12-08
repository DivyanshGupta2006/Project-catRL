import torch
from tqdm import tqdm

from src.strategy.model import Model
from src.strategy.agent import Agent
from src.strategy.environment import Environment
from src.strategy.buffer import Buffer
from src.utils import get_config, get_absolute_path, read_file, convert, check_dir

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
NUM_ROLLOUTS = hp['num_rollouts']
NUM_EPOCHS = hp['num_epochs']
LEARNING_RATE = hp['learning_rate']
BOUND_REWARD_FACTOR = hp['bound_reward_factor']

SYMBOLS = config['data']['symbols']
CAPITAL = config['strategy']['capital']
MODEL_PATH = get_absolute_path.absolute(config['paths']['model_directory'] + "model.pth")
RESULTS_PATH = get_absolute_path.absolute(config['paths']['report_directory'])  / 'equity_charts/'

check_dir.check(get_absolute_path.absolute(config['paths']['model_directory']))
check_dir.check(get_absolute_path.absolute(config['paths']['report_directory'] + 'equity_charts/'))

def train():
    print('Starting Training...')

    print(f'Number of assets: {NUM_ASSETS}')
    print(f'Input dim: {INPUT_DIM}')
    print(f'Action dim: {ACTION_DIM}')
    print(f'Num LSTM layers: {NUM_LSTM_LAYERS}')
    print(f'Hidden state dim: {HIDDEN_STATE_DIM}')
    print(f'Actor hidden dim: {ACTOR_HIDDEN_DIM}')
    print(f'Critic hidden dim: {CRITIC_HIDDEN_DIM}')
    print(f'Gamma: {GAMMA}')
    print(f'Gae lambda: {GAE_LAMBDA}')
    print(f'Clip epsilon: {CLIP_EPSILON}')
    print(f'Value loss coef: {VALUE_LOSS_COEF}')
    print(f'Entropy loss coef: {ENTROPY_LOSS_COEF}')
    print(f'Sequence length: {SEQUENCE_LENGTH}')
    print(f'Mini batch size: {MINI_BATCH_SIZE}')
    print(f'Rollout size: {ROLLOUT_SIZE}')
    print(f'Num rollouts: {NUM_ROLLOUTS}')
    print(f'Num epochs: {NUM_EPOCHS}')
    print(f'Learning rate: {LEARNING_RATE}')
    print(f'Bound reward: {BOUND_REWARD_FACTOR}')
    print(f'Symbols: {SYMBOLS}')
    print(f'Capital: {CAPITAL}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print('Loading data...')
    train_data_norm = read_file.read_merged_training_data(True)
    train_data_unnorm = read_file.read_merged_training_data(False)

    # drop the redundant columns
    train_data_norm.columns = [convert.convert_to_tuple(col) for col in train_data_norm.columns]
    train_data_norm.drop(columns=[col for col in train_data_norm.columns if
                                  col[0] == 'open' or col[0] == 'close' or col[0] == 'high' or col[0] == 'low'],
                         inplace=True)

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
                      bound_reward_factor=BOUND_REWARD_FACTOR,
                      seq_len=SEQUENCE_LENGTH,
                      capital=CAPITAL,
                      symbols=SYMBOLS,
                      results_path=RESULTS_PATH)

    buffer = Buffer(total_rollout_size=ROLLOUT_SIZE,
                    device=device)

    temp_buffer = Buffer(total_rollout_size=1,
                         device=device)

    if NUM_ROLLOUTS == 'd':
        num_rollouts = (int)(len(train_data_norm) / ROLLOUT_SIZE)
    else:
        num_rollouts = NUM_ROLLOUTS
    state = env.reset(train_data_norm)
    state_metric = read_file.read_state()
    portfolio = read_file.read_portfolio()

    for rollout in tqdm(range(num_rollouts), desc='Training'):
        buffer.clear()
        buffer.store_state(state)

        print('\nGathering Experiences...')
        # rollout
        for i in range(ROLLOUT_SIZE):
            buffer = agent.get_action_and_value(buffer)
            state, reward, done, state_metric, portfolio = env.step(buffer.actions[-1], train_data_norm, state_metric, portfolio)
            if done == 1:
                state = env.reset(train_data_norm, True)
                state_metric = read_file.read_state()
                portfolio = read_file.read_portfolio()
                # break
            buffer.store_state(state)
            buffer.store_rewards(reward)
            buffer.store_dones(done)

        next_value = 0
        if done != 1:
            if state is not None:
                temp_buffer.store_state(state)
                temp_buffer = agent.get_action_and_value(temp_buffer)
                next_value = temp_buffer.values[-1]
            temp_buffer.clear()

        print('Learning...')
        loss = agent.update(buffer, next_value)

        agent.save()

        print(f"\nRollout {rollout + 1}/{num_rollouts} | Loss: {loss:.2f}")
        # print(f"Trajectory length: {len(buffer.rewards)}")

    print('Training complete.')