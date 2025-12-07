from src.utils import get_config, check_dir

config = get_config.read_yaml()

def update():
    paths = config['paths']
    for path_name, path in paths.items():
        if path_name != 'env_directory' and path_name != 'state_directory' and path_name != 'portfolio_directory':
            check_dir.check(path)