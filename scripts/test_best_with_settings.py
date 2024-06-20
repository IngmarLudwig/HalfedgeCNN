from train_util.train_util import get_dataset_name_from_command_line
from test_with_settings import test

if __name__ == '__main__':
    """ The script expects the name of the dataset as an argument. A text file conaining the settings for the dataset must exist."""
    dataset_name = get_dataset_name_from_command_line()
    test( dataset_name=dataset_name, best_model=True)