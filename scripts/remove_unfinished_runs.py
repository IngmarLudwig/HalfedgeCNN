from train_util.train_util import get_target_number_of_epochs_from_command_line
from train_util.evaluation import remove_unfinished_runs


if __name__ == '__main__':
    target_epochs = get_target_number_of_epochs_from_command_line()
    remove_unfinished_runs(target_epochs)