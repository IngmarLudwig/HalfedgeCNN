from train_util.evaluation import evaluate_runs
from train_util.train_util import get_target_number_of_epochs_from_command_line

if __name__ == '__main__':
    target_epochs = get_target_number_of_epochs_from_command_line()
    evaluate_runs(target_epochs)