from train_util.train_util import get_target_number_of_epochs_from_command_line
from train_util.evaluation import show_runs_chart


if __name__ == '__main__':
    target_epochs = get_target_number_of_epochs_from_command_line()
    show_runs_chart(target_epochs)