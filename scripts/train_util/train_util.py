import sys


def get_target_number_of_epochs_from_command_line():
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        target_epochs = int(sys.argv[1])
    else:
        print("Please specify target for number of epochs.")
        exit(-1)
    print("Target for number of epochs:", target_epochs)
    return target_epochs

def get_dataset_name_from_command_line():
    if len(sys.argv) > 1:
        # The path to the folder containing the obj files and seg files.
        dataset_name = sys.argv[1]
    else:
        print("Please specify the name of the dataset.")
        exit(-1)
    return dataset_name