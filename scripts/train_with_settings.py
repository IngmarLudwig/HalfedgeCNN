import os
from train_util.train_util import get_dataset_name_from_command_line
from train_util.delete_cache import delete_cache
from settings.settings import print_cuda_information, get_clas_or_seg_settings_dict, create_settings_string, get_cuda_settings_string, \
    get_dataset_settings_dict, get_training_settings_dict, get_general_settings_dict

if __name__ == '__main__':
    """ The script expects the name of the dataset as an argument. A text file conaining the settings for the dataset must exist."""
    print_cuda_information()

    dataset_name = get_dataset_name_from_command_line()

    dataset_dict           = get_dataset_settings_dict(dataset_name)
    clas_or_seg_settings_dict  = get_clas_or_seg_settings_dict(dataset_dict)
    training_settings_dict = get_training_settings_dict(dataset_dict)
    general_settings_dict  = get_general_settings_dict()

    # dataset settings override training settings, which override general settings
    combined_settings = {**general_settings_dict, **clas_or_seg_settings_dict, **training_settings_dict, **dataset_dict}

    # -W ignore used because of VisibleDeprecationWarning that otherwise clutter output
    command = 'python -W ignore train.py '
    command += create_settings_string(combined_settings)
    command += get_cuda_settings_string()
    print("Command:", command)

    if combined_settings["--number_of_runs"] is None and combined_settings["--number_of_runs"].isnumeric():
        raise Exception("No number of runs (--number_of_runs) specified in settings file.")
    number_of_runs = int(combined_settings["--number_of_runs"])

    for i in range(number_of_runs):
        delete_cache(dataset_name)
        os.system(command)