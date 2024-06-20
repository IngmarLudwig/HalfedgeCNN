import os
from train_util.train_util import get_dataset_name_from_command_line
from settings.settings import print_cuda_information, get_clas_or_seg_settings_dict, create_settings_string, get_cuda_settings_string, \
    get_dataset_settings_dict, get_test_settings_dict, get_general_settings_dict


def test(dataset_name, best_model=False, export_folder=None):
    print_cuda_information()

    dataset_dict           = get_dataset_settings_dict(dataset_name)
    clas_or_seg_settings_dict  = get_clas_or_seg_settings_dict(dataset_dict)
    test_settings_dict     = get_test_settings_dict()
    general_settings_dict  = get_general_settings_dict()

    # dataset settings override test settings, which override general settings
    combined_settings = {**general_settings_dict, **clas_or_seg_settings_dict, **test_settings_dict, **dataset_dict}


    # -W ignore used because of VisibleDeprecationWarning that otherwise clutter output
    command = 'python -W ignore test.py '
    command += create_settings_string(combined_settings)
    command += get_cuda_settings_string()
    if best_model:
        command += '--which_epoch best '
    else:
        command += '--which_epoch latest '
    if export_folder is not None:
        command += '--export_folder '+export_folder+ ' '
    print("Command:", command)
    os.system(command)


if __name__ == '__main__':
    """ The script expects the name of the dataset as an argument. A text file conaining the settings for the dataset must exist."""
    dataset_name = get_dataset_name_from_command_line()
    test(dataset_name=dataset_name, best_model=False)