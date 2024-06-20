import torch

def get_dataset_settings_dict(dataset_name):
    dataset_settings = _read_settings(dataset_name + '_settings.txt')
    return dataset_settings

def get_general_settings_dict():
    general_settings_dict = _read_settings('general_settings.txt')
    return general_settings_dict

def get_clas_or_seg_settings_dict(dataset_settings_dict):
    general_settings_dict = None
    if dataset_settings_dict["--dataset_mode"] == "segmentation":
        general_settings_dict = _read_settings('segmentation_general_settings.txt')
    elif dataset_settings_dict["--dataset_mode"] == "classification":
        general_settings_dict = _read_settings('classification_general_settings.txt')
    else:
        raise Exception("Unknown dataset mode (--dataset_mode):", dataset_settings_dict["--dataset_mode"])
    return general_settings_dict

def get_training_settings_dict(dataset_settings_dict):
    training_settings_dict = None
    if dataset_settings_dict["--dataset_mode"] == "segmentation":
        training_settings_dict = _read_settings('segmentation_training_settings.txt')
    elif dataset_settings_dict["--dataset_mode"] == "classification":
        training_settings_dict = _read_settings('classification_training_settings.txt')
    else:
        raise Exception("Unknown dataset mode (--dataset_mode):", dataset_settings_dict["--dataset_mode"])
    return training_settings_dict


def get_test_settings_dict():
    test_settings = _read_settings('test_settings.txt')
    return test_settings

def create_settings_string(settings_dict):
    settings_keys = list(settings_dict)
    settings_keys.sort()

    command = ''
    for key in settings_keys:
        command += key + ' ' + settings_dict[key] + ' '

    return command


def print_cuda_information():
    cuda_available = torch.cuda.is_available()
    print("Cuda available:", cuda_available)
    if cuda_available:
        print("Number of available cuda devices:", torch.cuda.device_count())


def get_cuda_settings_string():
    cuda_str = '--gpu_ids '
    if not torch.cuda.is_available():
        cuda_str += '-1 '
    else:
        cuda_count = torch.cuda.device_count()
        for i in range(cuda_count):
            cuda_str += str(i) + ' '
    return cuda_str


def _read_settings(file):
    settings_dict = {}
    with open('scripts/settings/' + file) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            if len(line.split(' ', 1)) == 1:
                key = line
                value = ''
            elif len(line.split(' ', 1)) == 2:
                key, value = line.split(' ', 1)
            else:
                raise Exception("Error in settings file:", file, "Line:", line)
            settings_dict[key] = value.strip()

    return settings_dict



