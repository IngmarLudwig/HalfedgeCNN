import os
import os.path
import shutil
from os import path


def delete_dataset_cache(dataset_path):
    paths = os.walk(dataset_path)
    for p in paths:
        if "cache" in p[0]:
            try:
                shutil.rmtree(p[0])
            except OSError as e:
                print("Error: %s : %s" % (p[0], e.strerror))


def delete_half_edge_hard_segmentation_folder(dataset_path):
    hseg_path = path.join(dataset_path, "hseg")
    if path.exists(hseg_path):
        shutil.rmtree(hseg_path)


def delete_mean_std_cache(mean_std_cache_path):
    try:
        if path.exists(mean_std_cache_path):
            absolute_mean_std_path = os.path.abspath(mean_std_cache_path)
            os.remove(absolute_mean_std_path)
            #print("Deleted", mean_std_cache_path)
    except OSError as e:
        print("Error: %s : %s" % (mean_std_cache_path, e.strerror))


def delete_checkpoints(checkpoints_path):
    try:
        if path.exists(checkpoints_path):
            absolute_checkpoint_path = os.path.abspath(checkpoints_path)
            shutil.rmtree(absolute_checkpoint_path)
            #print("Deleted", absolute_checkpoint_path)
    except OSError as e:
        print("Error: %s : %s" % (checkpoints_path, e.strerror))


def delete_cache(dataset_name):
    print("Deleting Cache and checkpoints.")
    delete_dataset_cache(path.join("datasets", dataset_name))
    delete_half_edge_hard_segmentation_folder(path.join("datasets", dataset_name))
    delete_mean_std_cache(path.join("datasets", dataset_name, "mean_std_cache.p"))
    delete_checkpoints(path.join("checkpoints", dataset_name))


if __name__ == '__main__':
    delete_cache("shrec_16")
    delete_cache("debug")
    delete_cache("human_seg")
    delete_cache("coseg_chairs")
    delete_cache("coseg_vases")
    delete_cache("coseg_aliens")