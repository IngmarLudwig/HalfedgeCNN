import argparse
import os
from util import util
import torch


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--arch', type=str, choices={"mconvnet", "meshunet"}, help='Selects network architecture to use.')
        self.parser.add_argument('--batch_size', type=int, help='Input batch size for training.')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Models are saved here.')
        self.parser.add_argument('--dataroot', type=str, required=True, help='Path to meshes (should have subfolders train, test and (if validation used) val).')
        self.parser.add_argument('--dataset_mode', required=True, type=str, choices={"classification", "segmentation"}, help='Choose if segmentation or classification should be performed.')
        self.parser.add_argument('--export_folder', type=str, default='', help='If set, export intermediate collapses to this folder.')
        self.parser.add_argument('--fc_n', type=int, help='Number between fc and nclasses.')
        self.parser.add_argument('--feat_selection', type=int, choices={0, 1, 2}, help='Feature Selection to use. See extract_features_from_he in half_edge_mesh_prepare.py for details.')
        self.parser.add_argument('--gpu_ids', type=str, help='IDs of the GPUs to use, e.g. 0  0,1,2, 0,2. Use -1 to use only CPU.')
        self.parser.add_argument('--init_gain', type=float, help='Scaling factor if normal, xavier or orthogonal initialization for the weights is used.')
        self.parser.add_argument('--init_type', type=str, choices={"normal", "xavier", "kaiming", "orthogonal"}, help='The initialization method to use for the weights of the network.')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples per epoch.')
        self.parser.add_argument('--name', type=str, default='debug', help='Name of the experiment. It decides where to store samples and models.')
        self.parser.add_argument('--nbh_size', type=int, choices={2, 3, 4, 5, 7, 9}, help='Size of the neighborhood.')
        self.parser.add_argument('--ncf', nargs='+', type=int,
                                 help='Defines number of convolution filters and there number of Channels.'
                                      'E.g.: [16, 32, 64] means 3 filters, first with 16 channels and '
                                      'second with 32 channels and third with 64 channels, applied in this order.'
                                      'If arch = "meshunet", this also defines the number of channels during upconvolution in the reverse order.')
        self.parser.add_argument('--number_input_faces', type=int, help='Max number of faces in the training and testing meshes.')
        self.parser.add_argument('--norm', type=str, choices={"batch", "instance", "group"}, help='The type of normalization to use.')
        self.parser.add_argument('--num_groups', type=int, help='Number of groups if group-normalization is used.')
        self.parser.add_argument('--num_threads', type=int, help='Number of threads for loading data.')
        self.parser.add_argument('--pool_res', nargs='+', type=int,
                                 help='Pooling resolutions in the pooling phases.'
                                      ' E.g.: [2280, 1560, 1160]: During first pooling, the number of halfedges is reduced to 2280,'
                                      ' during second pooling to 1560 and during the third Pooling to 1160.'
                                      'If arch = "meshunet", this also defines the resolution during uppooling in the reverse order.'
                                      'Doubled relative to original MeshCNN values due to halfedge use.')
        self.parser.add_argument('--pooling', type=str, choices={"edge_pooling", "half_edge_pooling"},
                                 help='Pooling method to use. edge_pooling is the pooling method used in MeshCNN. See "HalfedgeCNN Paper for details.')
        self.parser.add_argument('--print_labels', action='store_true', help='Print labels and results during testing.')
        self.parser.add_argument('--resblocks', type=int, help='Number of res blocks.')
        self.parser.add_argument('--seed', type=int, help='If specified, uses seed. This leads to deterministic behavior only if --gpu_ids is set to -1 and therefore no GPU is used.')
        self.parser.add_argument('--serial_batches', action='store_true', help='If true, takes meshes in order, otherwise takes them randomly.') # Already broken in original MeshCNN for Classification
        self.parser.add_argument('--segmentation_base', type=str, choices={"edge_based", "halfedge_based", "face_based"},
                                 help='Define whether the provided labels are edge based (eseg and heseg files in in seg and sseg folder) '
                                      'or halfedge based (heseg and sheseg files in hseg and hsseg folder). ')


        self.initialized = True


    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt, unknown = self.parser.parse_known_args()
        self.opt.is_train = self.is_train   # Set to True if training functionality like autograd is neccessary.

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        self.args = vars(self.opt)

        if self.opt.seed is not None:
            import numpy as np
            import random
            torch.manual_seed(self.opt.seed)
            np.random.seed(self.opt.seed)
            random.seed(self.opt.seed)

        if self.opt.export_folder:
            self.opt.export_folder = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.export_folder)
            util.mkdir(self.opt.export_folder)

        self.opt.number_input_edges      = self.opt.number_input_faces * 2
        self.opt.number_input_half_edges = self.opt.number_input_faces * 3

        return self.opt


    def formatted_str(self):
        msg = '\n------------- Options: -------------\n'

        for k, v in sorted(self.args.items()):
            msg += '%s: %s \n' % (str(k), str(v))
        msg += '-------------- End ----------------\n'
        return msg