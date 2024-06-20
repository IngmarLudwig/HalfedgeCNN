import os
import numpy as np

from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad
from models.layers import half_edge_mesh
from models.layers.input_data_interface_layer import EdgeBasedDataInputInterfaceLayer, HalfEdgeBasedDataInputInterfaceLayer, FaceBasedDataInputInterfaceLayer


class SegmentationData(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.obj_file_paths = self.make_dataset(os.path.join(opt.dataroot, opt.phase))

        self.seg_data_interface_layer = None
        if opt.segmentation_base == 'edge_based':
            self.seg_data_interface_layer = EdgeBasedDataInputInterfaceLayer(opt.number_input_half_edges)
        elif opt.segmentation_base == 'halfedge_based':
            self.seg_data_interface_layer = HalfEdgeBasedDataInputInterfaceLayer(opt.number_input_half_edges)
        elif opt.segmentation_base == 'face_based':
            self.seg_data_interface_layer = FaceBasedDataInputInterfaceLayer(opt.number_input_half_edges)
        else:
            raise NotImplementedError('Segmentation base {} is not implemented'.format(opt.segmentation_base))

        self.size = len(self.obj_file_paths)

        classes, offset = self.get_n_segs(os.path.join(opt.dataroot, 'classes.txt'))
        self.offset = offset

        self.get_mean_std()

        # modify for network later.
        nclasses = len(classes)
        opt.nclasses = nclasses
        opt.input_nc = self.ninput_channels

    def __getitem__(self, index):
        obj_file_path = self.obj_file_paths[index]
        mesh = half_edge_mesh.HalfEdgeMesh(file=obj_file_path, opt=self.opt, hold_history=True, export_folder=self.opt.export_folder)

        meta = {}
        meta['mesh'] = mesh

        half_edge_features = pad(mesh.half_edge_features, self.opt.number_input_half_edges)
        meta['half_edge_features'] = (half_edge_features - self.mean) / self.std
        meta['label']= self.seg_data_interface_layer.read_hard_segmentation_for_training(obj_file_path, padding=True, offset=self.offset)
        meta['soft_label'] = self.seg_data_interface_layer.read_soft_segmentation_for_testing(obj_file_path, padding=True, perform_ceil=True)

        return meta

    def __len__(self):
        return self.size

    def get_n_segs(self, classes_file):
        if not os.path.isfile(classes_file):
            all_segs = np.array([], dtype='float64')
            for path in self.obj_file_paths:
                all_segs = np.concatenate((all_segs, self.seg_data_interface_layer.read_hard_segmentation_for_training(path, padding=False)))
            segnames = np.unique(all_segs)
            np.savetxt(classes_file, segnames, fmt='%d')
        classes = np.loadtxt(classes_file)
        offset = classes[0]
        classes = classes - offset
        return classes, offset


    @staticmethod
    def make_dataset(path):
        meshes = []
        assert os.path.isdir(path), '%s is not a valid directory' % path

        for root, _, fnames in sorted(os.walk(path)):
            for fname in fnames:
                if is_mesh_file(fname):
                    path = os.path.join(root, fname)
                    meshes.append(path)

        return meshes


