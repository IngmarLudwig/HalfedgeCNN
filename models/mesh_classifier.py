import os
import torch
from os.path import join

from . import networks
from models.layers.accuracy_calculators import EdgeBasedAccuracyCalculator, HalfEdgeBasedAccuracyCalculator, FaceBasedAccuracyCalculator
from .layers.output_data_interface_layer import EdgeBasedDataOutputInterfaceLayer, HalfEdgeBasedDataOutputInterfaceLayer, FaceBasedDataOutputInterfaceLayer


class ClassifierModel:
    """ Class for training Model weights.
        :args opt: structure containing configuration params
        e.g.,
        --dataset_mode -> classification / segmentation)
        --arch -> network type
    """
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.half_edge_features = None
        self.labels = None
        self.meshes = None
        self.soft_label = None
        self.loss = None

        self.nclasses = opt.nclasses

        self.accuracy_calculator = None
        self.output_interface_layer = None
        if opt.dataset_mode == 'segmentation':
            if opt.segmentation_base == 'edge_based':
                self.accuracy_calculator    = EdgeBasedAccuracyCalculator()
                self.output_interface_layer = EdgeBasedDataOutputInterfaceLayer()
            elif opt.segmentation_base == 'halfedge_based':
                self.accuracy_calculator    = HalfEdgeBasedAccuracyCalculator()
                self.output_interface_layer = HalfEdgeBasedDataOutputInterfaceLayer()
            elif opt.segmentation_base == 'face_based':
                self.accuracy_calculator    = FaceBasedAccuracyCalculator()
                self.output_interface_layer = FaceBasedDataOutputInterfaceLayer()
            else:
                raise NotImplementedError('Segmentation base {} not implemented'.format(opt.segmentation_base))

        # Load/define networks.
        self.net = networks.define_classifier(opt.input_nc, opt.ncf, opt.number_input_half_edges, opt.nclasses, opt,
                                              self.gpu_ids, opt.arch, opt.init_type, opt.init_gain)
        self.net.train(self.is_train)
        self.criterion = networks.define_loss(opt).to(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, opt)

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def set_input(self, data):
        input_half_edge_features = torch.from_numpy(data['half_edge_features']).float()
        labels = torch.from_numpy(data['label']).long()
        # set inputs
        self.half_edge_features = input_half_edge_features.to(self.device).requires_grad_(self.is_train)
        self.labels = labels.to(self.device) # labels are halfedge based
        self.meshes = data['mesh']
        if self.opt.dataset_mode == 'segmentation' and not self.is_train:
            self.soft_label = torch.from_numpy(data['soft_label']) # soft labels have segmentation base selected in segmentatio_base

    def forward(self):
        out = self.net(self.half_edge_features, self.meshes)
        if self.opt.dataset_mode == 'segmentation':
            out = self.output_interface_layer.transform_predictions_to_match_segmentation_base(out, self.meshes)
        return out

    def backward(self, out):
        self.loss = self.criterion(out, self.labels)
        self.loss.backward()


    def optimize_parameters(self):
        self.optimizer.zero_grad()

        # forward without conversion to segmentation base
        out = self.net(self.half_edge_features, self.meshes)

        self.backward(out)
        self.optimizer.step()


##################

    def load_network(self, which_epoch):
        """ load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module

        # PyTorch newer than 0.4 (e.g., built from GitHub source), you can remove str() on self.device.
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)


    def save_network(self, name):
        """ Save model to disk."""
        save_filename = '%s_net.pth' % (name)
        save_path = join(self.save_dir, save_filename)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        """ Update learning rate (called once every epoch)."""
        self.scheduler.step()

    def test(self):
        """ Tests model
            returns: number correct and total number
        """
        with torch.no_grad():
            out = self.forward()
            predictions = out.data.max(1)[1]

            if self.opt.dataset_mode == 'segmentation' and self.opt.export_folder is not None:
                self.export_segmentation(predictions.cpu())

            labels = self.labels
            correct = self.get_accuracy(predictions=predictions, labels=labels)
        return correct, len(labels)

    def get_accuracy(self, predictions, labels):
        """ Computes accuracy for classification/segmentation. """
        correct = None
        if self.opt.dataset_mode == 'classification':
            correct = predictions.eq(labels).sum()
        elif self.opt.dataset_mode == 'segmentation':
            correct = self.accuracy_calculator.calculate_segmentation_accuracy(predictions, self.soft_label, self.meshes)
        return correct

    def export_segmentation(self, pred_seg):
        for i, mesh in enumerate(self.meshes):
            mesh.export_segmentation_of_mesh(pred_seg[i, :])

    def get_description(self):
        num_params = 0
        for param in self.net.parameters():
            num_params += param.numel()
        return 'Total number of parameters in network : %.3f M' % (num_params / 1e6)
