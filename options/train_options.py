from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--beta1', type=float, help='Momentum term of adam.')
        self.parser.add_argument('--continue_train', action='store_true', help='If true,  loads the latest model, otherwise creates a new model to train.')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='The starting epoch count. Used if continue_train is True.')
        self.parser.add_argument('--flip_edges', type=float, default=0, help='Percent of edges to randomly flip.')
        self.parser.add_argument('--lr', type=float, help='Initial learning rate for adam.')
        self.parser.add_argument('--lr_decay_iters', type=int, help='Multiply by a gamma every lr_decay_iters iterations if lambda lr_policy is used.')
        self.parser.add_argument('--lr_policy', type=str, choices={"lambda", "step", "plateau"}, help='Learning rate policy.')
        self.parser.add_argument('--niter', type=int, help='Number of iter at starting learning rate.')
        self.parser.add_argument('--niter_decay', type=int, help='Number of iter to linearly decay learning rate to zero.')
        self.parser.add_argument('--no_vis', action='store_true', help='Set to True if you do not want to use tensorboard')
        self.parser.add_argument('--number_augmentations', type=int, help='Number of augmented versions of meshes. 1 means no augmentation.')
        self.parser.add_argument('--phase', type=str, choices={"train", "val", "test"}, default='train', help='Name of the subfolder in datasets to use the meshes from.')
        self.parser.add_argument('--run_test_freq', type=int, default=1, help='Frequency of testing and logging/printing training results.')
        self.parser.add_argument('--scale_verts', action='store_true', help='Non-uniformly scale the mesh e.g., in x, y or z.')
        self.parser.add_argument('--slide_verts', type=float, default=0, help='Percent vertices which will be shifted along the mesh surface.')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='Define whether to use the latest or the best model (for continue_train.')

        self.is_train = True # Set to True if training functionality like autograd is necessary. Since we are training here, this is necessary.
