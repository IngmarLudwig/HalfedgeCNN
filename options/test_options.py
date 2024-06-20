from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--number_augmentations', type=int, default=1, help='Number of augmented versions of meshes. 1 means no augmentation.')
        self.parser.add_argument('--phase', type=str, choices={"train", "val", "test"}, default='test', help='Name of the subfolder in datasets to use the meshes from.')
        self.parser.add_argument('--which_epoch', type=str, choices={"latest", "best"}, default='latest', help='Define whether to use the latest or the best model.')
        self.is_train = False  # Set to True if training functionality like autograd is necessary. Since we are testing here, this is not necessary, and we can save some computation time.
