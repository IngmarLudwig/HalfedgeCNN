import torch
import torch.nn as nn


class MeshUnpool(nn.Module):
    def __init__(self, unroll_target):
        super(MeshUnpool, self).__init__()
        # The unroll target is the number of edges after the unpooling.
        self.__half_edge_target_after_unpooling = unroll_target


    def __call__(self, half_edge_features, meshes):
        return self.forward(half_edge_features, meshes)


    def forward(self, half_edge_features, meshes):
        batch_size, number_of_channels, number_of_half_edges = half_edge_features.shape

        # Get the groups from the mesh before the pooling. The group contains information about which edges where pooled into which.
        # If the given pooling target is bigger then the size of groups, pad with zeros.
        # This is the case if number_input_half_edges is set to a number larger then the original number of half edges of the mesh.
        groups = [self.zero_pad_groups(mesh.get_last_groups_from_history(), number_of_half_edges) for mesh in meshes]
        groups = torch.cat(groups, dim=0).view(batch_size, number_of_half_edges, -1)

        # Get the"occurrences" data from the pooling step. Occurrences contains the number of half edges that were averaged together to get the resulting features.
        # Again, if the given pooling target is bigger then the lenght of occurrences, pad with ones.
        occurrences = [self.__one_pad_occurrences(mesh.get_last_occurrences_from_history()) for mesh in meshes]
        occurrences = torch.cat(occurrences, dim=0).view(batch_size, 1, -1)

        # The occurrences have the shape (batch_size, 1, number of edges before pooling),
        # so we need to expand the dimensions to (batch_size, number of edges after pooling, number of edges before pooling)
        # to be able to divide groups by occurrences
        occurrences = occurrences.expand(groups.shape)

        # In this step the unpooling matrix is constructed.
        # The unpooling matrix for one mesh needs to have the shape he_after_pool x he_before_pooling to be able to change the dimensions
        # of half_edge_features to the dimensions of it from before pooling. The entries of the unpooling matrix need to contain in position (n,m)
        # how much the features of half edge n before unpooling will contribute to the features of halfedge m after unpooling.
        # Groups contains in position (n,m) how often the features of half edge m before pooling where averaged into half edge n after pooling.
        # Therefor the unpooling matrix can be obtained by deviding groups by occurrences.
        # Remember the unpooling matrix contains the values for all matrices in the batch so the dimensions are (batch_size, shape he_after_pool, he_before_pooling).
        # Example
        # occurrences               =  [46.,    18.,   5.,     7.,     9,      ... ]
        # unpooling_matrix before  =  [0.,     0.,     1.,     1.,     2,      ... ]
        # unpooling_matrix after   =  [0.0000, 0.0000, 0.2000, 0.1429, 0,2222, ... ])
        unpooling_matrix = groups / occurrences

        unpooling_matrix = unpooling_matrix.to(half_edge_features.device)

        # reset the meshes to before the last pooling.
        for mesh in meshes:
            mesh.go_back_one_step_in_history()

        # After the multiplication with the unroll matrix, the number of edges (as given by the third dimension of the tensor) is the same as before the pooling.
        # Example: features dimensions: [12, 128, 1200], unpooling matrix dimensions: [12, 1200, 2700] result dimensions: [12, 128, 2700])
        return torch.matmul(half_edge_features, unpooling_matrix)


    def zero_pad_groups(self, groups, half_edge_target_before_unpooling):
        rows_of_groups, columns_of_groups = groups.shape
        # add padding rows if there are less half_edges before unpooling in the current groups then neccessary for the following matrix multiplications
        padding_rows = half_edge_target_before_unpooling - rows_of_groups
        # add padding columns if there are less half edges after unpooling in the current groups then neccessary for the following matrix multiplications
        padding_cols = self.__half_edge_target_after_unpooling - columns_of_groups
        if padding_rows != 0 or padding_cols !=0:
            padding = nn.ConstantPad2d((0, padding_cols, 0, padding_rows), 0)
            groups = padding(groups)
        return groups


    def __one_pad_occurrences(self, occurrences):
        len_of_occurrences = len(occurrences)
        padding_rows = self.__half_edge_target_after_unpooling - len_of_occurrences
        if padding_rows != 0:
            #         (padding_left, padding_right )
            padding = (0           , padding_rows  )
            padding_rows = nn.ConstantPad1d(padding=padding, value=1)
            occurrences = padding_rows(occurrences)
        return occurrences
