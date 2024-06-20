import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends


class HalfEdgeMeshConv(nn.Module):
    """ This class is used to build a columnar matrix. The features of the half-edges are converted into a fake image
    via the defined neighborhood. The changes in this class compared to the original are in the removal of the
    symmetrical functions. The rest of the class deals with the creation of the column matrix, which is then passed to
    the convolution operations of pytorch.

    half_edge_features: half edge features (Batch x Features x Half Edges)
    meshes: list of half edge mesh data-structure (len(mesh) == Batch)
    and applies convolution
    """

    def __init__(self, in_channels, out_channels, kernel_width=5, bias=True):
        super(HalfEdgeMeshConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_width), bias=bias)


    def __call__(self, half_edge_features, meshes):
        return self.forward(half_edge_features, meshes)


    # This function takes most of the computation time of the program during classification.
    def forward(self, half_edge_features, meshes):
        half_edge_features = half_edge_features.squeeze(-1)

        # Time of function to this point (Dell, GPU): 0,0 s (0%).

        # get neighbor information for each mesh in batch
        number_of_half_edges_in_features = half_edge_features.shape[2]
        device = half_edge_features.device
        batch_half_edge_neighborhoods = torch.cat([self.get_prepared_half_edge_neighborhoods_from_mesh(i, number_of_half_edges_in_features, device) for i in meshes], 0)

        # Time of function to this point ( Dell, GPU): 0,0009551048278808594 s (45%) => torch.cat loop takes 45 % of func time.

        # Dimensions of batch_half_edge_neighborhoods are (num_batches, num_channels, num_half_edges, nbh_size_plus_one)
        features_of_neighborhoods = self.__gather_neighborhood_features(half_edge_features, batch_half_edge_neighborhoods)

        # Time of function to this point (Dell, GPU): 0,0019474029541015625 s (92%) =>  create_GeMM_he takes 47 % of func time.

        # Dimensions of half_edge_features after the convolution are (num_batches, num_channels, num_half_edges, 1).
        # because the features of the nbh_size_plus_one half edges in the last dimension where convoluted into one value.
        half_edge_features = self.conv(features_of_neighborhoods)

        # Time of function to this point (Dell, GPU): 0,002115488052368164  s (100%)=>  conv takes 8 % of func time.

        return half_edge_features


    def get_prepared_half_edge_neighborhoods_from_mesh(self, mesh, padding_target, device):
        """ Gets the half_edge_neighborhoods of a mesh, adds the ids of the half_edges to the neighborhoods and
            pads to the number of half edges defined with the  --number_input_faces *3 parameter.
        """
        # Get neighborhood of mesh as tensor.
        half_edge_neighborhoods = torch.tensor(mesh.half_edge_neighborhoods, dtype=torch.float32, device=device)
        # Activate autograd on tensor.
        half_edge_neighborhoods = half_edge_neighborhoods.requires_grad_()

        # Add the indices of the half edges themselves to the neighborhoods.
        # Currently, the neighborhoods in half_edge_neighborhoods contain only the indices of the other half edges.
        # Create a list with entries from 0.0 to half_edge_count as floats.
        half_edge_ids = torch.arange(mesh.half_edge_count, dtype=torch.float32, device=device).unsqueeze(1)
        # Add the ids at the first position like: [[0.0, 0.2, 0.3, ...], [1.0, 0.6, 0.9, ...], ...]
        half_edge_neighborhoods = torch.cat((half_edge_ids, half_edge_neighborhoods), dim=1)

        # Add  number_of_half_edges - mesh.half_edge_count rows filled with zeros at the end of the list.
        # This serves as a padding in the case that the mesh has less half edges then defined by the parameter --number_input_faces * 3 at the bottom.
        #         ( padding_left, padding_right, padding_top, padding_bottom                                )
        padding = (0            , 0            , 0          , padding_target - mesh.half_edge_count)
        half_edge_neighborhoods = F.pad(half_edge_neighborhoods, pad=padding, mode="constant", value=0)

        # Adds one additional dimension, e.g.: [[[0.0, 0.2, 0.3, ...], [1.0, 0.6, 0.9, ...], ...]]
        half_edge_neighborhoods = half_edge_neighborhoods.unsqueeze(0)
        return half_edge_neighborhoods


    def __gather_neighborhood_features(self, half_edge_features, half_edge_neighborhoods):
        """ Gathers the half edge features for the neighborhoods of all meshes.
            Output dimensions: batch_size x nu_channels x num_half_edges x nbh_size_plus_one
        """

        # Get original half_edge_features dimensions before modification.
        num_batches, _, num_half_edges = half_edge_features.shape
        # nbh_size_plus_one is the size of the neighborhood plus one for the half edge itself which was added in get_prepared_half_edge_neighborhood_from_mesh.
        nbh_size_plus_one = half_edge_neighborhoods.shape[2]

        # Prepare half_edge_neighborhoods. The indices need to be adapted and the tensor needs to be flattened.
        half_edge_neighborhoods = self.__prepare_half_edge_indices(half_edge_neighborhoods)

        # Prepare the half_edge_features.
        half_edge_features = self.__prepare_half_edge_features(half_edge_features)

        # Now the features for all neighborhoods are looked up from half_edge_features. The result is a 2D tensor.
        features_of_neighborhoods = torch.index_select(half_edge_features, dim=0, index=half_edge_neighborhoods)
        # Reconstruct original tensor form from the flat features_of_neighborhoods tensor. This now has the shape (num_batches, num_half_edges, nbh_size_plus_one, num_channels).
        features_of_neighborhoods = features_of_neighborhoods.view(num_batches, num_half_edges, nbh_size_plus_one, -1)
        # Now (num_batches, num_half_edges, nbh_size_plus_one, num_channels) are permuted to (num_batches, num_channels, num_half_edges, nbh_size_plus_one).
        features_of_neighborhoods = features_of_neighborhoods.permute(0, 3, 1, 2)

        return features_of_neighborhoods


    def __prepare_half_edge_indices(self, half_edge_neighborhoods):
        """ This function transforms the indices in such a way that they are no longer for a 3D tensor but for a 2D matrix. It also flattens the tensor to 1D and deals with invalid half edge ids.

            The indices in half_edge_neighborhoods (shape: number_meshes_in_batch , number_of_half_edges_in_mesh, nhb_size)
            refer to a 3D tensor with features of the shape (number_meshes_in_batch, number_of_half_edges_in_mesh, number_of_features).
            After this function was applied to half_edge_neighborhoods, the indices now apply to a 2D feature tensor of the shape (number_meshes_in_batch x number_of_half_edges_in_mesh, number_of_features).
            Therefor for mesh 0 in the batch the indices remain the same. For mesh 1 in the batch the number of half edges per mesh needs to be added to every index.
            For mesh 2 in the batch 2x the number of half edges per mesh needs to be added to every index and so forth.

            In addition, all indices in half_edge_neighborhoods, that has originally the shape (number_of_meshes_in_batch, number_of_half_edges_in_mesh, nbh_size_plus_one) are in one vector
            of the shape (number_of_meshes_in_batch x number_of_half_edges_in_mesh x nbh_size_plus_one).
        """

        # First, every half edge index in half_edge_neighborhoods is incremented by one. This is necessary, because during creation of the neighborhood invalid half_edges get assigned to -1.
        # So to deal with those values all indices are incremented, making all invalid indices point to the half edge with index 0. In __prepare_half_edge_features then a new half edge
        # is entered into position 0 with all features = 0.
        half_edge_neighborhoods = half_edge_neighborhoods + 1

        # Now, the indices are changed so that they apply to a flattened 2D half_edge_features tensor.
        # Before the next block, half_edge_neighborhoods contains the half edge indices of all the half edges in the neighborhood of one half edge identified by [mesh_index, half_edge_index] (plus one from last call).
        # E.g. (with a nbh size of 5, because the half-edge itself is also in the neighborhood): [2, 42, 3, 1, 40, 41]
        # This might look as follows for mesh 0 and mesh 1 (The entries at the end are all 1s because of the one padding of half_edge_neighborhoods before (get_prepared_half_edge_neighborhoods_from_mesh), this is not always the case):
        # [[[1, 91, 2, 3, 92, 93],  [2, 42, 3, 1, 40, 41],      [3, 4392, 1, 2, 439, 4391],  ..., [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
        #  [[1, 29, 2, 3, 3.0, 28], [2, 4362, 3, 1, 436, 4361], [3, 4343, 1, 2, 4344, 4342], ..., [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], ...]
        #
        # In the function mesh_index * number_of_half_edges_in_mesh+1 is added to each entry, depending on the current mesh. Again, because of the one-Padding, the last entries might be the same.
        # This might look as follows for mesh 0 and mesh 1:
        # [[[1, 91, 2, 3, 92, 93],                [2, 42, 3, 1, 40, 41],                 [3, 4392, 1, 2, 439, 43910],          ..., [1, 1, 1, 1, 1, 1],                   [1, 1, 1, 1, 1, 1],                   [1, 1, 1, 1, 1, 1]],
        #  [[4562, 4590, 4563, 4564, 4591, 4589], [4563, 8923, 4564, 4.562, 8921, 8922], [4564, 8904, 4562, 4563, 8905, 8903], ..., [4562, 4562, 4562, 4562, 4562, 4562], [4562, 4562, 4562, 4562, 4562, 4562], [4562, 4562, 4562, 4562, 4562, 4562]], ...]
        number_of_meshes_in_batch, number_of_half_edges_in_mesh, nbh_size_plus_one = half_edge_neighborhoods.shape
        for mesh_number in range(number_of_meshes_in_batch):
            half_edge_neighborhoods[mesh_number] += (number_of_half_edges_in_mesh+1) * mesh_number

        # Flatten half_edge_neighborhoods to 1D.
        half_edge_neighborhoods = half_edge_neighborhoods.view(-1).long()

        return half_edge_neighborhoods


    def __prepare_half_edge_features(self, half_edge_features):
        """ This function prepares half_edge_features for the construction of the convolution tensor by padding and flattening it.
        """
        num_batches, num_channels, _ = half_edge_features.shape
        device = half_edge_features.device

        # Pad half_edge_features.
        # This is necessary, because during creation of the neighborhood invalid half_edges get assigned to -1.
        # Then, in __prepare_half_edge_indices, all indices are incremented, making all invalid indices point to the half edge with index 0.
        # Now we enter a new half edge at position 0 with all features = 0.
        # To do this, we first create a tensor with num_batches matrices with num_channels rows and 1 column, filled with zeros.
        padding = torch.zeros((num_batches, num_channels, 1), requires_grad=True, device=device)
        # Now this tensor is used to effectively add a new half edge in every mesh with every feature equaling zero. E.g. half_edge_features.shape before: (12, 3, 4560), after:  (12, 3, 4561)
        half_edge_features = torch.cat((padding, half_edge_features), dim=2)

        # Flatten half_edge_features.
        num_half_edges_new = half_edge_features.shape[2] #Because of the padding, half_edge_features now has one more half edge in each mesh.
        # Switch rows (num channels/features) and columns (num half edges).
        half_edge_features = half_edge_features.permute(0, 2, 1)
        # Make sure that half_edge_features is in one contiguous block in the memory.
        half_edge_features = half_edge_features.contiguous()
        # Make 2D from 3D by putting all meshes one after the other.
        half_edge_features = half_edge_features.view(num_batches * num_half_edges_new, num_channels)

        return half_edge_features