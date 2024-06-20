import torch
from torch.nn import ConstantPad2d


class MeshUnion:
    def __init__(self, n, device=torch.device('cpu')):
        self.__size = n
        self.rebuild_features = self.rebuild_features_average
        # Groups contains a tensor with the size n x n with n being the current number of edges.
        # Each row initially contains a one at the index of the edge, representing,
        # that only this edge contributed to the features up to this point.
        self.groups = torch.eye(n, device=device)

    def union(self, source, target):
        # In the case of a union between two halfedges, the contribution of the halfedge with the index "source"
        # is noted in the tensor "groups" by adding a one to the row with the index "target" at the index [target][source].
        # In the case, that already multiple halfedges contributed to the features of the halfedge with the index "source",
        # all the contributions to "source" are are also added to "target".
        # For example: There are 6 edges and to the features of the edge with the index 3 already contributed 2 other edges.
        # The row with the index 3 is therefore [0, 0, 0, 1, 1, 0, 1].
        # The edge with the index 0 has not yet been merged to and its row is therefore [1, 0, 0, 0, 0, 0, 0].
        # The edge with the index 3 (source) is now merged into the edge with the index 0 (target),
        # resulting in the row 0 being now [1, 0, 0, 1, 1, 0, 1].
        self.groups[target, :] += self.groups[source, :]
        

    def get_group(self, edge_key):
        return self.groups[edge_key, :]

    def get_occurrences(self):
        # Occurences is a 1D Array, containing for each halfedge index how many features were added up in the features tensor.
        occurrences = torch.sum(self.groups, 0)
        return occurrences

    def get_groups(self, tensor_mask):
        # Groups is an 2D Array containing for each halfedge all the features that contributed to the current features in this step.
        self.groups = torch.clamp(self.groups, 0, 1)
        return self.groups[tensor_mask, :]

    def rebuild_features_average(self, half_edge_features, mask, target_edges):
        # Reduces the group size to the number of edges after the pooling in one direction by removing the deleted edges.
        # Also clamps values between 0 an 1.
        self.prepare_groups(half_edge_features, mask)
        # self.groups contains which edges that contributed to the features of the edges in the current batch.
        # By multiplying the features with the groups, the features of the edges are summed up.
        # So fe contains the sums of the features of the edges in the case that multiple edges contributed to the features of an edge.
        # features has the dimensions number of channels x number of edges.
        # self.groups has the dimensions number of edges before pooling x number of edges after pooling (after the prepare_groups call).
        # fe therefore has the dimensions number of channels x number of edges after pooling.
        fe = torch.matmul(half_edge_features.squeeze(-1), self.groups)

        # "occurrences" is a tensor containting the number or features that were added up in the features tensor.
        # "occurrences" has the dimensions channels x number of edges after pooling, so the number of occurences for each edge "number of channels" times.
        occurrences = torch.sum(self.groups, 0).expand(fe.shape)
        # So by dividing through the occurrences we get the average of the features.
        fe = fe / occurrences
        padding_b = target_edges - fe.shape[1]
        if padding_b > 0:
            padding_b = ConstantPad2d((0, padding_b, 0, 0), 0)
            fe = padding_b(fe)
        return fe

    def prepare_groups(self, half_edge_features, mask):
        tensor_mask = torch.from_numpy(mask)
        self.groups = torch.clamp(self.groups[tensor_mask, :], 0, 1).transpose_(1, 0)
        padding_a = half_edge_features.shape[1] - self.groups.shape[0]
        if padding_a > 0:
            padding_a = ConstantPad2d((0, 0, 0, padding_a), 0)
            self.groups = padding_a(self.groups)
