import numpy as np
import torch


class AccuracyCalculator():
    def calculate_segmentation_accuracy(self, predictions, ssegs, meshes):
        """Depending on the base of the segmentation, the predictions need to be transformed to match the soft
           segmentations (see read_soft_segmentation above).
           This function caclulates the accuracy of the predictions calling the right transformation function before.
        """
        return None


class HalfEdgeBasedAccuracyCalculator(AccuracyCalculator):
    def calculate_segmentation_accuracy(self, predictions, ssegs, meshes):
        half_edge_areas = []
        for mesh in meshes:
            half_edge_areas.append(torch.from_numpy(mesh.half_edge_areas))
        half_edges_per_half_edge = 1
        return _calculate_seg_accuracy(predictions, ssegs, meshes, half_edges_per_half_edge, half_edge_areas)


class EdgeBasedAccuracyCalculator(AccuracyCalculator):
    def calculate_segmentation_accuracy(self, predictions, ssegs, meshes):
        edge_areas = []
        for mesh in meshes:
            edge_areas.append(torch.from_numpy(mesh.edge_areas))
        half_edges_per_edge = 2
        return _calculate_seg_accuracy(predictions, ssegs, meshes, half_edges_per_edge, edge_areas)


class FaceBasedAccuracyCalculator(AccuracyCalculator):
    def calculate_segmentation_accuracy(self, predictions, ssegs, meshes):
        face_areas = []
        for mesh in meshes:
            face_areas.append(torch.from_numpy(mesh.face_areas))
        half_edges_per_face = 3
        return _calculate_seg_accuracy(predictions, ssegs, meshes, half_edges_per_face, face_areas)


def _calculate_seg_accuracy(predictions, ssegs, meshes, num_halfedges_per_entety, areas):
    correct = 0
    ssegs = ssegs.squeeze(-1)
    correct_mat = ssegs.gather(2, predictions.cpu().unsqueeze(dim=2))
    for mesh_id, mesh in enumerate(meshes):
        entity_count = int(mesh.half_edge_count / num_halfedges_per_entety)
        correct_vec = correct_mat[mesh_id, :entity_count, 0]
        correct += (correct_vec.float() * areas[mesh_id]).sum()
    return correct
