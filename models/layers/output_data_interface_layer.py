import os
from tempfile import mkstemp
import shutil
import torch


class OutputDataInterfaceLayer():
    def create_obj_file(self, file, vertex_positions, faces, half_edges, old2new_vertex_indices, vcolor):
        """Creates a .obj file at the path "file" from the given data."""
        pass

    def export_segmentation(self, file, half_edges, edges, faces, segments):
        """ Adds segmentation information to the .obj file at the path "file" from the given data."""
        pass

    def transform_predictions_to_match_segmentation_base(self, raw_predictions, meshes):
        return None

class HalfEdgeBasedDataOutputInterfaceLayer(OutputDataInterfaceLayer):
    def create_obj_file(self, file, vertex_positions, faces, half_edges, old2new_vertex_indices, vcolor):
        with open(file, 'w+') as f:
            _write_vertices(f, vertex_positions, vcolor)
            _write_faces(f, faces, old2new_vertex_indices)
            _write_half_edges(f, half_edges, old2new_vertex_indices)

    def export_segmentation(self, file, half_edges, edges, faces, segments):
        segmentations_dict = _get_edge_segmentation_dict(half_edges, segments)
        _write_segmentation_to_obj_file(file=file, segments=segments, segmentations_dict=segmentations_dict)

    def transform_predictions_to_match_segmentation_base(self, raw_predictions, meshes):
        # In the half edge based segmentation, the predictions do not need to be transformed.
        return raw_predictions


class EdgeBasedDataOutputInterfaceLayer(OutputDataInterfaceLayer):
    def create_obj_file(self, file, vertex_positions, faces, half_edges, old2new_vertex_indices, vcolor):
        with open(file, 'w+') as f:
            _write_vertices(f, vertex_positions, vcolor)
            _write_faces(f, faces, old2new_vertex_indices)
            _write_edges(f, half_edges, old2new_vertex_indices)

    def export_segmentation(self, file, half_edges, edges, faces, segments):
        segmentations_dict = _get_edge_segmentation_dict(edges, segments)
        _write_segmentation_to_obj_file(file=file, segments=segments, segmentations_dict=segmentations_dict)

    def transform_predictions_to_match_segmentation_base(self, raw_predictions, meshes):
        batches, n_classes, n_half_edges = raw_predictions.shape
        n_edges = int(n_half_edges/2)

        raw_predictions_transformed = torch.zeros((batches, n_classes, n_edges), device=raw_predictions.device)
        for mesh_index, mesh in enumerate(meshes):
            raw_predictions_of_he_and_opposite_he = raw_predictions[mesh_index, :, mesh.edge_index_to_halfedge_indices]
            combined_predictions = raw_predictions_of_he_and_opposite_he.sum(dim=2)
            raw_predictions_transformed[mesh_index, :, 0:len(mesh.edge_index_to_halfedge_indices)] = combined_predictions

        return raw_predictions_transformed


class FaceBasedDataOutputInterfaceLayer(OutputDataInterfaceLayer):
    def create_obj_file(self, file, vertex_positions, faces, half_edges, old2new_vertex_indices, vcolor):
        with open(file, 'w+') as f:
            _write_vertices(f, vertex_positions, vcolor)
            _write_faces(f, faces, old2new_vertex_indices)

    def export_segmentation(self, file, half_edges, edges, faces, segments):
        segmentations_dict = _get_face_segmentation_dict(faces, segments)
        _write_segmentation_to_obj_file(file=file, segments=segments, segmentations_dict=segmentations_dict, face_based_segmentation=True)

    def transform_predictions_to_match_segmentation_base(self, raw_predictions, meshes):
        batches, n_classes, n_half_edges = raw_predictions.shape
        n_faces = int(n_half_edges/3)

        raw_predictions_transformed = torch.zeros((batches, n_classes, n_faces), device=raw_predictions.device)
        for mesh_index, mesh in enumerate(meshes):
            raw_predictions_of_face_halfedes = raw_predictions[mesh_index, :, mesh.face_index_to_halfedge_indices]
            combined_predictions = raw_predictions_of_face_halfedes.sum(dim=2)
            raw_predictions_transformed[mesh_index, :, 0:len(mesh.face_index_to_halfedge_indices)] = combined_predictions

        return raw_predictions_transformed


def _get_edge_segmentation_dict(edges, predictions):
    edge_segmentations = dict()
    for i, edge in enumerate(edges):
        prediction = predictions[i]
        key = tuple([edge[0].item(), edge[1].item()])
        edge_segmentations[key] = prediction
    return edge_segmentations


def _get_face_segmentation_dict(faces, predictions):
    face_segmentations = dict()
    for i, face in enumerate(faces):
        prediction = predictions[i]
        key = tuple([face[0].item(), face[1].item(), face[2].item()])
        face_segmentations[key] = prediction
    return face_segmentations


def _write_segmentation_to_obj_file(file, segments, segmentations_dict, face_based_segmentation=False):
    """ Ads the segmentation information in the form of an integer representing the segment to an existing .obj file.
        e.g. 'e 0 1 0' means that the edge between vertices 0 and 1 belongs to segment 0
        e.g. 'he 0 1 0' means that the half edge between vertices 0 and 1 belongs to segment 0
    """
    fh, abs_path = mkstemp()

    with os.fdopen(fh, 'w') as new_file:
        with open(file) as old_file:
            # For each line in the file, if it represents an edge, write the segmentation of the edge.
            edge_key = 0
            for line in old_file:
                splitted_line = line.split()
                if not splitted_line:
                    continue

                # If the segmentation is based on half edges, there are lines starting with 'he'.
                # Add the segmentation to those lines.
                if splitted_line[0] == 'he':
                    new_file.write('%s %d\n' % (line.strip(), segments[edge_key]))
                    edge_key += 1

                # If the segmentation is based on edges, there are lines starting with 'e'.
                elif splitted_line[0] == 'e':
                    vertex_0 = int(splitted_line[1])
                    vertex_1 = int(splitted_line[2])
                    key = tuple(sorted([vertex_0, vertex_1]))
                    seg = segmentations_dict[key]
                    new_file.write('%s %d\n' % (line.strip(), seg))
                elif splitted_line[0] == 'f' and face_based_segmentation:
                    # The vertex indices in the .obj file start at 1 so we need to subtract 1 from the indices to get the dict index.
                    vertex_0 = int(splitted_line[1]) - 1
                    vertex_1 = int(splitted_line[2]) - 1
                    vertex_2 = int(splitted_line[3]) - 1
                    key = tuple([vertex_0, vertex_1, vertex_2])
                    seg = segmentations_dict[key]
                    new_file.write('%s %d\n' % (line.strip(), seg))

                else:
                    # Otherwise just copy the line from the file.
                    new_file.write(line)
    # Now the original file is restored.
    os.remove(file)
    shutil.move(abs_path, file)


def _write_vertices(file, vertex_positions, vcolor=None):
    for vi, v in enumerate(vertex_positions):
        vcol = ' %f %f %f' % (vcolor[vi][0], vcolor[vi][1], vcolor[vi][2]) if vcolor is not None else ''
        file.write("v %f %f %f%s\n" % (v[0], v[1], v[2], vcol))


def _write_faces(file, faces, old2new_vertex_indices):
    for face in faces:
        new_vertex_id_1 = old2new_vertex_indices[face[0]]
        new_vertex_id_2 = old2new_vertex_indices[face[1]]
        new_vertex_id_3 = old2new_vertex_indices[face[2]]
        # The vertex indices in the .obj file start at 1 so we need to add 1 to the indices.
        file.write("f %d %d %d\n" % (new_vertex_id_1 + 1, new_vertex_id_2 + 1, new_vertex_id_3 + 1))


def _write_edges(file, half_edges, old2new_vertex_indices, offset=0):
    edges = _get_edges_from_half_edges(half_edges)
    for edge in edges:
        string = "\ne " + str(old2new_vertex_indices[edge[0]] + offset) + " " + str(old2new_vertex_indices[edge[1]] + offset)
        file.write(string)


def _get_edges_from_half_edges(half_edges):
    edges = set()
    for half_edge in half_edges:
        edge = tuple(sorted([half_edge[0].item(), half_edge[1].item()]))
        edges.add(edge)
    edges = list(edges)
    return edges


def _write_half_edges(file, half_edges, old2new_vertex_indices, offset=0):
    for half_edge in half_edges:
        string = "\nhe " + str(old2new_vertex_indices[half_edge[0]] + offset) + " " + str(old2new_vertex_indices[half_edge[1]] + offset)
        file.write(string)

