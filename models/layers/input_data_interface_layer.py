import os
import numpy as np

from util.util import pad

class InputDataInterfaceLayer():
    def __init__(self, target_num_half_edges):
        super().__init__()
        self.num_half_edges = int(target_num_half_edges)

    def read_hard_segmentation_for_training(self, obj_file_path, padding=False, offset=0):
        """ Reads the file with one label per half edge or creates such a file from the segmentation file based on the selected entity."""
        hard_labels = None
        return hard_labels

    def read_soft_segmentation_for_testing(self, obj_file_path, padding=False, perform_ceil=False):
        """ Reads the file with multiple segmentation labels per edge.
           In such a file, for every edge, number of segmentation classes numbers are given,
           each determining how much an edge belongs to this class.
           The soft segmentation is not translated, so in case of edge based segmentation, the edge based soft
           segmentation is returned.
        """
        soft_labels = None
        return soft_labels

    def _read_segmentation_file(self, seg_file_path, padding, perform_ceil, offset):
        labels = None
        if not os.path.exists(seg_file_path):
            raise Exception("Segmentation file " + seg_file_path + " does not exist. Maybe set wrong segmentation_base flag in train.py or test.py?")

        with open(seg_file_path, 'r') as f:
            labels = np.loadtxt(f, dtype='float64')

        if perform_ceil:
            labels = np.array(labels > 0, dtype=np.int32)

        if offset != 0:
            labels = labels - offset

        # Padding needs to be done after offsetting, because the offset might be -1, which is the padding value.
        if padding:
            labels = pad(labels, self.num_half_edges, val=-1, dim=0)

        return labels


class HalfEdgeBasedDataInputInterfaceLayer(InputDataInterfaceLayer):
    def __init__(self, target_num_half_edges):
        super().__init__(target_num_half_edges)

    def read_hard_segmentation_for_training(self, obj_file_path, padding=False, offset=0):
        half_edge_based_hard_segmentations_path = create_heseg_file_path(obj_file_path)
        return self._read_segmentation_file(half_edge_based_hard_segmentations_path, padding, perform_ceil=False, offset=offset)

    def read_soft_segmentation_for_testing(self, obj_file_path, padding=False, perform_ceil=False):
        half_edge_based_soft_segmentation_path = create_sheseg_file_path(obj_file_path)
        return self._read_segmentation_file(half_edge_based_soft_segmentation_path, padding, perform_ceil=perform_ceil, offset=0)


class EdgeBasedDataInputInterfaceLayer(InputDataInterfaceLayer):
    def __init__(self, target_num_half_edges):
        super().__init__(target_num_half_edges)

    def read_hard_segmentation_for_training(self, obj_file_path, padding=False, offset=0):
        half_edge_based_hard_segmentations_path = create_heseg_file_path(obj_file_path)
        # The hard segmentation is used for training and training is always half edge based.
        # Since a half edge is in a 1:n realation to a vertex, edge and face, a simple translation is always possible.
        if not os.path.exists(half_edge_based_hard_segmentations_path):
            translate_edge_to_half_edge_based_hard_segmentation(obj_file_path)
        return self._read_segmentation_file(half_edge_based_hard_segmentations_path, padding, perform_ceil=False, offset=offset)

    def read_soft_segmentation_for_testing(self, obj_file_path, padding=False, perform_ceil=False):
        edge_based_soft_segmentation_path = create_seseg_file_path(obj_file_path)
        return self._read_segmentation_file(edge_based_soft_segmentation_path, padding, perform_ceil, offset=0)


class FaceBasedDataInputInterfaceLayer(InputDataInterfaceLayer):
    def __init__(self, target_num_half_edges):
        super().__init__(target_num_half_edges)

    def read_hard_segmentation_for_training(self, obj_file_path, padding=False, offset=0):
        half_edge_based_hard_segmentations_path = create_heseg_file_path(obj_file_path)
        # The hard segmentation is used for training and training is always half edge based.
        # Since a half edge is in a 1:n realation to a vertex, edge and face, a simple translation is always possible.
        if not os.path.exists(half_edge_based_hard_segmentations_path):
            translate_face_to_half_edge_based_hard_segmentation(obj_file_path)
        return self._read_segmentation_file(half_edge_based_hard_segmentations_path, padding, perform_ceil=False, offset=offset)

    def read_soft_segmentation_for_testing(self, obj_file_path, padding=False, perform_ceil=False):
        face_based_soft_segmentation_path = create_fsseg_file_path(obj_file_path)
        return self._read_segmentation_file(face_based_soft_segmentation_path, padding, perform_ceil, offset=0)


# Helper functions

def read_vertex_positions(file):
    vertex_positions = []
    with open(file) as f:
        for line in f:
            line = line.strip()
            splitted_line = line.split()
            if not splitted_line:
                continue
            elif splitted_line[0] == 'v':
                vertex_positions.append([float(v) for v in splitted_line[1:4]])
    vertex_positions = np.asarray(vertex_positions, dtype=float)
    return vertex_positions


def read_faces(file):
    """ Reads the mesh data from an .obj file with vertices marked with v and faces marked with f.
        The faces are in a list of the form [[index1_1][index2_1][index3_1], [index1_2][index2_2][index3_2], ...].
    """
    num_vertices = _get_num_vertices(file)

    faces = []
    with open(file) as f:
        for line in f:
            line = line.strip()
            splitted_line = line.split()
            if not splitted_line:
                continue
            elif splitted_line[0] == 'f':
                face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
                assert len(face_vertex_ids) == 3
                face_vertex_ids = [(ind - 1) if (ind >= 0) else (num_vertices + ind) for ind in face_vertex_ids]
                faces.append(face_vertex_ids)

    faces = np.asarray(faces, dtype=int)
    # The indices in faces should be between 0 and the highest vertex index.
    assert np.logical_and(faces >= 0, faces < num_vertices).all()
    return faces


def extract_half_edges(faces):
    """Returns a list of half edges of the form [[index1_1][index2_1], [index1_2][index2_2], ...]
       with the indices of the vertices of the half edges.
    """
    return _extract_edges_or_half_edges(faces, extract_half_edges=True)


def extract_edges(faces):
    """ Returns a list of edges of the form [[index1_1][index2_1], [index1_2][index2_2], ...]
       with the indices of the vertices of the edges.
    """
    return _extract_edges_or_half_edges(faces, extract_half_edges=False)


def _extract_edges_or_half_edges(faces, extract_half_edges):
    edges_set = set()
    edges = []
    for face in faces:
        faces_edges = get_edges_from_face(face)
        for edge in faces_edges:
            if not extract_half_edges:
                edge = sorted(edge)
            edge = tuple(edge)
            # Since the order is important here, we use a list and a set instead of creating a list from a set.
            if edge not in edges_set:
                edges_set.add(edge)
                edges.append(list(edge))
    return edges


def get_edges_from_face(face):
    faces_edges = []
    for i in range(3):
        cur_edge = [face[i], face[(i + 1) % 3]]
        faces_edges.append(cur_edge)
    return faces_edges


def _get_num_vertices(obj_file_path):
    num_vertices = 0
    with open(obj_file_path) as f:
        for line in f:
            line = line.strip()
            splitted_line = line.split()
            if not splitted_line:
                continue
            elif splitted_line[0] == 'v':
                num_vertices += 1
    return num_vertices


def create_index_dict(edges):
    return {tuple(edge): i for i, edge in enumerate(edges)}


def translate_edge_to_half_edge_based_hard_segmentation(obj_path):
    """ Translates the edge based hard segmentation file to a half edge based segmentation file.
        The generated files are stored in the newly created hseg folder of the dataset.
    """
    create_hseg_folder(obj_path)
    eseg_path = create_eseg_file_path(obj_path)
    output_file_name = create_heseg_file_path(obj_path)
    translate_edge_to_half_edge_based_segmentation(obj_path, output_file_name, eseg_path)


def translate_edge_to_half_edge_based_segmentation(obj_path, file_path, seg_path):
    """ Common part of translation code for hard and soft segmentations.
        Translates edge based segmentations to half edge based segmentations.
    """
    faces = read_faces(obj_path)
    half_edges = extract_half_edges(faces)
    edges = extract_edges(faces)
    edge_to_edge_index = create_index_dict(edges)
    # Can be normal or soft segmentation.
    edge_based_segmentation = read_segmentation_file(seg_path)

    with open(file_path, "w") as f:
        for half_edge in half_edges:
            edge = tuple(sorted(list(half_edge)))
            edge_index = edge_to_edge_index[edge]
            f.write(str(edge_based_segmentation[edge_index]) + "\n")


def translate_face_to_half_edge_based_hard_segmentation(obj_path):
    create_hseg_folder(obj_path)
    fseg_path = create_fseg_file_path(obj_path)
    output_file_name = create_heseg_file_path(obj_path)
    translate_face_to_half_edge_based_segmentation(obj_path, output_file_name, fseg_path)


def translate_face_to_half_edge_based_segmentation(obj_path, file_path, seg_path):
    """ Common part of translation code for hard and soft segmentations.
        Translates edge based segmentations to half edge based segmentations.
    """
    faces = read_faces(obj_path)
    half_edges = extract_half_edges(faces)
    half_edge_to_face_index = create_half_edge_to_face_index_dict(faces)
    # Can be normal or soft segmentation.
    edge_based_segmentation = read_segmentation_file(seg_path)

    with open(file_path, "w") as f:
        for half_edge in half_edges:
            face_index = half_edge_to_face_index[tuple(half_edge)]
            segmentation = edge_based_segmentation[face_index]
            f.write(str(segmentation) + "\n")


def create_half_edge_to_face_index_dict(faces):
    he_to_face_id_dict = dict()
    for face_index, face in enumerate(faces):
        face_half_edges = get_edges_from_face(face)
        for half_edge in face_half_edges:
            key = tuple(half_edge)
            he_to_face_id_dict[key] = face_index
    return he_to_face_id_dict


def read_segmentation_file(file):
    """ Reads the segmentation data from a segmentation file with segments in the same order as the edges in the .obj file.
        Can be one or multiple values per line.
    """
    segmentation = []
    with open(file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            else:
                segmentation.append(line)
    return segmentation


###### Folders and file paths ######

def create_heseg_file_path(obj_file_path):
    """ Halfedge based hard segmentation file."""
    return _create_segmentation_file_path(obj_file_path, 'hseg', '.heseg')


def create_sheseg_file_path(obj_file_path):
    """ Halfedge based soft segmentation."""
    return _create_segmentation_file_path(obj_file_path, 'hsseg', '.sheseg')


def create_eseg_file_path(obj_file_path):
    """ Edge based hard segmentation."""
    return _create_segmentation_file_path(obj_file_path, 'seg', '.eseg')


def create_seseg_file_path(obj_file_path):
    """ Edge based soft segmentation."""
    return _create_segmentation_file_path(obj_file_path, 'sseg', '.seseg')


def create_fseg_file_path(obj_file_path):
    """ Face based hard segmentation."""
    return _create_segmentation_file_path(obj_file_path, 'fseg', '.seg')


def create_fsseg_file_path(obj_file_path):
    """ Face based soft segmentation."""
    return _create_segmentation_file_path(obj_file_path, 'fsseg', '.sseg')


def _create_segmentation_file_path(obj_file_path, folder_name, postfix):
    """ Returns the path to the half edge based segmentation file that corresponds to the given obj file."""
    obj_base_path = os.path.dirname(obj_file_path)
    obj_base_path = os.path.normpath(obj_base_path)
    base_path = os.path.dirname(obj_base_path)
    seg_file_path = os.path.join(base_path, folder_name, os.path.basename(obj_file_path).split('.')[0] + postfix)
    return seg_file_path


def create_hseg_folder(obj_path):
    """ Folder for halfedge based hard segmentations."""
    _create_segmentation_folder(obj_path, 'hseg')


def _create_segmentation_folder(obj_file_path, folder_name):
    obj_base_path = os.path.dirname(obj_file_path)
    base_path = os.path.dirname(obj_base_path)
    folder_path = os.path.join(base_path, folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
