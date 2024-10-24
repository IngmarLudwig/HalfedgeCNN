import numpy as np
import os
import ntpath

from models.layers.input_data_interface_layer import extract_edges, read_faces, read_vertex_positions, get_edges_from_face, extract_half_edges, \
    create_index_dict


class MeshData:
    def __getitem__(self, item):
        return eval('self.' + item)


def get_mesh_data(file: str, opt):
    """ Fills empty mesh object from path to .obj file. """
    path = get_random_mesh_path(file, opt.number_augmentations)

    mesh_data = None
    if os.path.exists(path):
        mesh_data = np.load(path, encoding='latin1', allow_pickle=True)
        
        # Restore nested version of vertex_to_half_edges 
        vertex_to_half_edges_flat=mesh_data['vertex_to_half_edges_flat']
        length_list = mesh_data['vertex_to_half_edges_flat_length_list']

        vertex_to_half_edges= []    
        cnt=0
        for length in length_list:
            list_part = vertex_to_half_edges_flat[cnt: cnt+length]
            vertex_to_half_edges.append(list_part.tolist())
            cnt += length

        mesh_data_dict = {}
        for key in mesh_data:
            mesh_data_dict[key] = mesh_data[key]
        mesh_data = mesh_data_dict

        mesh_data['vertex_to_half_edges'] = vertex_to_half_edges
    else:
        mesh_data = from_scratch(file, opt)
        save_mesh_data(path, mesh_data)

    return mesh_data


def get_random_mesh_path(file: str, number_augmentations: int):
    """ Constructs a path to store or load mesh data from or to.
        Has the functionality to create a filename that contains a RANDOM number.
        This enables the functionality to generate, store and use more and more randomly augmented
        meshes. The augmentations are performed in fill_mesh.
    """
    filename, _ = os.path.splitext(file)
    dir_name = os.path.dirname(filename)
    prefix = os.path.basename(filename)
    load_dir = os.path.join(dir_name, 'cache')
    load_file = os.path.join(load_dir, '%s_%03d.npz' % (prefix, np.random.randint(0, number_augmentations)))
    if not os.path.isdir(load_dir):
        os.makedirs(load_dir, exist_ok=True)
    return load_file


def save_mesh_data(load_path, mesh_data):

    # Create non-nested version of vertex_to_half_edges to be able to store as NPZ-File
    vertex_to_half_edges_flat = []
    length_list = []
    for xs in mesh_data.vertex_to_half_edges:
        vertex_to_half_edges_flat.extend(xs)
        length_list.append(len(xs))
    
    np.savez_compressed(load_path,
                        vertex_positions=mesh_data.vertex_positions,
                        filename=mesh_data.filename,
                        half_edges=mesh_data.half_edges,
                        edges=mesh_data.edges,
                        faces=mesh_data.faces,
                        half_edge_next=mesh_data.half_edge_next,
                        half_edge_opposite=mesh_data.half_edge_opposite,
                        half_edge_count=mesh_data.half_edge_count,
                        half_edge_areas=mesh_data.half_edge_areas,
                        edge_areas=mesh_data.edge_areas,
                        face_areas=mesh_data.face_areas,
                        edge_index_to_halfedge_indices=mesh_data.edge_index_to_halfedge_indices,
                        face_index_to_halfedge_indices=mesh_data.face_index_to_halfedge_indices,
                        #vertex_to_half_edges=mesh_data.vertex_to_half_edges, # New Version of savez_compressed does not except inhomogeneous list shape
                        vertex_to_half_edges_flat=vertex_to_half_edges_flat,
                        vertex_to_half_edges_flat_length_list = length_list,
                        half_edge_features=mesh_data.half_edge_features)


def from_scratch(file, opt):
    """ Reads the neccessary data from an obj-file and performs the desired augmentations.
        CAUTION: First call comes from "get_mean_std" of base_dataset.
                 In this call opt.number_augmentations is manually set to 0!
                 So in the first from_scratch call for ALL Meshes, no augmentation is performed!
     """
    mesh_data = MeshData()
    mesh_data.filename = ntpath.split(file)[1]

    mesh_data.vertex_positions = read_vertex_positions(file)
    mesh_data.faces, mesh_data.face_normals, mesh_data.face_areas = get_and_clean_face_data(file, opt, mesh_data.vertex_positions, mesh_data.filename)

    if opt.number_augmentations > 1:
        mesh_data.faces = pre_augmentation(mesh_data.vertex_positions, opt, mesh_data.faces)

    mesh_data.edges = np.array(extract_edges(mesh_data.faces), dtype=np.int32)

    mesh_data.half_edges, mesh_data.half_edge_count, mesh_data.half_edge_opposite, mesh_data.half_edge_next = get_and_check_half_edge_data(mesh_data.faces, mesh_data.filename)

    mesh_data.vertex_to_half_edges = extract_vertex_to_half_edges_map(mesh_data.faces, mesh_data.vertex_positions)

    mesh_data.half_edge_areas = extract_half_edge_areas(mesh_data.face_areas, mesh_data.faces)
    mesh_data.edge_areas      = extract_edge_areas(     mesh_data.face_areas, mesh_data.faces)

    if opt.number_augmentations > 1:
        post_augmention(mesh_data, opt)

    mesh_data.half_edge_lengths = calculate_edge_lengths(mesh_data)

    mesh_data.half_edge_features = extract_features(mesh_data, opt.feat_selection)

    mesh_data.edge_index_to_halfedge_indices = get_edge_index_to_halfedge_indices_map (file)
    mesh_data.face_index_to_halfedge_indices = get_face_index_to_half_edge_indices_map(file)

    # At the very end norm face areas.
    mesh_data.face_areas = np.array(mesh_data.face_areas, dtype=np.float32) / np.sum(mesh_data.face_areas)
    return mesh_data


def get_edge_index_to_halfedge_indices_map(obj_file_path):
    faces = read_faces(obj_file_path)
    edges = extract_edges(faces)

    half_edges = extract_half_edges(faces)
    half_edge_to_index_dict = create_index_dict(half_edges)

    edge_index_to_half_edge_indices = np.empty((len(edges), 2), dtype=np.int32)
    for i, edge in enumerate(edges):
        opposite_half_edge = [edge[1], edge[0]]
        edge_index_to_half_edge_indices[i][0] = half_edge_to_index_dict[tuple(edge)]
        edge_index_to_half_edge_indices[i][1] = half_edge_to_index_dict[tuple(opposite_half_edge)]
    return edge_index_to_half_edge_indices


def get_face_index_to_half_edge_indices_map(obj_file_path):
    faces = read_faces(obj_file_path)
    half_edges = extract_half_edges(faces)
    half_edge_to_index_dict = create_index_dict(half_edges)

    face_index_to_half_edge_indices = np.empty((len(faces), 3), dtype=np.int32)
    for i, face in enumerate(faces):
        face_edges = get_edges_from_face(face)
        face_index_to_half_edge_indices[i] =[half_edge_to_index_dict[tuple(face_edge)] for face_edge in face_edges]
    return face_index_to_half_edge_indices


def get_and_clean_face_data(file, opt, vertex_positions, filename):
    faces = read_faces(file)
    face_normals, face_areas = compute_face_normals_and_areas(vertex_positions, filename, faces)

    faces, face_areas, face_normals = remove_faces_face_with_zero_area(face_areas, face_normals, faces)
    faces, face_areas, face_normals = remove_face_from_edges_with_more_then_two_faces(face_areas, face_normals, faces)

    assert np.logical_and(faces >= 0, faces < len(vertex_positions)).all()

    return faces, face_normals, face_areas


def get_and_check_half_edge_data(faces, filename):
    half_edge_as_unsorted_tuple2key = extract_half_edge_as_unsorted_tuple2key(faces)

    half_edges = extract_half_edges(faces)
    check_half_edges(half_edges, half_edge_as_unsorted_tuple2key, filename)
    half_edges = np.array(half_edges, dtype=np.int32)

    half_edge_opposite = get_opposite_half_edges(half_edges, half_edge_as_unsorted_tuple2key)
    half_edge_opposite = np.array(half_edge_opposite, dtype=np.int64)

    half_edge_next = extract_next_edges(faces)
    half_edge_next = np.array(half_edge_next, dtype=np.int64)

    half_edge_count = len(half_edges)

    assert check_halfedges_circularity(half_edge_count, half_edge_next)

    return half_edges, half_edge_count, half_edge_opposite, half_edge_next


def pre_augmentation(vertex_positions, opt, faces):
    if hasattr(opt, 'scale_verts') and opt.scale_verts:
        scale_verts(vertex_positions)
    if hasattr(opt, 'flip_edges') and opt.flip_edges:
        # Here the old edge (not half_edge) based version works fine, because only the face-data is changed.
        faces = flip_edges(vertex_positions, opt.flip_edges, faces)
    return faces


def scale_verts(vertex_positions, mean=1, var=0.1):
    for i in range(vertex_positions.shape[1]):
        vertex_positions[:, i] = vertex_positions[:, i] * np.random.normal(mean, var)


def get_vertex_and_face_ids_for_all_edges(faces):
    # A counter for synchronizing the indices in edge_as_sorted_tuple2key and key2vertex_and_face_ids_of_edge.
    edge_count = 0
    # A list containing a numpy array for eache edge with the indices of the two vertices of the edge (in the mesh_data.vertex_positions array) in place 1 and 2.
    # and the indices of the two faces belonging to the edge in the faces array in place 3 and 4 like this: [vi1, v12, fi1, fi2].
    key2vertex_and_face_ids_of_edge = []
    # A dictionary containing the index of the edge in key2vertex_and_face_ids_of_edge for each each in sorted tuple form.
    edge_as_sorted_tuple2key = dict()

    for face_id, face in enumerate(faces):
        # For each face get all edges one after the other, but sorted, so that the edge (1,2) equals (2,1).
        for i in range(3):
            current_edge = tuple(sorted((face[i], face[(i + 1) % 3])))

            # If the current edge has not yet bin found, add an entry to the edge_as_sorted_tuple2key dict
            # with the index of the edge in the key2vertex_and_face_ids_of_edge list and
            # add an array with the two vertex indices of the edge and two following -1 entries to the key2vertex_and_face_ids_of_edge list.
            # The two -1 entries are there for later adding the face_ids of the two faces belonging to the edge.
            if current_edge not in edge_as_sorted_tuple2key:
                edge_as_sorted_tuple2key[current_edge] = edge_count
                edge_count += 1
                key2vertex_and_face_ids_of_edge.append(np.array([current_edge[0], current_edge[1], -1, -1]))

            # Get the index of the current edge for the key2vertex_and_face_ids_of_edge list.
            edge_key = edge_as_sorted_tuple2key[current_edge]

            # If the first of the two face_id places in key2vertex_and_face_ids_of_edge is still -1, this means that we are in the
            # first face belonging to the edge. therefore add the face_id of the face in position 3.
            if key2vertex_and_face_ids_of_edge[edge_key][2] == -1:
                key2vertex_and_face_ids_of_edge[edge_key][2] = face_id
            # Otherwise we are in the second face belonging to the edge. in this case put the face_id in position 4.
            else:
                key2vertex_and_face_ids_of_edge[edge_key][3] = face_id

    return edge_count, np.array(key2vertex_and_face_ids_of_edge), edge_as_sorted_tuple2key


def remove_faces_face_with_zero_area(face_areas, face_normals, faces):
    mask = np.ones(len(faces), dtype=bool)
    for face_id, face in enumerate(faces):
        # Remove faces with area = 0.
        if face_areas[face_id] == 0:
            mask[face_id] = False
            print("Warning: Found a face with area = 0!")
    # Return the faces, face areas and face normals without the faces identified as problematic.
    return faces[mask], face_areas[mask], face_normals[mask]


def remove_face_from_edges_with_more_then_two_faces(face_areas, face_normals, faces):
    edges_set = set()
    mask = np.ones(len(faces), dtype=bool)
    for face_id, face in enumerate(faces):
        faces_edges = []
        is_wrong = False
        for i in range(3):
            # Get the three vertices of the face one after the other in the form (vertex_index_1, vertex_index_2).
            current_edge = (face[i], face[(i + 1) % 3])

            # Each edge is added to the edge set two times. Once in the form (i1,i2) and once in the form (i2, i1).
            # Once for each face it is connected two. Since each face should be connected to two faces, it should not
            # be in the edges set more then once. If an edge at this point is already in the edges list, it therefore
            # means that the current edge is connected to three faces.
            if current_edge in edges_set:
                is_wrong = True
                break
            else:
                faces_edges.append(current_edge)

        # If a face was identified as wrong, it is removed from the faces.
        if is_wrong:
            mask[face_id] = False
            print("Warning: Found an edge that is connected to three faces!")
        else:
            for idx, edge in enumerate(faces_edges):
                edges_set.add(edge)

    # Return the faces, face areas and face normals without the faces identified as problematic.
    return faces[mask], face_areas[mask], face_normals[mask]


def compute_face_normals_and_areas(vertex_positions, filename, faces):
    face_normals = np.cross(vertex_positions[faces[:, 1]] - vertex_positions[faces[:, 0]],
                            vertex_positions[faces[:, 2]] - vertex_positions[faces[:, 1]])
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    face_normals /= face_areas[:, np.newaxis]
    assert (not np.any(face_areas[:, np.newaxis] == 0)), 'has zero area face: %s' % filename
    face_areas *= 0.5
    return face_normals, face_areas


def check_halfedges_circularity(half_edge_count, half_edge_next):
    for i in range(half_edge_count):
        next_key = half_edge_next[i]
        previous_key = half_edge_next[next_key]
        next_previous_key = half_edge_next[previous_key]
        if next_previous_key != i:
            raise Exception('Half edge mesh_data is not intact')
    return True


def extract_features(mesh_data, feature_selection):
    features = []

    vertices_of_adjacent_faces = get_face_and_opposite_face_vertices(mesh_data)

    with np.errstate(divide='raise'):
        try:
            feature_extractors = []
            # Original features used in MeshCNN.
            if feature_selection == 0:
                feature_extractors = [calculate_dihedral_angles, symmetric_opposite_angles, symmetric_ratios]
            # Feature Set used in MeshCNN in its non symmetrized form.
            elif feature_selection == 1:
                feature_extractors =[calculate_dihedral_angles, get_ratios, get_opposite_angles]
            # Fundamental form input features [Milano et al. 2020; Barda et al. 2021].
            elif feature_selection == 2:
                feature_extractors = [calculate_dihedral_angles, get_normalized_edge_lengths]
            else:
                raise ValueError('Unknown feature selection: ' + str(feature_selection))

            for extractor in feature_extractors:
                feature = extractor(mesh_data, vertices_of_adjacent_faces)
                features.append(feature)
            return np.concatenate(features, axis=0)

        except Exception as e:
            print(e)
            raise ValueError(mesh_data.filename, 'bad features')


def symmetric_opposite_angles(mesh_data, edge_points):
    """ Computes two angles: one for each face of the edge.
        The angles are sorted to handle order ambiguity.
    """
    angles_a = get_opposite_angles(mesh_data, edge_points, 0)
    angles_a = angles_a.squeeze()

    angles_b = get_opposite_angles(mesh_data, edge_points, 3)
    angles_b = angles_b.squeeze()

    angles = np.concatenate((np.expand_dims(angles_a, 0), np.expand_dims(angles_b, 0)), axis=0)
    angles = np.sort(angles, axis=0)

    return angles


def symmetric_ratios(mesh_data, edge_points):
    """ Computes two ratios: one for each face of the edge
        The ratio is between the height and the base (edge) of each triangle.
        The ratios are sorted to handle order ambiguity.
    """
    ratios_a = get_ratios(mesh_data, edge_points, 0)
    ratios_a = ratios_a.squeeze()

    ratios_b = get_ratios(mesh_data, edge_points, 3)
    ratios_b = ratios_b.squeeze()

    ratios = np.concatenate((np.expand_dims(ratios_a, 0), np.expand_dims(ratios_b, 0)), axis=0)
    return np.sort(ratios, axis=0)


def calculate_dihedral_angles(mesh_data, vertices_of_adjacent_faces):
    normals_a = get_normals(mesh_data.vertex_positions, vertices_of_adjacent_faces, 0)
    normals_b = get_normals(mesh_data.vertex_positions, vertices_of_adjacent_faces, 3)
    dot = np.sum(normals_a * normals_b, axis=1).clip(-1, 1)
    angles = np.expand_dims(np.pi - np.arccos(dot), axis=0)
    return angles


def get_normals(vertex_positions, edge_points, side):
    edge_a = vertex_positions[edge_points[:, side // 2 + 2]] - vertex_positions[edge_points[:, side // 2]]
    edge_b = vertex_positions[edge_points[:, 1 - side // 2]] - vertex_positions[edge_points[:, side // 2]]
    normals = np.cross(edge_a, edge_b)
    div = handle_zero_entries(np.linalg.norm(normals, ord=2, axis=1), epsilon=0.1)
    normals /= div[:, np.newaxis]
    return normals


def get_face_and_opposite_face_vertices(mesh_data):
    edge_points = np.zeros([mesh_data.half_edge_count, 4], dtype=np.int32)
    for half_edge_id, half_edge in enumerate(mesh_data.half_edges):
        edge_points[half_edge_id] = get_face_and_opposite_face_vertices_from_one_half_edge(mesh_data, half_edge_id)
    return edge_points


def get_face_and_opposite_face_vertices_from_one_half_edge(mesh_data, half_edge_id):
    """ Returns the indices of the vertices of the face of the halfedge
        as well as the vertices of the face of the opposite halfedge.
        Four indices in total.
    """
    half_edge = mesh_data.half_edges[half_edge_id]
    next_half_edge_key = mesh_data.half_edge_next[half_edge_id]
    next_half_edge = mesh_data.half_edges[next_half_edge_key]

    opposite_half_edge_key = mesh_data.half_edge_opposite[half_edge_id]
    opposite_next_half_edge_key = mesh_data.half_edge_next[opposite_half_edge_key]
    opposite_next_half_edge = mesh_data.half_edges[opposite_next_half_edge_key]

    return [half_edge[1], half_edge[0], next_half_edge[1], opposite_next_half_edge[1]]


def extract_vertex_to_half_edges_map(faces, vertex_positions):
    vertex_to_half_edges = [[] for _ in vertex_positions]
    half_edge_set = set()
    current_half_edge_index = 0
    for face in faces:
        face_edges = get_edges_from_face(face)
        for edge in face_edges:
            current_half_edge = tuple(edge)
            if current_half_edge not in half_edge_set:
                half_edge_set.add(current_half_edge)
                vertex_to_half_edges[current_half_edge[0]].append(current_half_edge_index)
                vertex_to_half_edges[current_half_edge[1]].append(current_half_edge_index)
                current_half_edge_index += 1
    return vertex_to_half_edges


def extract_half_edge_areas(face_areas, faces):
    half_edge_areas = []
    half_edge_as_unsorted_tuple2key = dict()
    current_half_edge_index = 0
    for faceID, face in enumerate(faces):
        face_edges = get_edges_from_face(face)
        for edge in face_edges:
            current_half_edge = tuple(edge)
            if current_half_edge not in half_edge_as_unsorted_tuple2key:
                half_edge_as_unsorted_tuple2key[current_half_edge] = current_half_edge_index
                half_edge_areas.append(0)
                current_half_edge_index += 1
            current_half_edge_key = half_edge_as_unsorted_tuple2key[current_half_edge]
            half_edge_areas[current_half_edge_key] += face_areas[faceID] / 3
    half_edge_areas = np.array(half_edge_areas, dtype=np.float32) / np.sum(face_areas)
    return half_edge_areas


def extract_edge_areas(face_areas, faces):
    edge_areas = []
    edges_as_sorted_tuple2key = dict()
    current_edge_index = 0
    for faceID, face in enumerate(faces):
        face_edges = get_edges_from_face(face)
        for edge in face_edges:
            current_edge = tuple(sorted(edge))
            if current_edge not in edges_as_sorted_tuple2key:
                edges_as_sorted_tuple2key[current_edge] = current_edge_index
                edge_areas.append(0)
                current_edge_index += 1
            current_edge_key = edges_as_sorted_tuple2key[current_edge]
            edge_areas[current_edge_key] += face_areas[faceID] / 3
    edge_areas = np.array(edge_areas, dtype=np.float32) / np.sum(face_areas)
    return edge_areas


def get_opposite_half_edges(half_edges, half_edge_as_unsorted_tuple2key):
    half_edge_opposite = []
    for i, half_edge in enumerate(half_edges):
        half_edge_opposite.append(half_edge_as_unsorted_tuple2key[tuple([half_edge[1], half_edge[0]])])
    return half_edge_opposite


def check_half_edges(half_edges, half_edge_as_unsorted_tuple2key, filename):
    for i, half_edge in enumerate(half_edges):
        if not tuple([half_edge[0], half_edge[1]]) in half_edge_as_unsorted_tuple2key:
            raise ValueError("Error: Halfedge not found:", filename, tuple([half_edge[0], half_edge[1]]))
        if not tuple([half_edge[1], half_edge[0]]) in half_edge_as_unsorted_tuple2key:
            raise ValueError("Error: Opposite Halfedge not found:", filename, tuple([half_edge[1], half_edge[0]]))


def extract_next_edges(faces):
    half_edge_next = []
    half_edge_as_unsorted_tuple2key = dict()
    current_half_edge_index = 0
    for faceID, face in enumerate(faces):
        face_edges = get_edges_from_face(face)
        for i, half_edge in enumerate(face_edges):
            current_half_edge = tuple(half_edge)
            if current_half_edge not in half_edge_as_unsorted_tuple2key:
                half_edge_as_unsorted_tuple2key[current_half_edge] = current_half_edge_index
                # Build the list with the information which halfedge is the next from the current halfedge.
                if i == 2:
                    # If we deal with the last halfedge of the face, set next of it to the first halfedge of the face.
                    half_edge_next.append(current_half_edge_index - 2)
                else:
                    # Else set it to the halfedge that will be inserted in the lists next.
                    half_edge_next.append(current_half_edge_index + 1)

                current_half_edge_index += 1
    return half_edge_next


def extract_half_edge_as_unsorted_tuple2key(faces):
    half_edge_as_unsorted_tuple2key = dict()
    current_half_edge_index = 0
    for face in faces:
        face_edges = get_edges_from_face(face)
        for edge in face_edges:
            current_half_edge = tuple(edge)
            if current_half_edge not in half_edge_as_unsorted_tuple2key:
                half_edge_as_unsorted_tuple2key[tuple(current_half_edge)] = current_half_edge_index
                current_half_edge_index += 1
    return half_edge_as_unsorted_tuple2key


def post_augmention(mesh_data, opt):
    if hasattr(opt, 'slide_verts') and opt.slide_verts:
        slide_verts(mesh_data, opt.slide_verts)


def slide_verts(mesh_data, percent):
    """ Performce a data augmentation on a mesh by moving vertices on the surface of the mash.
        CAUTION: Although the suggested value used in the original version in scripts/human_seg/train.sh
                 for slide_verts is 0.2, in my experiment only in one out of eight meshes
                 are enough slidable vertices to satisfy this number.
    """
    # Calculate the angles between the two faces adjacent to every halfedge. Therefore get the vertices of both faces.
    face_vertices = get_face_and_opposite_face_vertices(mesh_data)
    dihedral_angles = calculate_dihedral_angles(mesh_data, face_vertices).squeeze()

    # For random selection of vertices, get permuted index list.
    permuted_vertex_indices = np.random.permutation(len(mesh_data.vertex_positions))
    target = int(percent * len(permuted_vertex_indices))
    shifted = 0

    for vertex_index in permuted_vertex_indices:
        if shifted < target:
            # Get all the halfedges connected to the vertex (leading towards or leading away).
            half_edges = mesh_data.vertex_to_half_edges[vertex_index]
            if min(dihedral_angles[half_edges]) > 2.65: # min 152°
                # Randomly select one of the halfedges of the vertex.
                random_halfedge_id = np.random.choice(half_edges)
                random_half_edge = mesh_data.half_edges[random_halfedge_id]

                # Get the index of the other vertex of the halfedge.
                if vertex_index == random_half_edge[0]:
                    index_other_vertex = random_half_edge[1]
                else:
                    index_other_vertex = random_half_edge[0]

                # Shift vertex towards or away from the other vertex of the halfedge.
                vertex_positon = mesh_data.vertex_positions[vertex_index]
                other_vertex_position = mesh_data.vertex_positions[index_other_vertex]
                new_vertex_position = vertex_positon + np.random.uniform(0.2, 0.5) * (other_vertex_position - vertex_positon)
                mesh_data.vertex_positions[vertex_index] = new_vertex_position

                shifted += 1
        else:
            break


def get_ratios(mesh_data, edge_points, side=0):
    edges_lengths = np.linalg.norm(mesh_data.vertex_positions[edge_points[:, side // 2]] - mesh_data.vertex_positions[edge_points[:, 1 - side // 2]], ord=2, axis=1)
    point_o = mesh_data.vertex_positions[edge_points[:, side // 2 + 2]]  # 2 oder 3
    point_a = mesh_data.vertex_positions[edge_points[:, side // 2]]  # 0 oder 1
    point_b = mesh_data.vertex_positions[edge_points[:, 1 - side // 2]]  # 1 oder 0
    line_ab = point_b - point_a
    projection_length = np.sum(line_ab * (point_o - point_a), axis=1) / handle_zero_entries(
        np.linalg.norm(line_ab, ord=2, axis=1), epsilon=0.1)
    closest_point = point_a + (projection_length / edges_lengths)[:, np.newaxis] * line_ab
    d = np.linalg.norm(point_o - closest_point, ord=2, axis=1)

    ratios = d / edges_lengths
    ratios = np.expand_dims(ratios, axis=0)
    return ratios


def get_opposite_angles(mesh_data, edge_points, side=0):
    edges_a = mesh_data.vertex_positions[edge_points[:, side // 2]] - mesh_data.vertex_positions[edge_points[:, side // 2 + 2]]
    edges_b = mesh_data.vertex_positions[edge_points[:, 1 - side // 2]] - mesh_data.vertex_positions[edge_points[:, side // 2 + 2]]

    edges_a /= handle_zero_entries(np.linalg.norm(edges_a, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
    edges_b /= handle_zero_entries(np.linalg.norm(edges_b, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
    dot = np.sum(edges_a * edges_b, axis=1).clip(-1, 1)

    opposite_angles =  np.arccos(dot)
    opposite_angles = np.expand_dims(opposite_angles, axis=0)
    return opposite_angles


def get_normalized_edge_lengths(mesh_data, _):
    return np.expand_dims(mesh_data.half_edge_lengths / np.max(mesh_data.half_edge_lengths), axis=0)


def calculate_edge_lengths(mesh_data):
    vertices_of_adjacent_faces = get_face_and_opposite_face_vertices(mesh_data)

    first_vertex_indices_of_edges  = vertices_of_adjacent_faces[:, 0]
    second_vertex_indices_of_edges = vertices_of_adjacent_faces[:, 1]

    positions_of_first_vertices_of_edges = mesh_data.vertex_positions[first_vertex_indices_of_edges]
    positions_of_second_vertices_of_edges = mesh_data.vertex_positions[second_vertex_indices_of_edges]

    # Calculate the length of every halfedge and store in an array.
    edge_lengths = np.linalg.norm(positions_of_first_vertices_of_edges - positions_of_second_vertices_of_edges, ord=2, axis=1)

    return edge_lengths


def flip_edges(vertex_positions, percent_to_randomly_flip, faces):
    num_edges, key2vertex_and_face_ids_of_edge, edge_as_sorted_tuple2key = get_vertex_and_face_ids_for_all_edges(faces)

    # Call angles_from_faces only with face_id part of key2vertex_and_face_ids_of_edge.
    dihedral_angles = calculate_angle_between_the_two_faces_in_each_row_of_faces(vertex_positions, key2vertex_and_face_ids_of_edge[:, 2:], faces)

    # Gives an array containing numbers from 0 to num_edges in random order.
    edges2flip = np.random.permutation(num_edges)

    number_of_edges_to_flip = int(percent_to_randomly_flip * num_edges)

    flipped = 0
    for edge_key in edges2flip:
        if flipped == number_of_edges_to_flip:
            break
        # Only flip if dihedral_angle > 155°.
        if dihedral_angles[edge_key] > 2.7:
            vertex_and_face_ids_of_edge = key2vertex_and_face_ids_of_edge[edge_key]

            # If edge belongs only to one face, continue.
            if vertex_and_face_ids_of_edge[3] == -1:
                continue

            first_face_id_of_edge  = vertex_and_face_ids_of_edge[2]
            second_face_id_of_edge = vertex_and_face_ids_of_edge[3]

            old_vertex_indices_first_face_of_edge  = faces[first_face_id_of_edge]  # e.g. [177 182 165]
            old_vertex_indices_second_face_of_edge = faces[second_face_id_of_edge] # e.g. [182 177 200]

            vertices_not_in_both_faces = set(old_vertex_indices_first_face_of_edge) ^ set(old_vertex_indices_second_face_of_edge) # e.g. {200, 165}

            new_edge = tuple(sorted(list(vertices_not_in_both_faces))) # e.g. (165, 200)

            # If there is already an edge between the two vertices, continue.
            if new_edge in edge_as_sorted_tuple2key:
                continue

            first_vertex_id_of_edge  = vertex_and_face_ids_of_edge[0]
            second_vertex_id_of_edge = vertex_and_face_ids_of_edge[1]

            new_faces = np.array([[second_vertex_id_of_edge, new_edge[0], new_edge[1]], [first_vertex_id_of_edge, new_edge[0], new_edge[1]]])


            if are_face_areas_bigger_then_zero(vertex_positions, new_faces):
                # remove old edge from dict
                del edge_as_sorted_tuple2key[(first_vertex_id_of_edge, second_vertex_id_of_edge)]

                # Set the vertex_id part of key2vertex_and_face_ids_of_edge for the edge to the new vertex ids.
                vertex_and_face_ids_of_edge[:2] = [new_edge[0], new_edge[1]]

                # Add new edge to dict with the old edge_key.
                edge_as_sorted_tuple2key[new_edge] = edge_key

                first_face_id_of_edge  = vertex_and_face_ids_of_edge[2]
                second_face_id_of_edge = vertex_and_face_ids_of_edge[3]

                old_vertex_indices_first_face_of_edge  = faces[first_face_id_of_edge]
                old_vertex_indices_second_face_of_edge = faces[second_face_id_of_edge]

                new_vertex_indices_first_face_of_edge  = new_faces[0]
                new_vertex_indices_second_face_of_edge =  new_faces[1]

                change_changed_vertex_of_face(old_vertex_indices_first_face_of_edge, new_vertex_indices_first_face_of_edge)
                change_changed_vertex_of_face(old_vertex_indices_second_face_of_edge, new_vertex_indices_second_face_of_edge)

                for i, face_id in enumerate([first_face_id_of_edge, second_face_id_of_edge]):
                    current_face = faces[face_id]
                    for j in range(3):
                        current_edge = tuple(sorted((current_face[j], current_face[(j + 1) % 3])))
                        if current_edge != new_edge:
                            cur_edge_key = edge_as_sorted_tuple2key[current_edge]
                            for idx, face_nb in enumerate(
                                    [key2vertex_and_face_ids_of_edge[cur_edge_key, 2], key2vertex_and_face_ids_of_edge[cur_edge_key, 3]]):
                                if face_nb == vertex_and_face_ids_of_edge[2 + (i + 1) % 2]:
                                    key2vertex_and_face_ids_of_edge[cur_edge_key, 2 + idx] = face_id
                flipped += 1
    return faces


def change_changed_vertex_of_face(old_vertex_indices_of_face, new_vertex_indices_of_face):
    """
    After an Edge flip exactly one vertex in the face is different.
    This vertex is replaced in this function.
    """
    # Get all vertices in new face, that are not in old face (should be exactly one).
    vertices_in_new_face_without_vertices_in_old_face = set(new_vertex_indices_of_face) - set(old_vertex_indices_of_face)
    # Get index from set with one element.
    new_point = list(vertices_in_new_face_without_vertices_in_old_face)[0]
    # Iterate over all three entries. If the one that is different is found, replace with new entry.
    for i in range(3):
        if old_vertex_indices_of_face[i] not in new_vertex_indices_of_face:
            old_vertex_indices_of_face[i] = new_point
            break


def are_face_areas_bigger_then_zero(vertex_positions, faces):
    face_normals = np.cross(vertex_positions[faces[:, 1]] - vertex_positions[faces[:, 0]],
                            vertex_positions[faces[:, 2]] - vertex_positions[faces[:, 1]])
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    face_areas *= 0.5
    return face_areas[0] > 0 and face_areas[1] > 0


def calculate_angle_between_the_two_faces_in_each_row_of_faces(vertex_positions, key2face_ids_of_edge, faces):
    face_normals = [None, None]
    for i in range(2):
        ith_collumn_of_indices = key2face_ids_of_edge[:, i]

        first_vertex_indices_of_faces  = faces[ith_collumn_of_indices, 0]
        second_vertex_indices_of_faces = faces[ith_collumn_of_indices, 1]
        third_vertex_indices_of_faces  = faces[ith_collumn_of_indices, 2]

        positions_first_vertex_indices_of_faces  = vertex_positions[first_vertex_indices_of_faces]
        positions_second_vertex_indices_of_faces = vertex_positions[second_vertex_indices_of_faces]
        positions_third_vertex_indices_of_faces  = vertex_positions[third_vertex_indices_of_faces]


        directions_from_second_vertices_to_third_vertices = positions_third_vertex_indices_of_faces  - positions_second_vertex_indices_of_faces
        directions_from_first_vertices_to_second_vertices = positions_second_vertex_indices_of_faces - positions_first_vertex_indices_of_faces

        face_normals[i] = np.cross(directions_from_second_vertices_to_third_vertices, directions_from_first_vertices_to_second_vertices)

        # Normalize face_normals by dividing through the norms.
        face_normal_norms = np.linalg.norm(face_normals[i], ord=2, axis=1)
        face_normal_norms_without_zeros = handle_zero_entries(face_normal_norms, epsilon=0)
        face_normal_norms_without_zeros_increased_dimension = face_normal_norms_without_zeros[:, np.newaxis] # [1,2,3] -> [[1][2][3]]
        face_normals[i] /= face_normal_norms_without_zeros_increased_dimension

    # Calculate angle between normals.
    dot = np.sum(face_normals[0] * face_normals[1], axis=1).clip(-1, 1)
    angles = np.pi - np.arccos(dot)
    return angles


def handle_zero_entries(to_div, epsilon):
    if epsilon == 0:
        # Replace all entries that are zero with 0.1.
        to_div[to_div == 0] = 0.1
    else:
        to_div += epsilon
    return to_div
