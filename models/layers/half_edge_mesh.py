import os
import numpy as np
from models.layers import half_edge_mesh_prepare
from models.layers.output_data_interface_layer import EdgeBasedDataOutputInterfaceLayer, HalfEdgeBasedDataOutputInterfaceLayer, FaceBasedDataOutputInterfaceLayer


class HalfEdgeMesh:

    def __init__(self, file=None, opt=None, hold_history=False, export_folder=''):
        # Default init member variables.

        # Public member variables (for read only access). Please do not change outside this file!
        self.edge_areas = None
        self.half_edge_areas = None  # Relative part of the surface that is represented by the individual halfedge (diveded by the total surface area, therefor relative.
        self.half_edge_count = 0  # Total number of halfedges.
        self.half_edge_features = None  # Features of the individual halfedge ( see extract_features of half_edge_mesh_prepare for details)
        self.half_edge_neighborhoods = None  # Lists the halfedges by id in the neighborhood of the individual halfedge. Depends on setting. See build_half_edge_neighborhood of half_edge_mesh_prepare for details.
        self.half_edge_next = None  # The next halfedge in the face of the individual halfedges.
        self.half_edge_opposite = None  # The opposite halfedge of the individual halfedges.
        self.edges = None  # The vertices of the individual edges. Result of ordering the half edges before entry.
        self.half_edges = None  # The vertices of the individual halfedges (ordered, first the start vertex, then the end vertex). By vertex index
        self.vertex_to_half_edges = None  # Lists all halfedges that lead to or from the given vertex index.

        # Private member variables.
        self.__filename = None  # Filename of the original .obj file
        self.__nbh_size = 0  # Size of the neighborhood in halfedges.
        self.__pool_count = 0  # How often has this mesh been pooled (reduced)?
        self.__vertex_mask = None  # Boolean array with length len(__vertex_positions). Stores whether a vertex was deleted. If true, vertex with that index in __vertex_positions was not deleted.
        self.__vertex_positions = None  # Positions of the vertices in the form [[x1][y1][z1],[x2][y2][z2],...]

        self.__segmentation_base = opt.segmentation_base
        self.pooling = opt.pooling
        self.__nbh_size = opt.nbh_size
        self.__hold_history = hold_history
        self.__export_folder = export_folder # Folder to export resulting Meshes (e.g. after pooling).

        # Read mesh from cache or file and create augmented version of the mesh if selected. Done in a lazy evaluation fashion.
        mesh_data = half_edge_mesh_prepare.get_mesh_data(file, opt)
        self.__load_mesh_data(mesh_data)

        self.__vertex_mask = np.ones(len(self.__vertex_positions), dtype=bool)

        self.__update_half_edge_neighborhood()

        if hold_history:
            self.__groups_history = [] # All the features that contributed to the current features in this step.
            self.__half_edge_neighborhood_history = [self.half_edge_neighborhoods.copy()] # The halfedges that were in the neighborhood of each halfedge in this step.
            self.__occurrences_history = [] # How many features were added up in the features tensor in the last step.
            self.__half_edge_mask_history = [] # A list of half_edge_masks, with True if halfedge was still not pooled away in this step. If __export_folder == '', only the first entry is used.
            self.__half_edge_count_history = [self.half_edge_count] # How many halfedges existed in this step

        self.output_data_interface_layer = None
        # In the classification case, the output_data_interface_layer should be the simplest which is the face based one.
        if opt.dataset_mode == "classification":
            self.output_data_interface_layer = FaceBasedDataOutputInterfaceLayer()
        elif self.__segmentation_base == "edge_based":
            self.output_data_interface_layer = EdgeBasedDataOutputInterfaceLayer()
        elif self.__segmentation_base == "halfedge_based":
            self.output_data_interface_layer = HalfEdgeBasedDataOutputInterfaceLayer()
        elif self.__segmentation_base == "face_based":
            self.output_data_interface_layer = FaceBasedDataOutputInterfaceLayer()
        else:
            raise Exception("Unknown segmentation base: %s" % self.__segmentation_base)

        self.__export()

        # Only these Poolings are implemented.
        assert (self.pooling == 'edge_pooling' or self.pooling == 'half_edge_pooling')

        # Only certain neighborhoods are implemented.
        assert(   self.__nbh_size == 2
               or self.__nbh_size == 3
               or self.__nbh_size == 4
               or self.__nbh_size == 5
               or self.__nbh_size == 7
               or self.__nbh_size == 9)

    def __load_mesh_data(self, mesh_data):
        self.__filename = str(mesh_data['filename'])

        self.__vertex_positions = mesh_data['vertex_positions']
        self.half_edges = mesh_data['half_edges']
        self.edges = mesh_data['edges']
        self.faces = mesh_data['faces']

        self.half_edge_next = mesh_data['half_edge_next']
        self.half_edge_opposite = mesh_data['half_edge_opposite']

        self.half_edge_count = int(mesh_data['half_edge_count'])

        self.half_edge_areas = mesh_data['half_edge_areas']
        self.edge_areas = mesh_data['edge_areas']
        self.face_areas = mesh_data['face_areas']

        self.vertex_to_half_edges = mesh_data['vertex_to_half_edges']
        self.edge_index_to_halfedge_indices = mesh_data['edge_index_to_halfedge_indices']
        self.face_index_to_halfedge_indices = mesh_data['face_index_to_halfedge_indices']

        self.half_edge_features = mesh_data['half_edge_features']

    def redirect_half_edge(self, half_edge_to_redirect_id, target_half_edge_id):
        """ Redirects half-edges in the half-edge structure. Used to realize collapses."""
        self.half_edge_next[half_edge_to_redirect_id] = target_half_edge_id

    def remove_half_edge(self, half_edge_id):
        self.__remove_half_edge_from_vertex_to_half_edges(half_edge_id)
        self.half_edge_count -= 1

    def remove_vertex(self, v):
        self.__vertex_mask[v] = False

    def merge_vertices_of_halfedge(self, half_edge_id):
        vertex_ids_of_half_edge = self.half_edges[half_edge_id]
        first_vertex_of_halfedge_id  = vertex_ids_of_half_edge[0]
        second_vertex_of_halfedge_id = vertex_ids_of_half_edge[1]
        self.__merge_vertices(first_vertex_of_halfedge_id, second_vertex_of_halfedge_id)

    def correct_data_after_pooling(self, half_edge_mask, groups):
        """ This method cleans the data structures after performing the pooling operations. """
        self.__update_half_edge_neighborhood()
        self.__remove_deleted_halfedges_from_data(half_edge_mask)
        # After the removal of deleted halfedges several entries are missing in the data somewhere in between.
        # Therefore, the indices given in the lists are not correct anymore. We therefore need to correct the indices.
        self.__correct_indices_in_data(half_edge_mask)
        self.__update_history_data(groups, half_edge_mask)
        self.__pool_count += 1
        self.__export()

    def get_last_occurrences_from_history(self):
        return self.__occurrences_history[-1]

    def get_last_groups_from_history(self):
        return self.__groups_history[-1]

    def go_back_one_step_in_history(self):
        self.__half_edge_neighborhood_history.pop()
        self.half_edge_neighborhoods = self.__half_edge_neighborhood_history[-1]

        self.__half_edge_count_history.pop()
        self.half_edge_count = self.__half_edge_count_history[-1]

        self.__occurrences_history.pop()
        self.__groups_history.pop()

    def __get_faces(self):
        faces = set()
        for half_edge_index, vertex_ids_of_half_edge in enumerate(self.half_edges):
            next_half_edge_index = self.half_edge_next[half_edge_index]

            v_id_1 = vertex_ids_of_half_edge[0]
            v_id_2 = vertex_ids_of_half_edge[1]
            v_id_3 = self.half_edges[next_half_edge_index][1]

            face = (v_id_1, v_id_2, v_id_3)

            if not (v_id_1, v_id_2, v_id_3) in faces and not (v_id_2, v_id_3, v_id_1) in faces and not (v_id_3, v_id_1, v_id_2) in faces:
                faces.add(face)

        return faces


    def get_old2new_vertex_indices(self):
        old2new_vertex_indices = np.zeros(self.__vertex_mask.shape[0], dtype=np.int32)
        old2new_vertex_indices[self.__vertex_mask] = np.arange(0, np.ma.where(self.__vertex_mask)[0].shape[0])
        return old2new_vertex_indices


    def __export(self):
        if not self.__export_folder:
            return

        filename, file_extension = os.path.splitext(self.__filename)
        file = '%s/%s_%d%s' % (self.__export_folder, filename, self.__pool_count, file_extension)

        vertex_positions_active = self.__vertex_positions[self.__vertex_mask]
        old2new_vertex_indices = self.get_old2new_vertex_indices()

        # If the mesh is not pooled, we use the original faces (which have not been altered by the pooling), because in this case the segmentation is added to the file and
        # the segmentation needs the faces to be in the original form.
        faces = None
        if self.__pool_count == 0:
            faces = self.faces
        else:
            faces = self.__get_faces()

        self.output_data_interface_layer.create_obj_file(  file=file,
                                                           vertex_positions=vertex_positions_active,
                                                           faces=faces,
                                                           half_edges=self.half_edges,
                                                           old2new_vertex_indices=old2new_vertex_indices,
                                                           vcolor=None )

    def export_segmentation_of_mesh(self, segments):
        """ Exports the segmentation of each edge for every pooling step."""
        if not self.__export_folder:
            return

        filename, file_extension = os.path.splitext(self.__filename)
        # Export segmentations only for unpooled meshes. These have _0 in their name, giving the pooling level 0.
        file = '%s/%s_0%s' % (self.__export_folder, filename, file_extension)

        self.output_data_interface_layer.export_segmentation(file=file,
                                                             half_edges=self.half_edges,
                                                             edges=self.edges,
                                                             faces=self.faces,
                                                             segments = segments)

    def __update_half_edge_neighborhood(self):
        """ Based on neighborhood size as described in the paper in section 7, Table at the end and section 3.1."""
        half_edge_neighborhoods = np.full((len(self.half_edges), self.__nbh_size), -1, dtype=np.int64)
        for half_edge_id in range(len(self.half_edges)):
            # Indices of the half-edges of the face of self.
            N_id = self.half_edge_next[half_edge_id]  # next
            P_id = self.half_edge_next[N_id]  # previous

            # Indices of the halfedges of the face of next_opposite (top right).
            NO_id = self.half_edge_opposite[N_id]

            # Indices of the halfedges of the face of previous_opposite (top left).
            PO_id = self.half_edge_opposite[P_id]

            # Indices of the halfedges of the face of opposite (bottom).
            O_id = self.half_edge_opposite[half_edge_id]  # opposite
            ON_id = self.half_edge_next[O_id]
            OP_id = self.half_edge_next[ON_id]

            # Indices of the halfedges of the face of opposite_next_opposite (bottom left).
            ONO_id = self.half_edge_opposite[ON_id]

            # Indices of the halfedges of the face of opposite_previous_opposite (bottom right).
            OPO_id = self.half_edge_opposite[OP_id]

            # 4 is the Milano-neighborhood.
            if self.__nbh_size == 4:
                half_edge_neighborhoods[half_edge_id][0] = NO_id
                half_edge_neighborhoods[half_edge_id][1] = PO_id
                half_edge_neighborhoods[half_edge_id][2] = ON_id
                half_edge_neighborhoods[half_edge_id][3] = OP_id
            else:
                # self.nbh_size >= 2:
                half_edge_neighborhoods[half_edge_id][0] = O_id
                half_edge_neighborhoods[half_edge_id][1] = N_id
                if self.__nbh_size >= 3:
                    half_edge_neighborhoods[half_edge_id][2] = P_id
                if self.__nbh_size >= 5:
                    half_edge_neighborhoods[half_edge_id][3] = ON_id
                    half_edge_neighborhoods[half_edge_id][4] = OP_id
                if self.__nbh_size >= 7:
                    half_edge_neighborhoods[half_edge_id][5] = NO_id
                    half_edge_neighborhoods[half_edge_id][6] = PO_id
                if self.__nbh_size >= 9:
                    half_edge_neighborhoods[half_edge_id][7] = ONO_id
                    half_edge_neighborhoods[half_edge_id][8] = OPO_id

        self.half_edge_neighborhoods = half_edge_neighborhoods


    def __remove_half_edge_from_vertex_to_half_edges(self, half_edge_id):
        half_edge = self.half_edges[half_edge_id]

        for vertex in half_edge:
            self.vertex_to_half_edges[vertex].remove(half_edge_id)


    def __correct_indices_in_data(self, half_edge_mask):
        # First we create a Map from the old to the new indices:
        old2new_indices = np.zeros(half_edge_mask.shape[0], dtype=np.int32)
        old2new_indices[half_edge_mask] = np.arange(self.half_edge_count)
        # Then, in all arrays that contain halfedge indices, the old vertices are exchanged with the new vertices.
        self.half_edge_neighborhoods = old2new_indices[self.half_edge_neighborhoods]
        self.half_edge_next = old2new_indices[self.half_edge_next]
        self.half_edge_opposite = old2new_indices[self.half_edge_opposite]
        # The update of the indices of vertex_to_half_edges is done in a loop since it has an inhomogeneous shape.
        new_vertex_to_half_edges = []
        for v_index, old_half_edges_indices in enumerate(self.vertex_to_half_edges):
            new_half_edges_indices = []

            for old_half_edge_index in old_half_edges_indices:
                new_half_edges_indices.append(old2new_indices[old_half_edge_index])
            new_vertex_to_half_edges.append(new_half_edges_indices)
        self.vertex_to_half_edges = new_vertex_to_half_edges


    def __remove_deleted_halfedges_from_data(self, half_edge_mask):
        # Remove entries of halfedges that have been removed during pooling.
        self.half_edge_neighborhoods = self.half_edge_neighborhoods[half_edge_mask]
        self.half_edges = self.half_edges[half_edge_mask]
        self.half_edge_next = self.half_edge_next[half_edge_mask]
        self.half_edge_opposite = self.half_edge_opposite[half_edge_mask]


    def __merge_vertices(self, vertex_a_id, vertex_b_id):
        # Always remove the vertex with the lower index.
        if vertex_a_id < vertex_b_id:
            remaining_vertex = vertex_a_id
            vertex_to_delete = vertex_b_id
        else:
            remaining_vertex = vertex_b_id
            vertex_to_delete = vertex_a_id
        # Set remaining vertex to the mean of the two vertices that were connected by the halfedges.
        position_remaining_vertex = self.__vertex_positions[remaining_vertex]
        position_vertex_to_delete = self.__vertex_positions[vertex_to_delete]
        position_remaining_vertex += position_vertex_to_delete
        position_remaining_vertex /= 2
        self.__vertex_mask[vertex_to_delete] = False
        halfeges_of_vertex_to_delete = self.vertex_to_half_edges[vertex_to_delete]
        self.vertex_to_half_edges[remaining_vertex].extend(halfeges_of_vertex_to_delete)
        # Replace all occurrences of the old vertex index in half_edges with the index of the remaining vertex.
        half_edge_mask = self.half_edges == vertex_to_delete  # Gives a list like: [[False, True],[False, False],[True, False],[False, False] ] with a True everywhere where index == vertex_to_delete
        self.half_edges[
            half_edge_mask] = remaining_vertex  # Sets every value in half_edges to remaining_vertex, where there is a True in half_edge_mask.


    def __update_history_data(self, groups, half_edge_mask):
        if self.__hold_history:
            if self.__export_folder:
                self.__half_edge_mask_history.append(half_edge_mask.copy())
            # Save how many features were added up in the features tensor in the last step.
            self.__occurrences_history.append(groups.get_occurrences())
            # Save all the features that contributed to the current features in this step.
            self.__groups_history.append(groups.get_groups(half_edge_mask))
            # Save the halfedges that were in the neighborhood of each halfedge in this step.
            self.__half_edge_neighborhood_history.append(self.half_edge_neighborhoods.copy())
            # Save how many halfedges existed in the last step.
            self.__half_edge_count_history.append(self.half_edge_count)