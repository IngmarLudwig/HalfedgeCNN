class HalfEdgeNeighborhood():
    def __init__(self, mesh, half_edge_id):
        self.mesh = mesh

        # Indices of the half-edges of the face of self.
        self.S_id = half_edge_id # self
        self.N_id = mesh.half_edge_next[self.S_id] # next
        self.P_id = mesh.half_edge_next[self.N_id] # previous

        # Indices of the halfedges of the face of next_opposite (top right).
        self.NO_id = mesh.half_edge_opposite[self.N_id]
        self.NON_id = mesh.half_edge_next[self.NO_id]
        self.NOP_id = mesh.half_edge_next[self.NON_id]

        # Indices of the halfedges of the face of previous_opposite (top left).
        self.PO_id = mesh.half_edge_opposite[self.P_id]
        self.PON_id = mesh.half_edge_next[self.PO_id]
        self.POP_id = mesh.half_edge_next[self.PON_id]

        # Indices of the halfedges of the face of opposite (bottom).
        self.O_id = mesh.half_edge_opposite[self.S_id] # opposite

        # Indices of the halfedges of the face of next_opposite_next (top right of top right). For 1:3 triangle collapse.
        self.NONO_id = mesh.half_edge_opposite[self.NON_id]

        # Indices of the halfedges of the face of previous_opposite_previous (top left of top left). For 1:3 triangle collapse.
        self.POPO_id = mesh.half_edge_opposite[self.POP_id]


    def violates_link_condition(self):
        """ The link condition is violated, if the next opposite previous halfedge and the previous opposite next halfedge
            are identical. This means that the top left face and the top right face are adjacent, giving the opposite
            vertex of half_edge_id a valence of 3 and "trap" the face of half_edge_id inside the two faces of
            next opposite previous halfedge and the previous opposite next halfedge.
        """

        # Get vertex ids of the two halfedges that are identical in the case of a violated link-condition.
        vertex_ids_a = self.mesh.half_edges[self.NOP_id]
        vertex_ids_b = self.mesh.half_edges[self.PON_id]
        # Check if vertex ids and vertex order are the same. In this case, it is a shared halfedge and the
        # link-condition is violated.
        if set(vertex_ids_a) == set(vertex_ids_b):
            return True
        return False


    def perform_1_3_triangle_collapse(self, half_edge_groups, half_edge_mask):
        # In the case, that the link condition is violated we remove the "entrapped" face by redirecting
        # the halfedges around the face (or, as you might also say, around the valence 3 vertex).
        self.mesh.redirect_half_edge(self.S_id, self.NON_id)
        self.mesh.redirect_half_edge(self.NON_id, self.POP_id)
        self.mesh.redirect_half_edge(self.POP_id, self.S_id)

        # As a result of the redirection of the half edges above, the half edges that lay inside the new formed face
        # (p, n, po, no, pon and nop) are no longer in the mesh. To not lose the features of these half edges,
        # the values are averaged into the remaining half edges (s, o, non, nono, pop, popo).

        # The Construct consist of three triangles that are merged into one that surrounds them all.
        # To keep more information, the features are averaged not equally for all half edges but in three parts.

        # First, the features of the halfedges of the lower triangle are averaged into self and opposite.
        inner_half_edges_lower_triangle = [self.N_id, self.NO_id, self.P_id, self.PO_id]
        self.__union_source_half_edges_into_target_half_edge(half_edge_groups, inner_half_edges_lower_triangle + [self.O_id], self.S_id)
        self.__union_source_half_edges_into_target_half_edge(half_edge_groups, inner_half_edges_lower_triangle + [self.S_id], self.O_id)

        # Then the features of the right triangle are averaged into non and nono.
        inner_half_edges_right_triangle = [self.N_id, self.NO_id, self.PON_id, self.NOP_id, ]
        self.__union_source_half_edges_into_target_half_edge(half_edge_groups, inner_half_edges_right_triangle + [self.NONO_id], self.NON_id)
        self.__union_source_half_edges_into_target_half_edge(half_edge_groups, inner_half_edges_right_triangle + [self.NON_id], self.NONO_id)

        # Finally the features of the left triangle are averaged into pop and popo.
        inner_half_edges_left_triangle = [self.P_id, self.PO_id, self.NOP_id, self.PON_id]
        self.__union_source_half_edges_into_target_half_edge(half_edge_groups, inner_half_edges_left_triangle + [self.POPO_id], self.POP_id)
        self.__union_source_half_edges_into_target_half_edge(half_edge_groups, inner_half_edges_left_triangle + [self.POP_id], self.POPO_id)

        # Remove all the half edges that were inside the surrounding triangle and that are therefore not connected anymore.
        inner_half_edges = [self.N_id, self.NO_id, self.P_id, self.PO_id, self.NOP_id, self.PON_id]
        for half_edge_key in inner_half_edges:
            assert half_edge_mask[half_edge_key]
            half_edge_mask[half_edge_key] = False
            self.mesh.remove_half_edge(half_edge_key)

        # Remove the central vertex inside the surrounding triangle (the one with valence 3).
        # The vertex is always the one "next" is pointing to.
        central_vertex = self.mesh.half_edges[self.N_id][1]
        self.mesh.remove_vertex(central_vertex)


    def pool(self, half_edge_mask, half_edge_groups):
        assert half_edge_mask[self.P_id]
        assert half_edge_mask[self.PO_id]

        # Redirect former top right he of the face of Self to the top left face to perform the collapse of this neighborhood.
        self.mesh.redirect_half_edge(self.N_id, self.PON_id)
        self.mesh.redirect_half_edge(self.POP_id, self.N_id)

        # Avarage features of
        if self.mesh.pooling == 'edge_pooling':
            # Avarage all half-edges of this side and Opposite into Next.
            half_edge_groups.union(self.S_id,  self.N_id)
            half_edge_groups.union(self.O_id,  self.N_id)
            half_edge_groups.union(self.P_id,  self.N_id)
            half_edge_groups.union(self.NO_id, self.N_id)
            half_edge_groups.union(self.PO_id, self.N_id)

            # Avarage all half-edges of this side and Opposite into Next-Opposite.
            half_edge_groups.union(self.S_id,  self.NO_id)
            half_edge_groups.union(self.O_id,  self.NO_id)
            half_edge_groups.union(self.N_id,  self.NO_id)
            half_edge_groups.union(self.P_id,  self.NO_id)
            half_edge_groups.union(self.PO_id, self.NO_id)

        elif self.mesh.pooling == 'half_edge_pooling':
            # Avarage Self and Previous-Opposite into Next.
            half_edge_groups.union(self.S_id,  self.N_id)
            half_edge_groups.union(self.PO_id, self.N_id)

            # Avarage Self and Previous into Next-Opposite.
            half_edge_groups.union(self.S_id, self.NO_id)
            half_edge_groups.union(self.P_id, self.NO_id)
        else:
            raise ValueError(self.mesh.pooling.filename, 'is not a supported pooling procedure.')

        # The division through the number of added up features for averaging is done in rebuild_features_average in mesh_union.py.

        # Remove Previous and Previous-Opposite as removed. Self and Self-Opposite removed in __pool_neighborhoods of HalfEdgeMeshPool.
        half_edge_mask[self.P_id] = False
        half_edge_mask[self.PO_id] = False
        self.mesh.remove_half_edge(self.P_id)
        self.mesh.remove_half_edge(self.PO_id)


    def __union_source_half_edges_into_target_half_edge(self, group, source_half_edges, target_half_edge):
        for source_half_edge in source_half_edges:
            group.union(source_half_edge, target_half_edge)
