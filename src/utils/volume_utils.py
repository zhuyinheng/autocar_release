import numpy as np


def sparse_volume_to_dense_volume(coords: np.ndarray, feats: np.ndarray):
    """
    coords: N x 3, int, numpy array, from sparse tensor
    feats: N x C, float, numpy array, from sparse tensor
    """

    C = feats.shape[1]
    _coords = (
        coords[:, None, :]
        + np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=np.int32,
        )[None, :, :]
    ).reshape(-1, 3)
    _feats = (feats[:, None, :] * np.ones((1, 8, 1))).reshape(-1, C)

    coords_max = _coords.max(0)
    coords_min = _coords.min(0)
    pos_coords = _coords - coords_min
    dense_volume_shape = np.ceil(coords_max - coords_min + 2).astype(np.int32)
    dense_volume = np.zeros(dense_volume_shape.tolist() + [C], dtype=np.float32)
    # FIXME: limited resources: all sparse voxel are sampled in even set to fit 11G GPU memory
    dense_volume[pos_coords[:, 0], pos_coords[:, 1], pos_coords[:, 2], :] = _feats
    return dense_volume, coords_min


def save_volume_to_mesh(fn, v):
    def write_triangle_surface_mesh(fn, v, f, n, check_surface=False):
        import numpy as np
        import trimesh

        mesh = trimesh.Trimesh(vertices=v, faces=f, vertex_normals=n)

        if check_surface:
            # Define a function to calculate triangle areas
            def triangle_areas(mesh):
                triangles = mesh.faces
                vertices = mesh.vertices
                vec_cross = np.cross(
                    vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                    vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
                )
                areas = 0.5 * np.linalg.norm(vec_cross, axis=1)
                return areas

            # Calculate areas of the triangles
            areas = triangle_areas(mesh)

            # Find indices of zero-area triangles
            zero_area_indices = np.where(areas == 0)[0]

            # Remove zero-area triangles
            mesh.update_faces(
                np.setdiff1d(np.arange(len(mesh.faces)), zero_area_indices)
            )
            mesh.remove_unreferenced_vertices()

            # Fix any potential inversion in the mesh
            mesh.fix_normals()

        # Export the cleaned mesh
        mesh.export(fn)

    from skimage.measure import marching_cubes

    v, f, n, _ = marching_cubes(v > 0, 0, allow_degenerate=False)
    write_triangle_surface_mesh(fn, v, f, n, False)
