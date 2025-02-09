from typing import List

import trimesh
import napari
import numpy as np


def psudo_color(vertices: np.ndarray):
    return (vertices - vertices.min(axis=0)) / (
        vertices.max(axis=0) - vertices.min(axis=0)
    )


def napari_add_point_cloud_video(
    viewer: napari.Viewer,
    pts_time: List[trimesh.PointCloud],
    name: str,
    use_psudo_color=False,
):
    new_pts_time = []
    for t in range(len(pts_time)):
        new_pts_time.append(
            np.concatenate([np.ones_like(pts_time[t][:, 0:1]) * t, pts_time[t]], -1)
        )
    new_pts_time = np.concatenate(new_pts_time, 0)
    if use_psudo_color:
        new_pts_time_color = []

        for t in range(len(pts_time)):
            new_pts_time_color.append(psudo_color(pts_time[t]))
        new_pts_time_color = np.concatenate(new_pts_time_color, 0)
    if use_psudo_color:
        viewer.add_points(
            new_pts_time, name=name, size=1, face_color=new_pts_time_color
        )
    else:
        viewer.add_points(new_pts_time, name=name, size=1)


def napari_add_mesh_video(
    viewer: napari.Viewer,
    meshs_time: List[trimesh.Trimesh],
    name: str,
):

    time_f = []
    time_c = []
    time_v = []
    pvn = 0
    for t, mesh in enumerate(meshs_time):

        time_f += [(mesh.faces + pvn).astype(np.int32)]
        time_v += [
            np.concatenate([np.ones_like(mesh.vertices[:, 0:1]) * t, mesh.vertices], -1)
        ]
        time_c += [psudo_color(mesh.vertices)]
        pvn = pvn + mesh.vertices.shape[0]
    time_v = np.concatenate(time_v, 0).astype(np.float32)
    time_c = np.concatenate(time_c, 0).astype(np.float32)
    time_f = np.concatenate(time_f, 0).astype(np.int32)
    viewer.add_surface(
        (time_v, time_f),
        vertex_colors=time_c,
        name=name,
    )


def viewing_sparse_volume(viewer: napari.Viewer, sv, name="sparse_volume"):
    c = sv.C[:, 1:].cpu().numpy()
    f = sv.F.cpu().numpy()
    f_num = f.shape[1]
    for i in range(f_num):
        viewer.add_points(
            c,
            size=1,
            features={"feat": f[:, i]},
            face_color="feat",
            name=f"{name}_{i}",
        )
        viewer.add_points(
            c,
            size=1,
            features={"feat": f[:, i]},
            face_color="feat",
            name=f"{name}_{i}",
        )
    return viewer


def napari_viewer_add_graph(viewer: napari.Viewer, graph, position, name="graph"):
    lines = []
    for u, v in graph.edges:
        lines.append(np.stack([position[u], position[v]], 0))
    lines = np.stack(lines, 0)
    pts_dist = np.linalg.norm(lines[:, 0] - lines[:, 1], axis=-1).mean()
    viewer.add_points(position, name="graph_points", size=pts_dist * 0.5)
    viewer.add_shapes(
        lines,
        name=name,
        shape_type="line",
        face_color=[0] * 4,
        edge_color="coral",
        edge_width=pts_dist * 0.05,
    )

    return viewer
