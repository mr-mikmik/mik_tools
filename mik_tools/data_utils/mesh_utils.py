import numpy as np
import trimesh


def discretize_mesh(mesh:trimesh.Trimesh):
    """
    Discretize the mesh into a set of points, normals and As.
     - Points are the center of the faces.
     - Normals are the normals of the faces.
    - As are the areas of the faces.
    :param mesh (trimesh.Trimesh): input mesh
    :return points: (N, 3) numpy array
    :return normals: (N, 3) numpy array
    :return As: (N,) numpy array
    """
    triangles = mesh.triangles # (N, 3, 3) numpy array
    points = triangles.mean(axis=1) # (N, 3) numpy array
    normals = mesh.face_normals # (N, 3) numpy array
    As = mesh.area_faces # (N,) numpy array
    return points, normals, As