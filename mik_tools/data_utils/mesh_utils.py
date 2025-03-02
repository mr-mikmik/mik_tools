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


def get_mesh_inertia(mesh:trimesh.Trimesh, mass:float):
    """
    Compute the inertia tensor of a mesh.
    :param mesh (trimesh.Trimesh): input mesh
    :param mass (float): mass of the object
    :return inertia_tensor: (3, 3) numpy array
    """
    # return mesh.moment_inertia --- This gives some weird results sometimes
    points, normals, As = discretize_mesh(mesh) # (N, 3), (N, 3), (N,)

    # assume that the object surface is uniform so the mass per point is proportional to its area
    points_mass = As / As.sum() * mass # (N,)

    # The inertia tensor is the mass times the sum of the squares of the distances to the center of mass
    center_of_mass = np.sum(points * points_mass[:, None], axis=0)
    points_centered = points - center_of_mass
    # inertia_tensor = np.einsum('ni,nj,n->ij', points_centered, points_centered, points_mass) # (N, 3, 3)
    inertia_tensor = np.einsum('ni,nj,n->ij', points, points, points_mass) # (N, 3, 3)
    return inertia_tensor