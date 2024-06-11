import numpy as np
from scipy.spatial import Delaunay, ConvexHull


def get_angle_between_vectors(v1, v2):
    # v1: R^3 vector
    # v2: R^3 vector
    # angle: angle between v1 and v2
    # norm_vector: normalized vector orthogonal to v1 and v2
    norm_vector = np.cross(v1, v2)
    norm_vector = norm_vector / np.linalg.norm(norm_vector)
    angle = np.arccos(np.inner(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return angle, norm_vector


def get_2d_line_intersection(l1, l2):
    """
    Compute the intersection of two lines in 2D
    :param l1: [n_x, n_y, d] format where n is the normal vector and |d| is the distance to the origin
        in other words, the line is defined as n_x*x + n_y*y + d = 0
    :param l2: [n_x, n_y, d] format where n is the normal vector and d is the distance to the origin
    :return:
    """
    p_hom = np.cross(l1, l2)
    p = p_hom / p_hom[2]
    return p


def compute_line_from_points(p1, p2):
    """
    Compute the line in 2D from 2 points in the line
    :param p1:
    :param p2:
    :return: line in the form [n_x, n_y, d] where n is the normal vector and d is the distance to the origin
    """
    # compute line 1 form points
    v = p2 - p1
    n = np.array([-v[1], v[0]]) # normal vector
    n = n / np.linalg.norm(n)
    d = -np.inner(n, p1)
    l = np.concatenate([n, [d]])
    return l


def get_2d_line_intersection_from_points(p1_1, p1_2, p2_1, p2_2):
    """
    Compute the intersection of two lines in 2D from 2 points in each of the lines
    :param p1_1: point in line 1 as (x, y)
    :param p1_2: point in line 1 as (x, y)
    :param p2_1: point in line 2 as (x, y)
    :param p2_2: point in line 2 as (x, y)
    :return:
    """
    # compute line 1 form points
    l1 = compute_line_from_points(p1_1, p1_2)
    l2 = compute_line_from_points(p2_1, p2_2)
    p = get_2d_line_intersection(l1, l2)
    return p


def compute_polygon_centroid(polygon_vertices):
    """
    Given a set of 2D vertices cooridinates, compute the centroid of the polygon
    :param polygon_vertices: np array of shape (K, 2) with the coordinates of the polygon vertices
    :return: (2) centroid of the polygon
    """
    # compute the centroid of the polygon
    centroid = np.mean(polygon_vertices, axis=0)
    return centroid


def generate_regular_polygon(num_vertices, radius):
    """
    Generate a regular polygon
    :param num_vertices: <int> number of vertices
    :param num_points: <int> number of points in the polygon
    :param radius: <float> radius of the polygon
    :return:
    """
    angles = np.linspace(0, np.pi * 2, num_vertices, endpoint=False)
    xys = radius * np.stack([np.sin(angles), np.cos(angles)], axis=-1)
    return xys


def generate_regular_star_polygon(num_vertices, radius, inner_radius):
    angles = np.linspace(0, np.pi*2, num_vertices, endpoint=False)
    xys = radius * np.stack([np.sin(angles), np.cos(angles)], axis=-1)
    angles = np.pi/num_vertices + np.linspace(0, np.pi*2, num_vertices, endpoint=False)
    xys_inner = inner_radius * np.stack([np.sin(angles), np.cos(angles)], axis=-1)
    vertices = np.stack([xys, xys_inner], axis=1).reshape(-1,2)
    return vertices


def generate_irregular_star_polygon(num_vertices, out_radius, inner_radius, radius_perturbation):
    angles = np.linspace(0, np.pi * 2, num_vertices, endpoint=False)
    big_radius = out_radius + radius_perturbation * np.random.rand(num_vertices)
    xys =  np.stack([big_radius *np.sin(angles), big_radius *np.cos(angles)], axis=-1)
    angles = np.pi / num_vertices + np.linspace(0, np.pi * 2, num_vertices, endpoint=False)
    small_radius = inner_radius + radius_perturbation * np.random.rand(num_vertices)
    xys_inner = inner_radius * np.stack([small_radius*np.sin(angles), small_radius*np.cos(angles)], axis=-1)
    vertices = np.stack([xys, xys_inner], axis=1).reshape(-1, 2)
    return vertices


def generate_random_polygon(num_vertices, max_radius, min_radius=0):
    """
    Generate a random polygon
    :param num_vertices: <int> number of vertices
    :param max_radius: <float> maximum radius of the polygon
    :param min_radius: <float> minimum radius of the polygon
    :return:
    """
    angles_raw = np.random.rand(num_vertices + 1) * np.pi * 2
    angles = 2 * np.pi * (np.cumsum(angles_raw) / np.sum(angles_raw))[:-1]
    radius = np.random.rand(num_vertices) * (max_radius - min_radius) + min_radius
    xys = np.stack([radius * np.sin(angles), radius * np.cos(angles)], axis=-1)
    return xys


def sample_polygon(polygon_vertices, num_points):
    """
    Given the vertices of a polygon, Sample points from a polygon
    :param polygon_vertices: numpy array of shape (num_vertices, 2)
    :param num_points: <int> number of points to sample
    :return:
    """
    num_vertices = polygon_vertices.shape[0]
    num_points_per_edge = num_points // num_vertices
    points = []
    for i in range(num_vertices):
        p1 = polygon_vertices[i]
        p2 = polygon_vertices[(i + 1) % num_vertices]
        points.extend(np.linspace(p1, p2, num_points_per_edge, endpoint=False))
    return np.array(points)


def polygon_triangulation(polygon_vertices, filter_out_external_triangles=False):
    """
    Triangulate a polygon
    :param polygon_vertices: numpy array of shape (num_vertices, 2)
    :return:
    """
    tri = Delaunay(polygon_vertices)
    tri_vertices = polygon_vertices[tri.simplices]  # (num_triangles, 3, 2)
    if filter_out_external_triangles:
        # remove triangles with edges outside the polygon
        # get the convex hull of the polygon
        hull = ConvexHull(polygon_vertices)
        # get all the edges of the triangles:
        tri_edges_roundtrip = np.stack([tri.simplices[:, [0, 1]], tri.simplices[:, [1, 2]], tri.simplices[:, [2, 0]]], axis=1)

        # get hull edges
        hull_edges = np.stack([hull.vertices, np.roll(hull.vertices, -1)], axis=-1)  # (M, 2)

        # get edges of the polygon
        polygon_edges_roundtrip = np.stack(
            [np.arange(polygon_vertices.shape[0]), np.roll(np.arange(polygon_vertices.shape[0]), -1)], axis=-1)  # (N,2)

        # Get the hull edges that are not actual vertices of the polygon
        outside_edges = []
        for hull_edge in hull_edges:
            is_hull_edge_in_polygon_edges = np.any(np.all(np.isin(polygon_edges_roundtrip, hull_edge), axis=-1))
            if not is_hull_edge_in_polygon_edges:
                outside_edges.append(hull_edge)
        if len(outside_edges) > 0:
            outside_edges = np.stack(outside_edges, axis=0)

            # eges of the polygon (star)
            # tri_edges_roundtrip = np.concatenate([tri.simplices, tri.simplices[...,0:1]], axis=-1)
            # # select the vertices from the polygon edges
            tri_vtxs_roundtrip = polygon_vertices[tri_edges_roundtrip]
            # filter out the triangles with edges outside the polygon, i.e.
            good_triangle_edges = []
            for tri_edges in tri_edges_roundtrip:
                is_good = True
                for outside_edge in outside_edges:
                    is_outside_edge_in_triangle = np.any(np.all(np.isin(tri_edges, outside_edge), axis=-1))
                    if is_outside_edge_in_triangle:
                        is_good = False
                        break
                if is_good:
                    good_triangle_edges.append(tri_edges)
            good_triangle_edges = np.stack(good_triangle_edges, axis=0)  # (num_triangles, 3, 2)

            all_tri_vertices = polygon_vertices[good_triangle_edges] # (num_triangles, 3, 2, 2)
            tri_vertices = all_tri_vertices[...,0,:] # (num_triangles, 3, 2)

    return tri_vertices


def sample_points_inside_polygon(polygon_vertices, num_points):
    """
    Sample points inside a polygon
    :param polygon_vertices: numpy array of shape (num_vertices, 2)
    :param num_points: <int> number of points to sample
    :return:
    """
    tri_vertices = polygon_triangulation(polygon_vertices, filter_out_external_triangles=True) # (num_triangles, 3, 2)
    tri_areas = 0.5 * np.abs(np.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0]))
    tri_probs = tri_areas / np.sum(tri_areas)
    tri_indices = np.random.choice(len(tri_probs), num_points, p=tri_probs) # (num_points)

    # sample points inside the triangles
    r1s = np.random.rand(num_points)
    r2s = np.random.rand(num_points)
    r1s = np.sqrt(r1s)
    # extend the dimensions of r1s and r2s to be (num_points, 2)
    r1s = np.repeat(np.expand_dims(r1s, axis=-1), 2, axis=-1) # (num_points, 2)
    r2s = np.repeat(np.expand_dims(r2s, axis=-1), 2, axis=-1) # (num_points, 2)

    points = (1 - r1s) * tri_vertices[tri_indices, 0] + r1s * (1 - r2s) * tri_vertices[tri_indices, 1] + r1s * r2s * tri_vertices[tri_indices, 2]
    return points


def generate_prism(xy_vertices, num_points, num_points_z, z_length, cover_top=True, cover_bottom=True,
                   num_cover_points=100):
    """
    Generate a prism from a set of xy points
    :param xy_points: numpy array of shape (num_points, 2)
    :param num_points_z: <int> number of points in the z direction
    :param z_length: <float> length of the prism in the z direction
    :return:
    """
    num_vertices = xy_vertices.shape[0]
    xy_points = sample_polygon(xy_vertices, num_points)
    num_points_xy = xy_points.shape[0]
    zs = np.linspace(-z_length * 0.5, z_length * 0.5, num_points_z)
    xyzs = np.repeat(xy_points, num_points_z, axis=0)  # (num_points_xy*num_points_z, 2)
    xyzs = np.concatenate([xyzs, np.tile(zs, num_points_xy).reshape(-1, 1)], axis=-1)  # (num_points_xy*num_points_z, 3)

    if cover_top or cover_bottom:
        # sample points on the top and bottom
        cover_points = sample_points_inside_polygon(xy_vertices, num_cover_points)
        if cover_bottom:
            bottom = np.concatenate([cover_points, np.array([[-z_length * 0.5]] * num_cover_points)], axis=-1)
            xyzs = np.concatenate([xyzs, bottom], axis=0)
        if cover_top:
            top = np.concatenate([cover_points, np.array([[z_length * 0.5]] * num_cover_points)], axis=-1)
            xyzs = np.concatenate([xyzs, top], axis=0)
    return xyzs


def generate_prism_edges_only(xy_vertices, z_length, num_points_per_edge=10):
    """
    Generate a prism with only points along the edges.
    :param xy_vertices:
    :param z_length:
    :param num_points_per_edge: <int> number of points along the edges (this does not count the vertices. i.e. if num_points_per_edge=10, there will be 12 points in the edge)
            i.e. if num_points_per_edge=0, there will only be points on the vertices.
    :return:
    """
    num_vertices = xy_vertices.shape[0]
    xyzs = generate_prism(xy_vertices, num_points=(num_points_per_edge+2)*num_vertices, num_points_z=num_points_per_edge+2, z_length=z_length, cover_top=False, cover_bottom=False,
                   num_cover_points=0)
    return xyzs