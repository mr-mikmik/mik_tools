import numpy as np

from mik_tools import tr, pose_to_matrix, matrix_to_pose, transform_matrix_inverse, transform_points_3d
from mik_tools.camera_tools.img_utils import project_depth_image, project_point_to_image
from .rendering_tools import render_view, render_tactile, get_default_intrinsics, create_intrinsic_matrix


class Camera(object):
    def __init__(self, pose, K=None, img_size=(400, 400), render_scale=1, tactile_plane_distance=None, znear=None, zfar=None):
        self.pose = pose # pose of the camera optical frame w.r.t the world frame (i.e. w_pose_cf)
        self.K = K
        self.img_size = img_size # as (h, w) -- CAREFUL! This is flipped.
        self.render_scale = render_scale
        self.znear = znear
        self.zfar = zfar
        self.crop_size = None
        self.crop_center_uv = None
        self.tactile_plane_distance = tactile_plane_distance
        if self.K is None:
            self.K = get_default_intrinsics()

    @property
    def w_X_cf(self):
        return pose_to_matrix(self.pose)

    @w_X_cf.setter
    def w_X_cf(self, value):
        self.pose = matrix_to_pose(value)

    @property
    def K_rendering(self):
        # update cx and cy to match the sub rendering area.
        rend_img_size = self.render_img_size
        rendering_center_uv = self.rendering_center_uv
        cx_original = self.K[0, 2]
        cy_original = self.K[1, 2]
        # to scale the rendering, our formula is:
        # cx_new = img_size_x_new / img_size_x_original * cx_original - cx_original + new_center_x
        # note new_center x in the original image coordiantes, i.e. the corp center
        # also img_size_x_new is the corp image size
        cx_delta = cx_original - rendering_center_uv[0]
        cy_delta = cy_original - rendering_center_uv[1]
        # Below is the theoretical values for the transformation:
        # cx_rend = rend_img_size[0] / self.img_size[0]  * cx_original - cx_delta
        # cy_rend = rend_img_size[1] / self.img_size[1]  * cy_original - cy_delta
        # NOTE: We have to tune the theoretical ones to account that the original cameras as not centered at the (0,0)
        # This is a little hacky, but it turned out to work fine.
        cx_rend = rend_img_size[0] * 0.5 - cx_delta + 2 * (cx_original - self.img_size[0]//2)
        cy_rend = rend_img_size[1] * 0.5 - cy_delta + 2 * (cy_original - self.img_size[1]//2)
        K_rend = self.K.copy() # The focal lengths stay the same
        K_rend[0,2] = cx_rend
        K_rend[1,2] = cy_rend
        return K_rend

    @property
    def render_img_size(self):
        if self.crop_size is None or self.crop_center_uv is None:
            return self.img_size
        rendering_img_size = self.crop_size
        return (rendering_img_size[0], rendering_img_size[1])
        # return (rendering_img_size[1], rendering_img_size[0])

    @property
    def rendering_center_uv(self):
        # return the rendering center coordinates in the original image cooridnates
        if self.crop_size is None or self.crop_center_uv is None:
            # render full image
            rcenter_uv = (self.img_size[0]//2, self.img_size[1]//2)
        else:
            # render only the crop region.
            rcenter_uv = (self.crop_center_uv[0], self.crop_center_uv[1])
        return rcenter_uv

    def get_rendering_limits_uv(self):
        rendering_limits_uv = self.get_cropping_limits_uv()
        return rendering_limits_uv

    def get_cropping_limits_uv(self):
        if self.crop_size is None or self.crop_center_uv is None:
            crop_center_uv = np.array(self.img_size) // 2
            crop_size = self.img_size
        else:
            crop_center_uv = self.crop_center_uv
            crop_size = self.crop_size
        # compute the cropping limits in order (top-left, top-right, bottom-left, bottom-right)
        half_size = np.array(crop_size) // 2
        cropping_limits_uv = np.array([
            [crop_center_uv[0] - half_size[0], crop_center_uv[1] - half_size[1]],
            [crop_center_uv[0] - half_size[0], crop_center_uv[1] + half_size[1]],
            [crop_center_uv[0] + half_size[0], crop_center_uv[1] - half_size[1]],
            [crop_center_uv[0] + half_size[0], crop_center_uv[1] + half_size[1]],
        ]).astype(np.int64)
        # Adjust the cropping limits to be within the image range
        cropping_limits_uv[..., 0] = np.clip(cropping_limits_uv[..., 0], 0, self.img_size[0])
        cropping_limits_uv[..., 1] = np.clip(cropping_limits_uv[..., 1], 0, self.img_size[1])

        return cropping_limits_uv

    def render_tactile(self, object_mesh, w_X_of, blur=False):
        of_X_w = transform_matrix_inverse(w_X_of)
        of_X_cf = of_X_w @ self.w_X_cf
        tactile_depth = render_tactile(object_mesh, of_X_cf, blur=blur,
                                       K=self.K, img_size=self.img_size,
                                       tactile_plane_distance=self.tactile_plane_distance
                                       )
        return tactile_depth

    def render_vision(self, object_mesh, w_X_of, blur=False):
        of_X_w = transform_matrix_inverse(w_X_of)
        of_X_cf = of_X_w @ self.w_X_cf
        _, vision_depth = render_view(object_mesh, of_X_cf, blur=blur, K=self.K, img_size=self.img_size)
        return vision_depth

    def project_depth_image(self, depth_img, world_frame=False):
        img_xyz_cf = project_depth_image(depth_img, self.K)
        if world_frame:
            img_xyz_w = transform_points_3d(img_xyz_cf, self.w_X_cf, points_have_point_dimension=False)
            return img_xyz_w
        return img_xyz_cf

    def project_point_to_camera_frame(self, points_w, points_have_point_dimension=True):
        point_w = points_w[..., :3] # position of the point in the world frame # (..., 3)
        point_cof = transform_points_3d(point_w, transform_matrix_inverse(self.w_X_cf), points_have_point_dimension=points_have_point_dimension) #(..., 3)
        return point_cof

    def project_point_to_img(self, point_w, as_int=False):
        point_w = point_w[..., :3] # position of the point in the world frame
        point_cof = transform_points_3d(point_w, transform_matrix_inverse(self.w_X_cf), points_have_point_dimension=False)
        point_uv = project_point_to_image(point_cof, K=self.K, as_int=as_int)
        return point_uv

    def project_points_to_img(self, points_w, as_int=False):
        point_w = points_w[..., :3] # position of the point in the world frame # (K, 3)
        point_cof = transform_points_3d(point_w, transform_matrix_inverse(self.w_X_cf), points_have_point_dimension=True) #(K, 3)
        point_uv = project_point_to_image(point_cof, K=self.K, as_int=as_int) # (K, 2)
        return point_uv

    def crop_rendering(self, img_rendered):
        # img_rendered: (..., rendered_w, rendered_h)
        # img_out: (..., crop_size_w, crop_size_h)
        # get the cropping limits coordinates in the rendering coordinates
        rendering_lims_uv = self.get_rendering_limits_uv()
        cropping_lims_uv = self.get_cropping_limits_uv()
        copping_lims_in_redering_uvs = (cropping_lims_uv - rendering_lims_uv[0:1]).astype(np.int64) # (4,2)
        # img_out = img_rendered[..., copping_lims_in_redering_uvs[0,0]:copping_lims_in_redering_uvs[3,0], copping_lims_in_redering_uvs[0,1]:copping_lims_in_redering_uvs[3,1]]
        img_out = img_rendered[..., copping_lims_in_redering_uvs[0,1]:copping_lims_in_redering_uvs[3,1], copping_lims_in_redering_uvs[0,0]:copping_lims_in_redering_uvs[3,0]]
        return img_out
        # return img_rendered

    def crop_img(self, img):
        # img: (..., w, h)
        # img_out: (..., crop_size_w, crop_size_h)
        cropping_lims_uv = self.get_cropping_limits_uv()
        img_out = img[..., cropping_lims_uv[0, 1]:cropping_lims_uv[3, 1],
                  cropping_lims_uv[0, 0]:cropping_lims_uv[3, 0]]
        return img_out

    def tr_uvs_to_cropped_uvs(self, uvs):
        # transform uvs in original image space to cropped space
        # uvs: (..., 2)
        cropping_lims_uv = self.get_cropping_limits_uv()
        uvs_cropped = uvs - np.array([cropping_lims_uv[0,0], cropping_lims_uv[0,1]])
        return uvs_cropped

    def check_point_in_cropping_limits(self, point_uvs):
        """

        Args:
            point_uvs (np.ndarray): of shape (..., 2)
        Returns:
            in_limits (np.ndarray): (...,)
        """
        cropping_lims_uv = self.get_cropping_limits_uv()
        in_limits = np.all(point_uvs>=cropping_lims_uv[0], axis=-1) & np.all(point_uvs<=cropping_lims_uv[-1], axis=-1)
        return in_limits


def create_camera_from_params(camera_params):
    camera_kwargs = {
        'pose': np.asarray(camera_params['pose']),
        'K': create_intrinsic_matrix(camera_params['fx'], camera_params['fy'], camera_params['cx'], camera_params['cy']),
        'img_size' : np.asarray(camera_params['img_size'], dtype=np.float32),
        'render_scale' : camera_params['render_scale']
    }
    if 'tactile_plane_distance' in camera_params:
        camera_kwargs['tactile_plane_distance'] = camera_params['tactile_plane_distance']
    camera_i = Camera(**camera_kwargs)
    return camera_i
