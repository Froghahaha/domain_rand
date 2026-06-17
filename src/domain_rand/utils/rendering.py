"""Rendering utilities for MuJoCo — RGB, depth, and camera matrices."""

import mujoco
import numpy as np


def render_rgb(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    camera: int | str = 0,
) -> np.ndarray:
    """Render an RGB image from the given camera.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        renderer: MuJoCo Renderer instance.
        camera: Camera index (int) or name (str).

    Returns:
        uint8 RGB array of shape (H, W, 3).
    """
    if isinstance(camera, str):
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    else:
        cam_id = camera

    renderer.update_scene(data, camera=cam_id)
    rgb = renderer.render()
    return rgb  # (H, W, 3) uint8


def render_depth(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    camera: int | str = 0,
) -> np.ndarray:
    """Render a depth image from the given camera.

    Returns planar depth (distance from camera plane) as normalized values
    in [0, 1], where 0 = near plane and 1 = far plane.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        renderer: MuJoCo Renderer instance.
        camera: Camera index (int) or name (str).

    Returns:
        float32 depth array of shape (H, W).
    """
    if isinstance(camera, str):
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    else:
        cam_id = camera

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam_id)
    depth = renderer.render()
    renderer.disable_depth_rendering()
    return depth  # (H, W) float32


def get_camera_extrinsics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: int | str = 0,
) -> np.ndarray:
    """Get the 4x4 camera extrinsics matrix (world-to-camera transform).

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        camera: Camera index (int) or name (str).

    Returns:
        4x4 float32 extrinsics matrix.
    """
    if isinstance(camera, str):
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    else:
        cam_id = camera

    # Camera position in world frame
    cam_pos = data.cam_xpos[cam_id].copy()
    # Camera rotation matrix (world-to-camera)
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()

    # Build 4x4 extrinsics: [R | t; 0 0 0 1]
    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[:3, :3] = cam_mat
    extrinsics[:3, 3] = -cam_mat @ cam_pos  # world origin in camera frame
    return extrinsics


def get_camera_intrinsics(
    model: mujoco.MjModel,
    renderer: mujoco.Renderer,
    camera: int | str = 0,
) -> np.ndarray:
    """Get the 3x3 camera intrinsics matrix.

    Args:
        model: MuJoCo model.
        renderer: MuJoCo Renderer instance.
        camera: Camera index (int) or name (str).

    Returns:
        3x3 float32 intrinsics matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1].
    """
    if isinstance(camera, str):
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    else:
        cam_id = camera

    h = renderer._height
    w = renderer._width
    fovy = model.cam_fovy[cam_id]  # degrees

    # Compute focal length from FOV
    f = (h / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)

    intrinsics = np.array(
        [
            [f, 0, w / 2.0],
            [0, f, h / 2.0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    return intrinsics
