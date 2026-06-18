"""Object placement randomizer for MuJoCo scenes.

Randomly places 1-3 STL/mesh objects on a table surface per episode.
Objects not selected are moved far away (hidden).
"""

import mujoco
import numpy as np

from domain_rand.core.config import PlacementRandomizationConfig
from domain_rand.randomizers.base import Randomizer


class ObjectPlacementRandomizer(Randomizer):
    """Randomly places 1–N objects on a table surface per episode.

    On each call to randomize():
      1. Randomly picks how many objects to show (1 to max).
      2. Randomly selects which object bodies to use.
      3. Places each selected object at a random (x, y) on the table
         with random yaw rotation.
      4. Moves unselected objects to a hidden location (far below).

    save_nominal() remembers all original body positions/quaternions.
    restore() puts everything back.
    """

    def __init__(self, config: PlacementRandomizationConfig, rng: np.random.Generator):
        super().__init__(rng)
        self.config = config
        self._nominal_pos: dict[int, np.ndarray] = {}
        self._nominal_quat: dict[int, np.ndarray] = {}
        self._body_ids: list[int] = []
        self._active_bodies: list[int] = []  # which bodies are visible this episode
        self._mesh_z_bottom: dict[int, float] = {}  # per-body Z offset to sit on surface

    def save_nominal(self, model: mujoco.MjModel) -> None:
        """Resolve body names, remember their starting pose, and compute
        mesh Z extents so objects sit on the surface instead of in it."""
        self._body_ids = []
        self._nominal_pos = {}
        self._nominal_quat = {}
        self._mesh_z_bottom = {}

        for name in self.config.object_bodies:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                continue
            self._body_ids.append(bid)
            self._nominal_pos[bid] = model.body_pos[bid].copy()
            self._nominal_quat[bid] = model.body_quat[bid].copy()

            # Compute how far the mesh extends below its body origin.
            # Without this offset, the mesh origin sits at table_z and
            # the geometry is half-embedded in the table.
            z_bottom = self._compute_mesh_z_bottom(model, bid)
            self._mesh_z_bottom[bid] = z_bottom

    @staticmethod
    def _compute_mesh_z_bottom(model: mujoco.MjModel, body_id: int) -> float:
        """Return the distance from the body origin to the lowest vertex
        of any mesh geom attached to this body.  Positive = mesh extends
        below the origin, so we must raise the body by this amount."""
        offset = 0.0
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] != body_id:
                continue
            if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh_id = model.geom_dataid[gid]
            if mesh_id < 0 or mesh_id >= model.nmesh:
                continue
            vert_start = model.mesh_vertadr[mesh_id]
            vert_count = model.mesh_vertnum[mesh_id]
            if vert_count <= 0:
                continue
            verts = model.mesh_vert[vert_start:vert_start + vert_count]
            z_min = verts[:, 2].min()
            # If the mesh origin is above the lowest vertex, offset needed
            if z_min < offset:
                # geom_pos contributes a Z shift inside the body frame
                geom_z = model.geom_pos[gid, 2]
                offset = max(offset, -(z_min + geom_z))
        return offset

    def randomize(self, model: mujoco.MjModel) -> None:
        """Pick how many objects, choose which ones, and place them randomly."""
        if not self.config.enabled or not self._body_ids:
            return

        n_min, n_max = self.config.num_objects_range
        n = int(self.rng.integers(n_min, n_max + 1))

        # Pick which bodies to show
        indices = self.rng.choice(len(self._body_ids), size=n, replace=False)
        self._active_bodies = [self._body_ids[i] for i in indices]
        active_set = set(self._active_bodies)

        x_min, x_max = self.config.table_x_range
        y_min, y_max = self.config.table_y_range
        z_table = self.config.table_z
        yaw_min, yaw_max = self.config.yaw_range

        for bid in self._body_ids:
            if bid in active_set:
                # Place randomly on the table, raised so the mesh sits on top
                x = self.rng.uniform(x_min, x_max)
                y = self.rng.uniform(y_min, y_max)
                z_offset = self._mesh_z_bottom.get(bid, 0.0)
                model.body_pos[bid] = np.array([x, y, z_table + z_offset])

                # Random yaw rotation (around Z)
                yaw_deg = self.rng.uniform(yaw_min, yaw_max)
                yaw_rad = np.deg2rad(yaw_deg)
                qw = np.cos(yaw_rad / 2)
                qz = np.sin(yaw_rad / 2)
                model.body_quat[bid] = np.array([qw, 0.0, 0.0, qz])
            else:
                # Move hidden objects far below the scene
                model.body_pos[bid] = np.array([0.0, 0.0, self.config.hide_z])

    def restore(self, model: mujoco.MjModel) -> None:
        """Restore all object bodies to their nominal poses."""
        for bid, pos in self._nominal_pos.items():
            model.body_pos[bid] = pos.copy()
        for bid, quat in self._nominal_quat.items():
            model.body_quat[bid] = quat.copy()

    @property
    def active_bodies(self) -> list[int]:
        """Body indices that are currently visible (after last randomize)."""
        return self._active_bodies

    def get_state(self, model: mujoco.MjModel) -> dict:
        """Return current placement state for recording."""
        state = {}
        for bid in self._body_ids:
            name = ""
            for i in range(model.nbody):
                if model.name_bodyadr[i] >= 0 and i == bid:
                    addr = model.name_bodyadr[i]
                    name = model.names[addr:].split(b"\x00")[0].decode()
                    break
            state[name or f"body_{bid}"] = {
                "pos": model.body_pos[bid].tolist(),
                "quat": model.body_quat[bid].tolist(),
                "active": bid in (set(self._active_bodies) if hasattr(self, '_active_bodies') else set()),
            }
        return {
            "num_active": len(self._active_bodies) if hasattr(self, '_active_bodies') else 0,
            "bodies": state,
        }
