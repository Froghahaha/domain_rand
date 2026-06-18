"""Task-agnostic interactive demo collection pipeline.

Orchestrates the full interactive loop:
  1. Load scene, build visual-DR randomizers.
  2. For each episode:
     a. Restore nominal state + apply visual DR.
     b. task.reset() → initial observation.
     c. policy.reset().
     d. Loop:
        - policy.get_action(obs) → action
        - task.step(action) → obs, reward, done
        - render camera → rgb
        - append to episode buffer
        - display with HUD
     e. Write trajectory to HDF5 via ILRecorder.

The Task and Policy are injected — the collector knows nothing
about the specific task logic or action-generation strategy.
"""

import logging
import time
from pathlib import Path

import mujoco
import numpy as np

from domain_rand.core.config import (
    DomainRandConfig,
    CameraRandomizationConfig,
    LightingRandomizationConfig,
    TextureRandomizationConfig,
    PlacementRandomizationConfig,
)
from domain_rand.core.scene import Scene
from domain_rand.randomizers.base import DomainRandomizer
from domain_rand.randomizers.texture import TextureRandomizer
from domain_rand.randomizers.lighting import LightingRandomizer
from domain_rand.randomizers.camera import CameraRandomizer
from domain_rand.randomizers.placement import ObjectPlacementRandomizer
from domain_rand.pipeline.il_recorder import ILRecorder
from domain_rand.pipeline.metadata import build_dataset_attrs
from domain_rand.tasks.base import Task
from domain_rand.policy.base import Policy
from domain_rand.utils.rendering import (
    render_rgb,
    render_depth,
    get_camera_extrinsics,
    get_camera_intrinsics,
)

logger = logging.getLogger(__name__)


class DemoCollector:
    """Task-agnostic interactive demo collector.

    Usage:
        config = load_config("configs/demo_base.yaml")
        task = MyPushTask()
        policy = KeyboardTeleop()

        collector = DemoCollector(config, task, policy)
        collector.run("demos.h5")
    """

    def __init__(self, config: DomainRandConfig, task: Task, policy: Policy):
        self.config = config
        self.task = task
        self.policy = policy

        # ── RNG ────────────────────────────────────────────────────────
        seed = config.seed if config.seed is not None else int(time.time() * 1e6) % (2**31)
        self.rng = np.random.default_rng(seed)
        logger.info(f"RNG seed: {seed}")

        # ── Scene ──────────────────────────────────────────────────────
        self.scene = Scene(config.scene_path)
        logger.info(
            f"Loaded scene: {config.scene_path} "
            f"(geoms={self.scene.ngeom}, lights={self.scene.nlight}, "
            f"cams={self.scene.ncam}, dofs={self.scene.model.nq})"
        )

        # ── Visual DR ──────────────────────────────────────────────────
        self.dr = self._build_domain_randomizer()
        self.scene.save_nominal()
        self.dr.save_nominal(self.scene.model)

        # ── Camera ─────────────────────────────────────────────────────
        self._camera_name = config.il_demo.camera
        self._cam_id = self.scene.get_camera_index(self._camera_name)
        if self._cam_id < 0:
            logger.warning(
                f"Camera '{self._camera_name}' not found in scene. "
                f"Falling back to camera 0."
            )
            self._cam_id = 0
            self._camera_name = self.scene.camera_names[0] if self.scene.ncam > 0 else "cam_0"

        # ── Display ────────────────────────────────────────────────────
        self._display = True
        self._cv2 = None
        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            logger.warning("opencv-python not installed — display disabled.")
            self._display = False

    # ── Domain randomizer construction (same pattern as DatasetCollector) ─

    def _build_domain_randomizer(self) -> DomainRandomizer:
        dr = DomainRandomizer(self.rng)
        dr.register(
            "texture",
            TextureRandomizer(self.config.texture, self.rng),
            enabled=self.config.texture.enabled,
        )
        dr.register(
            "lighting",
            LightingRandomizer(self.config.lighting, self.rng),
            enabled=self.config.lighting.enabled,
        )
        dr.register(
            "placement",
            ObjectPlacementRandomizer(self.config.placement, self.rng),
            enabled=self.config.placement.enabled,
        )
        dr.register(
            "camera",
            CameraRandomizer(self.config.camera, self.rng),
            enabled=self.config.camera.enabled,
        )
        return dr

    # ── Main loop ──────────────────────────────────────────────────────

    def run(self, output_path: str | Path | None = None) -> Path:
        """Run the interactive demo collection loop.

        Args:
            output_path: Output HDF5 path. Auto-generates if None.

        Returns:
            Path to the generated HDF5 file.
        """
        cfg = self.config.il_demo

        if output_path is not None:
            output_path = Path(output_path)
        else:
            output_path = Path(self.config.dataset.output_dir) / f"demos_{int(time.time())}.h5"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        renderer = self.scene.get_renderer(
            height=self.config.dataset.render_height,
            width=self.config.dataset.render_width,
        )

        attrs = build_dataset_attrs(self.config, cfg.num_demos)
        attrs["action_dim"] = self.task.action_spec.get("shape", (0,))[0]
        attrs["state_dim"] = self.task.state_spec.get("shape", (0,))[0]
        attrs["task_type"] = type(self.task).__name__
        attrs["policy_type"] = type(self.policy).__name__

        with ILRecorder(output_path) as recorder:
            recorder.open(attrs)

            ep = 0
            while ep < cfg.num_demos:
                logger.info(f"── Episode {ep + 1}/{cfg.num_demos} ──")

                # ── Reset pipeline ─────────────────────────────────
                self.scene.restore()
                self.dr.restore(self.scene.model)

                # Apply visual DR
                self.dr.randomize(self.scene.model)
                self.scene.forward()

                # Task reset
                obs = self.task.reset(self.scene, self.rng)
                self.policy.reset()

                # Buffers
                rgb_list, state_list, action_list = [], [], []
                reward_list, done_list = [], []

                step = 0
                done = False

                while not done and step < cfg.max_steps:
                    # ── Action ────────────────────────────────────
                    action = self.policy.get_action(obs)
                    if action is None:
                        # No action — keep rendering but don't record
                        if self._display:
                            self._show_frame(
                                obs.get("rgb",
                                        self._render_rgb(renderer)),
                                step, ep, done,
                            )
                        self._check_teleop_commands(ep)
                        continue

                    # ── Step task ─────────────────────────────────
                    obs, reward, done, info = self.task.step(
                        self.scene, action,
                    )

                    # ── Render ────────────────────────────────────
                    rgb = obs.get("rgb") if "rgb" in obs else self._render_rgb(renderer)
                    depth = obs.get("depth") if "depth" in obs else (
                        self._render_depth(renderer) if cfg.save_depth else None
                    )
                    state = obs.get("state", np.zeros(0, dtype=np.float32))

                    # ── Record ────────────────────────────────────
                    rgb_list.append(rgb)
                    state_list.append(state)
                    action_list.append(action)
                    reward_list.append(reward)
                    done_list.append(done)

                    step += 1

                    # ── Display ───────────────────────────────────
                    if self._display:
                        self._show_frame(rgb, step, ep, done)

                    # ── Check teleop meta-commands ────────────────
                    cmd = self._check_teleop_commands(ep)
                    if cmd == "exit":
                        self.policy.close()
                        recorder.close()
                        logger.info(f"Exit requested. Saved {ep} episodes.")
                        return output_path
                    elif cmd == "reset":
                        logger.info("Reset requested — redoing episode.")
                        done = True  # will discard this episode below
                        break

                # ── Write episode (if useful) ────────────────────
                if len(rgb_list) >= 2:
                    rgb_stack = np.stack(rgb_list, axis=0)
                    state_stack = np.stack(state_list, axis=0)
                    action_stack = np.stack(action_list, axis=0)
                    reward_stack = np.array(reward_list, dtype=np.float32)
                    done_stack = np.array(done_list, dtype=bool)

                    meta = {
                        "episode": ep,
                        "steps": step,
                        "camera": self._camera_name,
                        "task": type(self.task).__name__,
                    }
                    recorder.write_episode(
                        ep, rgb_stack, state_stack, action_stack,
                        reward_stack, done_stack,
                        meta=meta,
                    )
                    ep += 1
                    logger.info(
                        f"  Episode {ep} written: {step} steps, "
                        f"final reward={reward:.3f}"
                    )
                else:
                    # Episode too short — skip and retry
                    logger.info(f"  Episode too short ({step} steps), skipping.")

            self.policy.close()

        logger.info(f"Demos saved to: {output_path}")
        return output_path

    # ── Internal helpers ────────────────────────────────────────────────

    def _render_rgb(self, renderer: mujoco.Renderer) -> np.ndarray:
        return render_rgb(
            self.scene.model, self.scene.data, renderer,
            camera=self._cam_id,
        )

    def _render_depth(self, renderer: mujoco.Renderer) -> np.ndarray | None:
        try:
            return render_depth(
                self.scene.model, self.scene.data, renderer,
                camera=self._cam_id,
            )
        except Exception:
            return None

    def _check_teleop_commands(self, current_ep: int) -> str | None:
        """Check for teleop meta-commands and handle n-key advance.

        Returns 'exit', 'reset', or None.
        """
        if not hasattr(self.policy, "wants_exit"):
            return None
        if self.policy.wants_exit():           # type: ignore[union-attr]
            return "exit"
        if self.policy.wants_reset():          # type: ignore[union-attr]
            return "reset"
        if self.policy.wants_next_episode():   # type: ignore[union-attr]
            return "skip"  # handled by caller checking wants_next_episode
        return None

    def _show_frame(
        self,
        rgb: np.ndarray,
        step: int,
        episode: int,
        done: bool,
    ) -> None:
        """Display the current frame with HUD overlay via OpenCV."""
        if self._cv2 is None:
            return

        display = rgb.copy()
        if display.shape[1] > 0 and self.config.il_demo.display_scale != 1.0:
            s = self.config.il_demo.display_scale
            display = self._cv2.resize(display, (0, 0), fx=s, fy=s)

        # HUD overlay
        hud_lines = [
            f"Episode: {episode + 1}/{self.config.il_demo.num_demos}",
            f"Step: {step}/{self.config.il_demo.max_steps}",
            f"Camera: {self._camera_name}",
            f"Done: {done}",
            "[WASD] move  [QE] rotate  [N] next ep  [R] reset  [ESC] quit",
        ]
        font = self._cv2.FONT_HERSHEY_SIMPLEX
        y0 = 25
        for i, line in enumerate(hud_lines):
            color = (0, 255, 0) if i < len(hud_lines) - 1 else (200, 200, 200)
            self._cv2.putText(
                display, line, (10, y0 + i * 22), font,
                0.55, color, 1, self._cv2.LINE_AA,
            )

        self._cv2.imshow("Demo Collection — Domain Rand", display)
        key = self._cv2.waitKey(10)

        # Forward key to teleop policy
        if hasattr(self.policy, "set_key"):
            self.policy.set_key(key)           # type: ignore[union-attr]
