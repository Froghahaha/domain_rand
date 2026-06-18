"""Task-agnostic interactive demo collection pipeline.

Orchestrates the full interactive loop:
  1. Load scene, build visual-DR randomizers, create pluggable recorder.
  2. For each episode:
     a. Restore nominal state + apply visual DR.
     b. task.reset() → initial observation.
     c. policy.reset().
     d. Loop:
        - policy.get_action(obs) → action
        - task.step(action) → obs, reward, done, info
        - render camera → supplement obs with rgb
        - append obs keys + info keys to episode buffers
        - display with HUD
     e. Write trajectory to HDF5 via pluggable BaseRecorder.

The Task and Policy are injected — the collector knows nothing
about the specific task logic or action-generation strategy.

Recorder selection:
    config.il_demo.recorder = "simple"        → SimpleRecorder (default)
    config.il_demo.recorder = "robomimic"     → RobomimicRecorder
    config.il_demo.recorder = "pkg.MyRecorder" → custom (loaded dynamically)
"""

import importlib
import json
import logging
import time
from pathlib import Path

import mujoco
import numpy as np

from domain_rand.core.config import (
    DomainRandConfig,
)
from domain_rand.core.scene import Scene
from domain_rand.randomizers.base import DomainRandomizer
from domain_rand.randomizers.texture import TextureRandomizer
from domain_rand.randomizers.lighting import LightingRandomizer
from domain_rand.randomizers.camera import CameraRandomizer
from domain_rand.pipeline.recorders.base import BaseRecorder
from domain_rand.pipeline.metadata import build_dataset_attrs
from domain_rand.tasks.base import Task
from domain_rand.policy.base import Policy
from domain_rand.utils.rendering import (
    render_rgb,
    render_depth,
)

logger = logging.getLogger(__name__)

# ── Built-in recorder registry ──────────────────────────────────────────

_BUILTIN_RECORDERS: dict[str, str] = {
    "simple": "domain_rand.pipeline.recorders.simple.SimpleRecorder",
    "robomimic": "domain_rand.pipeline.recorders.robomimic.RobomimicRecorder",
}


def _make_recorder(config: DomainRandConfig, output_path: Path) -> BaseRecorder:
    """Instantiate the recorder specified in config.il_demo.recorder."""
    cfg = config.il_demo
    recorder_name = getattr(cfg, "recorder", "simple") or "simple"
    kwargs = dict(getattr(cfg, "recorder_kwargs", {}) or {})

    # Resolve class path
    if recorder_name in _BUILTIN_RECORDERS:
        class_path = _BUILTIN_RECORDERS[recorder_name]
    else:
        class_path = recorder_name

    parts = class_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid recorder class path: '{class_path}'")
    mod = importlib.import_module(parts[0])
    cls = getattr(mod, parts[1])

    # Read scene XML for robomimic model_file
    scene_xml = Path(config.scene_path).read_text(encoding="utf-8")

    recorder = cls(output_path, **kwargs)
    if hasattr(recorder, "set_scene_xml"):
        recorder.set_scene_xml(scene_xml)
    return recorder


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
            if self.scene.ncam > 0:
                self._camera_name = self.scene.camera_names[0]
            else:
                self._camera_name = "cam_0"

        # ── Display ────────────────────────────────────────────────────
        self._display = True
        self._cv2 = None
        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            logger.warning("opencv-python not installed — display disabled.")
            self._display = False

    # ── Domain randomizer construction ─────────────────────────────────

    def _build_domain_randomizer(self) -> DomainRandomizer:
        dr = DomainRandomizer(self.rng)

        # Task-agnostic visual randomizers (always registered)
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
            "camera",
            CameraRandomizer(self.config.camera, self.rng),
            enabled=self.config.camera.enabled,
        )

        # Task-specific randomizers (registered by the task itself)
        self.task.register_randomizers(dr, self.rng)

        return dr

    # ── Main loop ─────────────────────────────────────────────────────

    def run(self, output_path: str | Path | None = None) -> Path:
        """Run the interactive demo collection loop."""
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

        # Build recorder (pluggable — simple / robomimic / custom)
        recorder = _make_recorder(self.config, output_path)

        attrs = build_dataset_attrs(self.config, cfg.num_demos)
        attrs["action_dim"] = self.task.action_spec.get("shape", (0,))[0]
        attrs["state_dim"] = self.task.state_spec.get("shape", (0,))[0]
        attrs["task_type"] = type(self.task).__name__
        attrs["policy_type"] = type(self.policy).__name__

        recorder.open(attrs)

        ep = 0
        while ep < cfg.num_demos:
            logger.info(f"── Episode {ep + 1}/{cfg.num_demos} ──")

            # ── Reset pipeline ──────────────────────────────────
            self.scene.restore()
            self.dr.restore(self.scene.model)

            self.dr.randomize(self.scene.model)
            self.scene.forward()

            obs = self.task.reset(self.scene, self.rng)
            self.policy.reset()

            # Buffers: one list per observation key, plus info keys
            obs_buffers: dict[str, list[np.ndarray]] = {}
            info_buffers: dict[str, list] = {}
            action_list: list[np.ndarray] = []
            reward_list: list[float] = []
            done_list: list[bool] = []

            step = 0
            done = False

            while not done and step < cfg.max_steps:
                # ── Action ─────────────────────────────────────
                action = self.policy.get_action(obs)
                if action is None:
                    if self._display:
                        rgb = obs.get("rgb", self._render_rgb(renderer))
                        self._show_frame(rgb, step, ep, done)
                    self._check_teleop_commands(ep)
                    continue

                # ── Step task ──────────────────────────────────
                obs, reward, done, info = self.task.step(
                    self.scene, action,
                )

                # ── Supplement obs with rendered image ─────────
                if "rgb" not in obs:
                    obs["rgb"] = self._render_rgb(renderer)
                if cfg.save_depth and "depth" not in obs:
                    d = self._render_depth(renderer)
                    if d is not None:
                        obs["depth"] = d

                # ── Accumulate ────────────────────────────────
                action_list.append(action)
                reward_list.append(reward)
                done_list.append(done)
                step += 1

                for key, val in obs.items():
                    if key not in obs_buffers:
                        obs_buffers[key] = []
                    obs_buffers[key].append(val)

                for ikey, ival in info.items():
                    if ikey not in info_buffers:
                        info_buffers[ikey] = []
                    info_buffers[ikey].append(ival)

                # ── Display ────────────────────────────────────
                if self._display:
                    rgb = obs.get("rgb", np.zeros((64, 64, 3), dtype=np.uint8))
                    self._show_frame(rgb, step, ep, done)

                # ── Teleop meta-commands ───────────────────────
                cmd = self._check_teleop_commands(ep)
                if cmd == "exit":
                    self.policy.close()
                    recorder.close()
                    logger.info(f"Exit requested. Saved {ep} episodes.")
                    return output_path
                elif cmd == "reset":
                    logger.info("Reset requested — redoing episode.")
                    done = True
                    break

            # ── Write episode ──────────────────────────────────
            if len(action_list) >= 2:
                obs_stacks = {k: np.stack(v, axis=0) for k, v in obs_buffers.items()}
                action_stack = np.stack(action_list, axis=0)
                reward_stack = np.array(reward_list, dtype=np.float32)
                done_stack = np.array(done_list, dtype=bool)
                info_stacks = {
                    k: np.array(v, dtype=np.float32)
                    for k, v in info_buffers.items()
                } if info_buffers else None

                meta = {
                    "episode": ep,
                    "steps": step,
                    "camera": self._camera_name,
                    "task": type(self.task).__name__,
                }

                recorder.write_episode(
                    ep, obs_stacks, action_stack, reward_stack,
                    done_stack, infos=info_stacks, meta=meta,
                )
                ep += 1
                logger.info(
                    f"  Episode {ep} written: {step} steps, "
                    f"final reward={reward:.3f}"
                )
            else:
                logger.info(f"  Episode too short ({step} steps), skipping.")

        self.policy.close()
        recorder.close()

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
        if not hasattr(self.policy, "wants_exit"):
            return None
        if self.policy.wants_exit():           # type: ignore[union-attr]
            return "exit"
        if self.policy.wants_reset():          # type: ignore[union-attr]
            return "reset"
        return None

    def _show_frame(
        self, rgb: np.ndarray, step: int, episode: int, done: bool,
    ) -> None:
        if self._cv2 is None:
            return

        display = rgb.copy()
        s = self.config.il_demo.display_scale
        if s != 1.0 and display.size > 0:
            display = self._cv2.resize(display, (0, 0), fx=s, fy=s)

        recorder_name = getattr(self.config.il_demo, "recorder", "simple") or "simple"
        hud_lines = [
            f"Episode: {episode + 1}/{self.config.il_demo.num_demos}",
            f"Step: {step}/{self.config.il_demo.max_steps}",
            f"Camera: {self._camera_name}  Recorder: {recorder_name}",
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

        if hasattr(self.policy, "set_key"):
            self.policy.set_key(key)           # type: ignore[union-attr]
