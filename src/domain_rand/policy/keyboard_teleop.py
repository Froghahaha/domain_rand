"""Keyboard-teleoperation policy for demo collection.

Uses OpenCV's waitKey to read keyboard input in the render loop.
Maps configurable keys to action dimensions — users can extend
the keymap for custom action spaces.
"""

import numpy as np

from domain_rand.policy.base import Policy


class KeyboardTeleop(Policy):
    """Keyboard-based teleoperation.

    Designed to work with the DemoCollector's OpenCV display loop.
    Each call to get_action() reads the last keypress from the
    collector (stored on the policy instance) and converts it to
    an action vector.

    Key mappings are defined in `_keymap`: each entry is
        (key_code, dim_index, delta)
    where key_code is an OpenCV keycode (ord('w'), etc.),
    dim_index is the action dimension to modify, and delta is
    the signed increment.

    Built-in mapping (WASD for XY, QE for yaw):
        w / s  →  dim 0  (+ / - linear_speed)
        a / d  →  dim 1  (+ / - linear_speed)
        q / e  →  dim 2  (+ / - rot_speed)

    Special keys:
        n     →  signals episode-done (via info)
        r     →  signals reset request
        ESC   →  signals exit
    """

    # Default key → (dim, delta) mapping.  Override or extend in subclasses.
    DEFAULT_KEYMAP: list[tuple[int, int, float]] = []

    def __init__(
        self,
        linear_speed: float = 0.02,
        rot_speed: float = 0.1,
        action_dim: int = 3,
    ):
        self.linear_speed = linear_speed
        self.rot_speed = rot_speed
        self.action_dim = action_dim

        # Build default keymap if not overridden
        if not self.DEFAULT_KEYMAP:
            self.DEFAULT_KEYMAP = [
                (ord("w"), 0, +1),
                (ord("s"), 0, -1),
                (ord("a"), 1, +1),
                (ord("d"), 1, -1),
                (ord("q"), 2, +1),
                (ord("e"), 2, -1),
            ]

        self._keymap = list(self.DEFAULT_KEYMAP)
        self._last_key: int = -1

    # ── Public API ──────────────────────────────────────────────────────

    def set_key(self, keycode: int) -> None:
        """Called by the DemoCollector each frame with the latest keypress.

        Special keys trigger meta-actions:
            27 (ESC)  →  stored so collector can detect exit
            ord('n')  →  stored so collector can advance episode
            ord('r')  →  stored so collector can reset
        """
        self._last_key = keycode

    def get_action(self, observation: dict) -> np.ndarray | None:
        """Convert the last keypress to an action vector.

        Returns:
            np.ndarray of shape (action_dim,) if a mapped key is held.
            None if no key was pressed (robot should hold position).
        """
        if self._last_key < 0:
            return None

        action = np.zeros(self.action_dim, dtype=np.float32)

        for keycode, dim, sign in self._keymap:
            if self._last_key == keycode:
                speed = self.rot_speed if dim >= 2 else self.linear_speed
                action[dim] = sign * speed
                return action

        # Key was a special key (n/r/esc) or unmapped — no action
        return None

    def wants_next_episode(self) -> bool:
        """Check if the user pressed 'n' to advance to the next episode."""
        return self._last_key == ord("n")

    def wants_reset(self) -> bool:
        """Check if the user pressed 'r' to reset the current episode."""
        return self._last_key == ord("r")

    def wants_exit(self) -> bool:
        """Check if the user pressed ESC to quit."""
        return self._last_key == 27

    def reset(self) -> None:
        """Clear key state at episode start."""
        self._last_key = -1

    def add_key_binding(self, key: str, dim: int, sign: float) -> None:
        """Add or override a key binding.

        Args:
            key: Single-character string (e.g. 'w').
            dim: Action dimension index to modify.
            sign: +1 or -1 multiplier.
        """
        keycode = ord(key.lower())
        # Remove existing binding for this key
        self._keymap = [(k, d, s) for k, d, s in self._keymap if k != keycode]
        self._keymap.append((keycode, dim, sign))

    @property
    def status_text(self) -> str:
        """Human-readable status line for HUD display."""
        k = self._last_key
        if k < 0:
            return "idle"
        key_str = chr(k) if 32 <= k <= 126 else f"0x{k:x}"
        return f"key={key_str}"
