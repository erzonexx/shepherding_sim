"""Shared data contract between the high- and low-level controllers."""

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class ControlObservation:
    sheep_positions: np.ndarray
    dog_position: np.ndarray
    goal_position: np.ndarray
    alarms: Sequence[str]
    frame: int = 0
    sheep_statuses: Sequence[str] = ()
    current_mode: str = "TARGET_TRACKING"
    sheep_velocities: Optional[np.ndarray] = None
    dog_velocity: Optional[np.ndarray] = None


@dataclass(frozen=True)
class ControlDecision:
    mode: str
    action: str
    target_sheep_id: Optional[int]
    drive_point: np.ndarray
    navigation_point: np.ndarray
    issued_for_frame: Optional[int] = None
    source: str = "rule"
    metadata: dict[str, Any] = field(default_factory=dict)


class Controller(Protocol):
    def decide(self, observation: ControlObservation) -> ControlDecision:
        """Return one command for the current observation."""
