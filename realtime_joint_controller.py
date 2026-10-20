"""Run the existing mode-and-target policy on the measured background pipeline.

The target-only handoff-v1 contract explicitly forbids mode selection. This
adapter therefore retains the pre-existing joint policy, forecasts, holds and
releases, rather than silently changing that contract. Joint-vs-rule is the
primary comparison; joint-vs-handoff-v1 is not a one-switch ablation.
"""
import copy

from high_level_controller import HighLevelController
from realtime_controller import AsyncTargetController

VERSION = 'async-original-mode-target-v1'


class AsyncJointController(AsyncTargetController):
    def __init__(self, backend, low_level_controller=None, settings=None,
                 validation_max_age_frames=30):
        if settings is None:
            import config
            settings = config
        if (not settings.HIGH_LEVEL_MODE_SELECTION_ENABLED
                or not settings.HIGH_LEVEL_TARGET_SELECTION_ENABLED
                or settings.HIGH_LEVEL_TARGET_HANDOFF_ENABLED
                or settings.HIGH_LEVEL_EXPLICIT_EXECUTION):
            raise ValueError('Joint policy requires both permissions and original execution semantics')
        HighLevelController.__init__(self, low_level_controller, backend, settings)
        backend.controller = self
        backend.target_handoff_enabled = False
        backend.prompt_version = VERSION
        self.validation_max_age_frames = validation_max_age_frames
        self.latest_observation = self.latest_summary = None

    def _with_mode_forecasts(self, observation, summary, include_targets=False, response=None):
        # Inherited implementation consumes only completed worker forecasts and
        # refreshes explicit-target eligibility and geometry against current state.
        cached = super()._with_mode_forecasts(observation, summary, include_targets, response)
        if isinstance(response, dict) and response.get('target_sheep_id') is None:
            mode = response.get('mode')
            key = {'HARD_TO_GUIDE': 'hard_to_guide_candidate',
                   'COHESION_PRIORITY': 'cohesion_candidate'}.get(mode)
            if key and key in cached:
                current = copy.copy(self.low_level_controller).decide(observation, mode_override=mode)
                cached[key] = dict(cached[key], drive_point_boundary_margin=self._boundary_margin(current.drive_point))
        return cached

    def _validate_response(self, response, frame, alarms, current_summary):
        # Preserve the same concurrent-alarm admission condition as the latest
        # target-only cohort. Subsequent early release remains the original policy.
        if not {'flock_splitting', 'stagnation'} <= set(alarms):
            return None, None, 'concurrent_alarms_cleared'
        return super()._validate_response(response, frame, alarms, current_summary)
