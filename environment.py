import numpy as np
import config
import pandas as pd
import os
import json
from control_types import ControlObservation
from dynamics import advance_dog, advance_sheep
from low_level_controller import LowLevelController
from datetime import datetime

class SwarmEnv:
    def __init__(self, controller=None):
        if config.USE_FIXED_SEED:
            np.random.seed(config.RANDOM_SEED)
            
        self.goal_pos = np.random.uniform(config.GOAL_MARGIN, config.SPACE_SIZE - config.GOAL_MARGIN, 2)
        self.sheep_start_center = np.random.uniform(config.SHEEP_START_MARGIN, config.SPACE_SIZE - config.SHEEP_START_MARGIN, 2)
        self.dog_start_pos = np.random.uniform(config.DOG_START_MARGIN, config.SPACE_SIZE - config.DOG_START_MARGIN, 2)
            
        # randomly initialize the sheep's position and velocity
        self.sheep_pos = self.sheep_start_center + np.random.randn(config.NUM_SHEEP, 2) * config.SHEEP_START_RADIUS
        self.sheep_vel = np.zeros((config.NUM_SHEEP, 2))
        
        # assign abnormal status to sheep
        self.sheep_status = np.array(['Normal'] * config.NUM_SHEEP, dtype=object)
        indices = np.arange(config.NUM_SHEEP)
        np.random.shuffle(indices) # randomly pick sheep to be abnormal
        idx_A = indices[:config.NUM_ABNORMAL_A]
        idx_B = indices[config.NUM_ABNORMAL_A : config.NUM_ABNORMAL_A + config.NUM_ABNORMAL_B]
        self.sheep_status[idx_A] = 'A'
        self.sheep_status[idx_B] = 'B'

        # initialize the shepherd
        self.dog_pos = np.copy(self.dog_start_pos)
        self.dog_vel = np.zeros(2)

        # variables for experimental data recording
        self.current_frame = 0
        self.history_data = []
        self.metrics_data = []
        self.control_data = []
        self.current_mode = "TARGET_TRACKING"
        self.drive_point = np.copy(self.dog_pos)
        self.target_sheep_id = None
        self.control_action = None
        self.navigation_point = np.copy(self.dog_pos)
        self.controller = controller if controller is not None else LowLevelController()
        self.decision_source = "rule"
        self.decision_frame = None
        self.decision_metadata = {}

    def _record_state(self):
        # Record Dog's state
        self.history_data.append([self.current_frame, 'Dog', 0, round(float(self.dog_pos[0]), 2), round(float(self.dog_pos[1]), 2), 'Dog'])
        # Record each Sheep's state
        for i, pos in enumerate(self.sheep_pos):
            self.history_data.append([self.current_frame, 'Sheep', i, round(float(pos[0]), 2), round(float(pos[1]), 2), self.sheep_status[i]])

    def record_frame_metrics(self, metrics):
        """record the calculated metrics for the current frame. The input is a dictionary of metric_name: value."""
        metrics_dict = metrics.copy()
        metrics_dict['Frame'] = self.current_frame
        self.metrics_data.append(metrics_dict)

    def _record_control_state(self, frame, alarms, current_mode, drive_point, dog_pos_before):
        alarm_text = ";".join(alarms) if alarms else "none"
        self.control_data.append({
            'Frame': frame,
            'repair_enabled': int(config.REPAIR_ENABLED),
            'alarms': alarm_text,
            'current_mode': current_mode,
            'decision_source': getattr(self, 'decision_source', 'rule'),
            'decision_frame': getattr(self, 'decision_frame', frame),
            'high_level_status': self.decision_metadata.get('high_level_status'),
            'llm_request_id': self.decision_metadata.get('request_id'),
            'llm_trigger_frame': self.decision_metadata.get('trigger_frame'),
            'llm_trigger_alarm': self.decision_metadata.get('trigger_alarm'),
            'llm_trigger_alarms': json.dumps(
                self.decision_metadata.get('trigger_alarms'), sort_keys=True
            ),
            'llm_state_summary': json.dumps(
                self.decision_metadata.get('state_summary'), sort_keys=True
            ),
            'llm_selected_mode': self.decision_metadata.get('selected_mode'),
            'llm_reference_mode': self.decision_metadata.get('reference_mode'),
            'llm_selection_correct': self.decision_metadata.get('selection_correct'),
            'llm_latency_frames': self.decision_metadata.get('latency_frames'),
            'llm_latency_seconds': self.decision_metadata.get('latency_seconds'),
            'llm_hold_frames': self.decision_metadata.get('hold_frames'),
            'llm_response_mode': self.decision_metadata.get('response_mode'),
            'llm_response_hold_frames': self.decision_metadata.get(
                'response_hold_frames'
            ),
            'llm_reason': self.decision_metadata.get('reason'),
            'llm_model': self.decision_metadata.get('model'),
            'llm_prompt_version': self.decision_metadata.get('prompt_version'),
            'llm_cache_hit': self.decision_metadata.get('cache_hit'),
            'llm_response_id': self.decision_metadata.get('response_id'),
            'llm_response_status': self.decision_metadata.get('response_status'),
            'llm_incomplete_reason': self.decision_metadata.get('incomplete_reason'),
            'llm_input_tokens': self.decision_metadata.get('input_tokens'),
            'llm_output_tokens': self.decision_metadata.get('output_tokens'),
            'llm_backend_error': self.decision_metadata.get('backend_error'),
            'llm_rejection_reason': self.decision_metadata.get('rejection_reason'),
            'llm_release_reason': self.decision_metadata.get('release_reason'),
            'llm_released_request_id': self.decision_metadata.get(
                'released_request_id'
            ),
            'llm_released_mode': self.decision_metadata.get('released_mode'),
            'llm_released_active_frames': self.decision_metadata.get(
                'released_active_frames'
            ),
            'decision_metadata': json.dumps(
                self.decision_metadata, sort_keys=True, default=str
            ),
            'control_action': self.control_action,
            'target_sheep_id': self.target_sheep_id,
            'goal_retention_enabled': int(config.GOAL_RETENTION_ENABLED),
            'navigation_point_x': float(self.navigation_point[0]),
            'navigation_point_y': float(self.navigation_point[1]),
            'drive_point_x': round(float(drive_point[0]), 2),
            'drive_point_y': round(float(drive_point[1]), 2),
            'dog_x_before': round(float(dog_pos_before[0]), 2),
            'dog_y_before': round(float(dog_pos_before[1]), 2),
            'dog_x_after': round(float(self.dog_pos[0]), 2),
            'dog_y_after': round(float(self.dog_pos[1]), 2),
        })

    def save_data_to_parquet(self, timestamp):
        if not os.path.exists(config.DATA_LOG_DIR):
            os.makedirs(config.DATA_LOG_DIR)
            
        export_path = os.path.join(config.DATA_LOG_DIR, f"data_{timestamp}.parquet")
        
        df_agents = pd.DataFrame(self.history_data, columns=['Frame', 'Agent_Type', 'Agent_ID', 'X', 'Y', 'Status'])
        df_metrics = pd.DataFrame(self.metrics_data)

        # if metrics data is empty (e.g., if the detector never recorded any metrics), we should still save the agent data without merging
        if not df_metrics.empty:
            final_df = pd.merge(df_agents, df_metrics, on='Frame', how='left')
        else:
            final_df = df_agents
            
        final_df.to_parquet(export_path, engine='pyarrow')
        
        parquet_files = [os.path.join(config.DATA_LOG_DIR, f) for f in os.listdir(config.DATA_LOG_DIR) if f.endswith('.parquet')]
        if len(parquet_files) > config.MAX_LOG_FILES:
            parquet_files.sort(key=os.path.getmtime)
            for f in parquet_files[:-config.MAX_LOG_FILES]:
                try:
                    os.remove(f)
                    # Linked deletion of corresponding visual files
                    basename = os.path.basename(f)
                    ts = basename.replace("data_", "").replace(".parquet", "")
                    png_path = os.path.join(config.VISUAL_LOG_DIR, f"trajectory_{ts}.png")
                    mp4_path = os.path.join(config.VIDEO_LOG_DIR, f"animation_{ts}.mp4")
                    if os.path.exists(png_path): os.remove(png_path)
                    if os.path.exists(mp4_path): os.remove(mp4_path)
                except OSError as e:
                    print(f"Warning: Could not remove old log file {f}: {e}")

    def save_control_to_parquet(self, timestamp):
        if not os.path.exists(config.CONTROL_LOG_DIR):
            os.makedirs(config.CONTROL_LOG_DIR)

        export_path = os.path.join(config.CONTROL_LOG_DIR, f"control_{timestamp}.parquet")
        df_control = pd.DataFrame(self.control_data)
        df_control.to_parquet(export_path, engine='pyarrow')

        control_files = [os.path.join(config.CONTROL_LOG_DIR, f) for f in os.listdir(config.CONTROL_LOG_DIR) if f.endswith('.parquet')]
        if len(control_files) > config.MAX_LOG_FILES:
            control_files.sort(key=os.path.getmtime)
            for f in control_files[:-config.MAX_LOG_FILES]:
                try:
                    os.remove(f)
                except OSError as e:
                    print(f"Warning: Could not remove old control log file {f}: {e}")

    def step(self, alarms=None):
        """execute one frame of physics calculation. if the sheep reaches the goal, return True"""
        if alarms is None:
            alarms = []

        frame = self.current_frame
        self._record_state()
        self.current_frame += 1

        # 1. calculate the sheep's center of mass (Center of Mass)
        com = np.mean(self.sheep_pos, axis=0)
        
        # --- shepherd logic ---
        to_goal = self.goal_pos - com
        dist_to_goal = np.linalg.norm(to_goal)
        
        # success judgment: all sheep must be within distance 20 to the goal
        distances_to_goal = np.linalg.norm(self.sheep_pos - self.goal_pos, axis=1)
        if np.all(distances_to_goal < config.GOAL_RADIUS):
            return True

        sheep_snapshot = np.copy(self.sheep_pos)
        sheep_velocity_snapshot = np.copy(self.sheep_vel)
        dog_snapshot = np.copy(self.dog_pos)
        dog_velocity_snapshot = np.copy(self.dog_vel)
        goal_snapshot = np.copy(self.goal_pos)
        for snapshot in (
            sheep_snapshot,
            sheep_velocity_snapshot,
            dog_snapshot,
            dog_velocity_snapshot,
            goal_snapshot,
        ):
            snapshot.flags.writeable = False
        observation = ControlObservation(
            sheep_positions=sheep_snapshot,
            dog_position=dog_snapshot,
            goal_position=goal_snapshot,
            alarms=tuple(alarms),
            frame=frame,
            sheep_statuses=tuple(self.sheep_status.tolist()),
            current_mode=self.current_mode,
            sheep_velocities=sheep_velocity_snapshot,
            dog_velocity=dog_velocity_snapshot,
        )
        decision = self.controller.decide(observation)
        self._apply_control_decision(decision)

        dog_pos_before = np.copy(self.dog_pos)
        self.dog_pos, self.dog_vel = advance_dog(
            self.dog_pos,
            self.navigation_point,
            config,
        )
        self._record_control_state(frame, alarms, self.current_mode, self.drive_point, dog_pos_before)

        self.sheep_pos, self.sheep_vel = advance_sheep(
            self.sheep_pos,
            self.sheep_vel,
            self.sheep_status,
            self.dog_pos,
            self.goal_pos,
            config,
        )

        return False

    def _apply_control_decision(self, decision):
        self.current_mode = decision.mode
        self.control_action = decision.action
        self.target_sheep_id = decision.target_sheep_id
        self.drive_point = np.copy(decision.drive_point)
        self.navigation_point = np.copy(decision.navigation_point)
        self.decision_source = decision.source
        self.decision_frame = decision.issued_for_frame
        self.decision_metadata = dict(decision.metadata)

    def repair_sheep(self, indices):
        """
        Intervention API to repair abnormal sheep back to the 'Normal' state.
        :param indices: A single integer, a list of integers, or a NumPy array of sheep indices.
        """
        if isinstance(indices, int):
            indices = [indices]
        self.sheep_status[indices] = 'Normal'

    def close(self):
        close = getattr(self.controller, "close", None)
        if close is not None:
            close()
