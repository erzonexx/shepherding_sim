import numpy as np
import config
import pandas as pd
import os
from datetime import datetime

class SwarmEnv:
    def __init__(self):
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

        # --- shepherd mode decision ---
        if not config.REPAIR_ENABLED:
            current_mode = "TARGET_TRACKING"
        elif "flock_splitting" in alarms:
            current_mode = "COHESION_PRIORITY"
        elif "stagnation" in alarms:
            current_mode = "HARD_TO_GUIDE"
        else:
            current_mode = "TARGET_TRACKING"
        self.current_mode = current_mode

        # --- shepherd drive-point calculation ---
        vecs_to_com = self.sheep_pos - com
        dists_to_com = np.linalg.norm(vecs_to_com, axis=1)
        max_dist_to_com = np.max(dists_to_com)

        if current_mode == "TARGET_TRACKING":
            if max_dist_to_com > config.COHESION_THRESHOLD:
                furthest_idx = np.argmax(dists_to_com)
                furthest_pos = self.sheep_pos[furthest_idx]
                dir_from_com = vecs_to_com[furthest_idx] / (max_dist_to_com + config.EPSILON)
                drive_point = furthest_pos + dir_from_com * config.DOG_DRIVE_DISTANCE
            else:
                dir_to_goal = to_goal / (dist_to_goal + config.EPSILON)
                drive_point = com - dir_to_goal * config.DOG_DRIVE_DISTANCE
        elif current_mode == "HARD_TO_GUIDE":
            distances_to_goal = np.linalg.norm(self.sheep_pos - self.goal_pos, axis=1)
            target_idx = np.argmax(distances_to_goal)
            target_pos = self.sheep_pos[target_idx]
            dir_to_goal = (self.goal_pos - target_pos) / (distances_to_goal[target_idx] + config.EPSILON)
            drive_point = target_pos - dir_to_goal * 3.0
        elif current_mode == "COHESION_PRIORITY":
            furthest_idx = np.argmax(dists_to_com)
            furthest_pos = self.sheep_pos[furthest_idx]
            dir_from_com = vecs_to_com[furthest_idx] / (max_dist_to_com + config.EPSILON)
            drive_point = furthest_pos + dir_from_com * 10.0
        else:
            raise ValueError(f"Unknown shepherd mode: {current_mode}")
        self.drive_point = np.copy(drive_point)

        dog_desired_vel = drive_point - self.dog_pos
        dog_pos_before = np.copy(self.dog_pos)
        self.dog_vel = self._limit_speed(dog_desired_vel, config.DOG_MAX_SPEED)
        self.dog_pos += self.dog_vel
        self.dog_pos = np.clip(self.dog_pos, 0, config.SPACE_SIZE)
        self._record_control_state(frame, alarms, current_mode, drive_point, dog_pos_before)

        # --- dynamic weights based on sheep status ---
        cohesion_weights = np.where(
            self.sheep_status == 'B',
            config.WEIGHT_COHESION * config.WEIGHT_COHESION_B_FACTOR,
            config.WEIGHT_COHESION
        )[:, np.newaxis]
        dog_repulsion_weights = np.where(
            self.sheep_status == 'A',
            config.WEIGHT_DOG_REPULSION * config.WEIGHT_DOG_REPULSION_A_FACTOR,
            config.WEIGHT_DOG_REPULSION
        )[:, np.newaxis]
        separation_weights = np.where(self.sheep_status == 'B', config.WEIGHT_SEPARATION * config.WEIGHT_SEPARATION_B_FACTOR, config.WEIGHT_SEPARATION)[:, np.newaxis]

        # --- sheep logic ---
        # A. cohesion (converge to the center of mass)
        cohesion = (com - self.sheep_pos) * cohesion_weights
        
        # B. avoid the shepherd (strength is inversely proportional to distance d)
        vec_from_dog = self.sheep_pos - self.dog_pos
        dist_from_dog = np.linalg.norm(vec_from_dog, axis=1, keepdims=True)
        dir_from_dog = vec_from_dog / (dist_from_dog + 1e-6)
        dog_repulsion = dir_from_dog * (dog_repulsion_weights / (dist_from_dog + 1e-6))
        
        # only react to the dog if it is within sensing range
        dog_repulsion[dist_from_dog[:, 0] > config.DOG_SENSING_RANGE] = 0
        
        # C. boundary repulsion force (prevent the sheep from sticking to the wall)
        boundary_force = np.zeros_like(self.sheep_pos)
        boundary_force[self.sheep_pos < config.BOUNDARY_MARGIN] += config.WEIGHT_BOUNDARY
        boundary_force[self.sheep_pos > config.SPACE_SIZE - config.BOUNDARY_MARGIN] -= config.WEIGHT_BOUNDARY

        # D. separation (prevent sheep from overlapping)
        diffs = self.sheep_pos[:, np.newaxis, :] - self.sheep_pos[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        separation_radius = np.where(self.sheep_status == 'B', config.SEPARATION_RADIUS_B, config.SEPARATION_RADIUS_NORMAL)[:, np.newaxis]
        mask = (dists > 0) & (dists < separation_radius)
        repulsion_forces = diffs / (dists[..., np.newaxis]**2 + 1e-6)
        repulsion_forces[~mask] = 0
        separation = np.sum(repulsion_forces, axis=1) * separation_weights

        # combine the forces and update the velocity and position
        total_force = cohesion + dog_repulsion + boundary_force + separation
        
        # Type A keeps most of its current velocity direction and only weakly accepts new force.
        force_response = np.where(self.sheep_status == 'A', 1.0 - config.INERTIA_FACTOR_A, 1.0)[:, np.newaxis]
        self.sheep_vel += total_force * force_response
        
        self.sheep_vel = self._limit_speed_array(self.sheep_vel, config.SHEEP_MAX_SPEED)
        self.sheep_pos += self.sheep_vel

        return False

    def _limit_speed(self, vel, max_speed):
        speed = np.linalg.norm(vel)
        if speed > max_speed:
            return (vel / speed) * max_speed
        return vel

    def _limit_speed_array(self, vels, max_speed):
        speeds = np.linalg.norm(vels, axis=1, keepdims=True)
        speeds_with_epsilon = speeds + config.EPSILON
        scale = np.minimum(1.0, max_speed / speeds_with_epsilon)
        return vels * scale

    def repair_sheep(self, indices):
        """
        Intervention API to repair abnormal sheep back to the 'Normal' state.
        :param indices: A single integer, a list of integers, or a NumPy array of sheep indices.
        """
        if isinstance(indices, int):
            indices = [indices]
        self.sheep_status[indices] = 'Normal'
