import numpy as np

class Detector:
    """
    a simple rule-based detector that analyzes the flock's state each frame and outputs standardized metrics and danger reports.
    """
    def __init__(self, dispersion_threshold, stagnation_frames, stagnation_threshold):
        """
        Initialize the detector.
        :param dispersion_threshold: The danger threshold for flock dispersion.
        :param stagnation_frames: The number of consecutive frames to check for "stagnation" status.
        :param stagnation_threshold: The minimum distance change considered as "progress" during stagnation.
        """
        self.dispersion_threshold = dispersion_threshold
        self.stagnation_frames = stagnation_frames
        self.stagnation_threshold = stagnation_threshold
        
        # to keep track of the distance to goal over recent frames for stagnation detection
        self.history_dist_to_goal = []

    def analyze_flock(self, sheep_pos, goal_pos):
        """
        Analyze the current frame's flock state.
        :param sheep_pos: Numpy array of all sheep's positions (NUM_SHEEP, 2).
        :param goal_pos: Numpy array of the goal position (2,).
        :return: (metrics, report)
                 metrics (dict): Core metrics for data recording.
                 report (dict): Standardized report for real-time alerts.
        """
        # --- 1. Calculate core metrics ---
        num_sheep = sheep_pos.shape[0]
        if num_sheep == 0:
            com = np.array([0.0, 0.0])
            dist_to_goal = np.linalg.norm(goal_pos - com)
            dispersion = 0.0
        else:
            # (Center of Mass)
            com = np.mean(sheep_pos, axis=0)
            # The distance from the flock's center of mass to the goal
            dist_to_goal = np.linalg.norm(goal_pos - com)
            # The dispersion of the flock (distance of the furthest individual)
            vecs_to_com = sheep_pos - com
            dists_to_com = np.linalg.norm(vecs_to_com, axis=1)
            dispersion = np.max(dists_to_com) if dists_to_com.size > 0 else 0.0

        # update the history of distance to goal for stagnation detection
        self.history_dist_to_goal.append(dist_to_goal)
        if len(self.history_dist_to_goal) > self.stagnation_frames:
            self.history_dist_to_goal.pop(0)

        # --- 2. Danger Detection Rules ---
        is_danger = False
        danger_type = "None"
        description = "Flock status is normal."

        # Rule 1: Excessive dispersion
        if dispersion > self.dispersion_threshold:
            is_danger = True
            danger_type = "flock_splitting"  # Flock splitting tendency
            description = f"Flock dispersion is too high, furthest distance reached {dispersion:.2f} meters (threshold: {self.dispersion_threshold:.2f} meters)."

        # Rule 2: Stagnation (only checked when no other dangers are detected)
        if not is_danger and len(self.history_dist_to_goal) == self.stagnation_frames:
            progress = self.history_dist_to_goal[0] - self.history_dist_to_goal[-1]
            if progress < self.stagnation_threshold:
                is_danger = True
                danger_type = "stagnation"  # Stagnation
                description = f"Flock has been stagnant for {self.stagnation_frames} frames (distance change: {progress:.2f} meters)."

        # --- 3. Output standardized interface ---
        # assemble the metrics for data recording
        metrics = {
            'dist_to_goal': round(dist_to_goal, 2),
            'dispersion': round(dispersion, 2),
            'is_danger': 1 if is_danger else 0,
        }
        
        # assemble the report for real-time alerts
        report = {
            'is_danger': is_danger,
            'danger_type': danger_type,
            'description': description
        }

        return metrics, report
