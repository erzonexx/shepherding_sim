import numpy as np
import config
import csv
import os
from datetime import datetime

class SwarmEnv:
    def __init__(self):
        # randomly initialize the sheep's position and velocity
        self.sheep_pos = config.SHEEP_START_CENTER + np.random.randn(config.NUM_SHEEP, 2) * config.SHEEP_START_RADIUS
        self.sheep_vel = np.zeros((config.NUM_SHEEP, 2))
        
        # initialize the shepherd
        self.dog_pos = np.copy(config.DOG_START_POS)
        self.dog_vel = np.zeros(2)

        # variables for experimental data recording
        self.current_frame = 0
        self.history_data = []

    def _record_state(self):
        # Record Dog's state
        self.history_data.append([self.current_frame, 'Dog', 0, self.dog_pos[0], self.dog_pos[1]])
        # Record each Sheep's state
        for i, pos in enumerate(self.sheep_pos):
            self.history_data.append([self.current_frame, 'Sheep', i, pos[0], pos[1]])

    def export_data(self, filename):
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join('logs', f"simulation_{timestamp}.csv")
        
        with open(export_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Agent_Type', 'Agent_ID', 'X', 'Y'])
            writer.writerows(self.history_data)

    def step(self):
        """execute one frame of physics calculation. if the sheep reaches the goal, return True"""
        self._record_state()
        self.current_frame += 1

        # 1. calculate the sheep's center of mass (Center of Mass)
        com = np.mean(self.sheep_pos, axis=0)
        
        # --- shepherd logic ---
        to_goal = config.GOAL_POS - com
        dist_to_goal = np.linalg.norm(to_goal)
        
        # success judgment: the distance between the sheep's center of mass and the goal is less than 10
        if dist_to_goal < 10:
            return True

        # the shepherd finds the "driving point" behind the sheep
        dir_to_goal = to_goal / (dist_to_goal + 1e-6)
        drive_point = com - dir_to_goal * 25  # push the sheep 25 units behind the center of mass
        
        dog_desired_vel = drive_point - self.dog_pos
        self.dog_vel = self._limit_speed(dog_desired_vel, config.DOG_MAX_SPEED)
        self.dog_pos += self.dog_vel

        # --- sheep logic ---
        # A. cohesion (converge to the center of mass)
        cohesion = (com - self.sheep_pos) * config.WEIGHT_COHESION
        
        # B. avoid the shepherd (the closer the distance, the greater the repulsion force)
        vec_from_dog = self.sheep_pos - self.dog_pos
        dist_from_dog = np.linalg.norm(vec_from_dog, axis=1, keepdims=True)
        dog_repulsion = (vec_from_dog / (dist_from_dog**2 + 1e-6)) * config.WEIGHT_DOG_REPULSION
        
        # C. boundary repulsion force (prevent the sheep from sticking to the wall)
        boundary_force = np.zeros_like(self.sheep_pos)
        boundary_force[self.sheep_pos < 5] += config.WEIGHT_BOUNDARY
        boundary_force[self.sheep_pos > config.SPACE_SIZE - 5] -= config.WEIGHT_BOUNDARY

        # combine the forces and update the velocity and position
        self.sheep_vel += cohesion + dog_repulsion + boundary_force
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