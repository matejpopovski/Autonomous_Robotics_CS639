"""student_controller controller."""

import math
import numpy as np

# Constants
TIME_STEP = 32
MAX_SPEED = 6.28
DESIRED_DISTANCE = 0.25  # Desired distance from the wall (meters)
TURN_THRESHOLD = 0.25  # Distance threshold to detect corners (meters)
KP = 0.1  # Proportional gain for wall-following
KI = 0.0  # Integral gain for wall-following
KD = 0.0  # Derivative gain for wall-following
FILTER_WINDOW_SIZE = 1  # Size of the moving average filter window
ARENA_WIDTH = 3
ARENA_HEIGHT = 5


class StudentController:
    def __init__(self, control_law="P"):
        self.previous_lidar_data = []
        self.previous_heading_data = []
        self.previous_error = 0
        self.control_law = control_law

    def get_filtered_lidar_reading(self, lidar_data):
        """Get the filtered lidar reading at a specific angle."""
        self.previous_lidar_data.append([0 if math.isinf(x) else x for x in lidar_data])
        if len(self.previous_lidar_data) > FILTER_WINDOW_SIZE:
            self.previous_lidar_data.pop(0)
        previous_lidar_data = np.mean(self.previous_lidar_data, axis=0)
        return previous_lidar_data

    def get_filtered_heading(self, heading_reading):
        """Filter noisy heading readings over time."""
        self.previous_heading_data.append(heading_reading)
        if len(self.previous_heading_data) > FILTER_WINDOW_SIZE:
            self.previous_heading_data.pop(0)
        return np.mean(self.previous_heading_data)

    def compute_pose(self, heading, lidar):
        """Calculate robot's pose BETTER in arena."""
        right_wall_angle = (0 - heading) % (2 * np.pi)
        top_wall_angle = (right_wall_angle + np.pi / 2) % (2 * np.pi)
        left_wall_angle = (right_wall_angle + np.pi) % (2 * np.pi)
        bottom_wall_angle = (right_wall_angle - np.pi / 2) % (2 * np.pi)

        right_wall_angle *= 180 / np.pi
        top_wall_angle *= 180 / np.pi
        left_wall_angle *= 180 / np.pi
        bottom_wall_angle *= 180 / np.pi

        right_wall_index = int(len(lidar) * right_wall_angle / 360)
        top_wall_index = int(len(lidar) * top_wall_angle / 360)
        left_wall_index = int(len(lidar) * left_wall_angle / 360)
        bottom_wall_index = int(len(lidar) * bottom_wall_angle / 360)

        right_wall_index = (180 - right_wall_index + 360) % 360
        top_wall_index = (180 - top_wall_index + 360) % 360
        left_wall_index = (180 - left_wall_index + 360) % 360
        bottom_wall_index = (180 - bottom_wall_index + 360) % 360

        right_wall_dist = lidar[right_wall_index]
        top_wall_dist = lidar[top_wall_index]
        left_wall_dist = lidar[left_wall_index]
        bottom_wall_dist = lidar[bottom_wall_index]

        # Use the closest two walls to estimate the position
        if right_wall_dist < left_wall_dist:
            x = ARENA_WIDTH / 2 - right_wall_dist
        else:
            x = left_wall_dist - ARENA_WIDTH / 2
        if top_wall_dist > bottom_wall_dist:
            y = ARENA_HEIGHT / 2 - top_wall_dist
        else:
            y = bottom_wall_dist - ARENA_HEIGHT / 2

        return x, y, heading

    def step(self, sensors):
        """
        Compute robot control as a function of sensors.

        Input:
        sensors: dict, contains current sensor values.

        Output:
        control_dict: dict, contains control for "left_motor" and "right_motor"
        """
        control_dict = {"left_motor": 0.0, "right_motor": 0.0}

        # Initial code for sensing noisy observations and inferring robot pose.

        # Initialize data structures for filtering
        if self.previous_lidar_data is None:
            num_lidar_rays = len(sensors["lidar"])
            self.previous_lidar_data = [
                [] for _ in range(num_lidar_rays)
            ]  # Stores historical lidar data

        # # Main control loop
        lidar_data = sensors["lidar"]

        # Apply noise filtering
        filtered_lidar = self.get_filtered_lidar_reading(lidar_data)
        heading = self.get_filtered_heading(sensors["heading"])

        # Get x, y coordinates and heading in radians
        x, y, heading = self.compute_pose(heading, filtered_lidar)

        # TODO: add your controllers here.
        control_dict["left_motor"] = 0.5 * MAX_SPEED
        control_dict["right_motor"] = 0.5 * MAX_SPEED

        return control_dict
