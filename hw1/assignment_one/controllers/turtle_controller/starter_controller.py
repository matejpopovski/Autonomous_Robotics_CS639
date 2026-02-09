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
    def __init__(self, control_law="bang-bang"):
        self.previous_lidar_data = []
        self.previous_heading_data = []
        self.previous_error = 0
        self.control_law = control_law
        self.in_corner_turn = False


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
    
    # ## helpers: 
    # def _safe_min(self, arr, ignore_zeros=True):
    #     """Min that optionally ignores zeros (since inf gets mapped to 0)."""
    #     if ignore_zeros:
    #         arr = [v for v in arr if v > 1e-6]
    #     return min(arr) if arr else float("inf")

    # def _get_directional_distances(self, lidar):
    #     """
    #     Returns approximate distances in key directions in robot frame.
    #     Assumes 360-ish lidar where index 0 front, 90 right, 180 back, 270 left.
    #     If your lidar length differs, this converts degrees -> index.
    #     """
    #     n = len(lidar)

    #     def idx(deg):
    #         return int((deg % 360) * n / 360)

    #     front = lidar[idx(0)]
    #     right = lidar[idx(90)]
    #     left  = lidar[idx(270)]
    #     front_left = lidar[idx(315)]  # 45° to the left of front

    #     closest = self._safe_min(lidar, ignore_zeros=True)
    #     return front, right, left, closest, front_left

    def _idx(self, deg, n):
        return int((deg % 360) * n / 360)

    def _dir_dists(self, lidar):
        n = len(lidar)
        front = lidar[self._idx(0, n)]
        right = lidar[self._idx(90, n)]
        left  = lidar[self._idx(270, n)]
        return front, right, left



    def step(self, sensors):
        # Expect only lidar in this HW wrapper
        lidar_readings = sensors["lidar"]
        lidar = np.array(lidar_readings, dtype=float)

        # Handle inf -> 0 (common in your setup)
        filtered = np.copy(lidar)
        filtered[np.isinf(filtered)] = 0.0

        # Directional distances
        n = len(filtered)

        def idx(deg):
            return int((deg % 360) * n / 360)

        front = filtered[idx(0)]
        right = filtered[idx(90)]
        left  = filtered[idx(270)]

        # Follow RIGHT wall at DESIRED_DISTANCE
        error = DESIRED_DISTANCE - right

        control_dict = {"left_motor": 0.0, "right_motor": 0.0}

        # Corner / collision override: if wall in front, turn left
        if front > 1e-6 and front < TURN_THRESHOLD:
            control_dict["left_motor"]  = 0.25 * MAX_SPEED
            control_dict["right_motor"] = 0.85 * MAX_SPEED
            return control_dict

        # Steering command u
        u = 0.0

        if self.control_law == "bang-bang":
            deadband = 0.02
            if error > deadband:
                u = +0.35 * MAX_SPEED
            elif error < -deadband:
                u = -0.35 * MAX_SPEED
            else:
                u = 0.0

        elif self.control_law == "P":
            u = KP * error * MAX_SPEED

        elif self.control_law == "PID":
            # Safe init in case you didn't add these in __init__
            if not hasattr(self, "integral_error"):
                self.integral_error = 0.0
            if not hasattr(self, "prev_error"):
                self.prev_error = 0.0

            dt = TIME_STEP / 1000.0
            self.integral_error += error * dt
            self.integral_error = float(np.clip(self.integral_error, -1.0, 1.0))
            derr = (error - self.prev_error) / dt
            self.prev_error = error

            u = (KP * error + KI * self.integral_error + KD * derr) * MAX_SPEED

        # Mix into wheel speeds
        base = 0.6 * MAX_SPEED
        left_speed  = base - u
        right_speed = base + u

        left_speed  = float(np.clip(left_speed,  -MAX_SPEED, MAX_SPEED))
        right_speed = float(np.clip(right_speed, -MAX_SPEED, MAX_SPEED))

        control_dict["left_motor"] = left_speed
        control_dict["right_motor"] = right_speed
        return control_dict
