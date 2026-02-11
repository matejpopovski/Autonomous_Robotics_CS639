"""student_controller controller."""

import math
import numpy as np

# Constants
TIME_STEP = 32
MAX_SPEED = 6.28
DESIRED_DISTANCE = 0.25  # Desired distance from the wall (meters)
TURN_THRESHOLD = 0.25  # Distance threshold to detect corners (meters)
KP = 0.9  # Proportional gain for wall-following
KI = 0.001  # Integral gain for wall-following
KD = 1.5  # Derivative gain for wall-following
FILTER_WINDOW_SIZE = 1  # Size of the moving average filter window
ARENA_WIDTH = 3
ARENA_HEIGHT = 5


class StudentController:
    def __init__(self, control_law="P"):
        self.previous_lidar_data = []
        self.previous_heading_data = []
        self.previous_error = 0
        self.in_corner_turn = False
        self.control_law = control_law

        # Lidar robustness
        self.last_left_min = None
        self.last_right_min = None
        self.last_front_min = None
        self.follow_side = None

        # Controller state
        self.err_f = 0.0
        self.integral_error = 0.0
        self.prev_error = 0.0
        self._dbg_i = 0

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
    
    def _idx_robot(self, deg, n):
        """
        Robot frame degrees:
          0 = front
          90 = right
          180 = back
          270 = left
        """
        base = int((deg % 360) * n / 360)
        return (n // 2 + base) % n

    def step(self, sensors):
        """
        Compute robot control as a function of sensors.

        Input:
        sensors: dict, contains current sensor values.

        Output:
        control_dict: dict, contains control for "left_motor" and "right_motor"
        """

        lidar = np.array(sensors["lidar"], dtype=float)
        lidar[np.isinf(lidar)] = 0.0
        n = len(lidar)

        def idx(deg):
            return self._idx_robot(deg, n)

        def arc_min(center_deg, half_width):
            vals = []
            for d in range(center_deg - half_width, center_deg + half_width + 1):
                v = lidar[idx(d)]
                if v > 1e-6:
                    vals.append(v)
            return min(vals) if vals else None

        front_raw = arc_min(0, 15)
        left_raw = arc_min(270, 10)
        right_raw = arc_min(90, 10)

        front_min = front_raw if front_raw is not None else (self.last_front_min if self.last_front_min is not None else 10.0)
        left_min = left_raw if left_raw is not None else (self.last_left_min if self.last_left_min is not None else 10.0)
        right_min = right_raw if right_raw is not None else (self.last_right_min if self.last_right_min is not None else 10.0)

        self.last_front_min = front_min
        self.last_left_min = left_min
        self.last_right_min = right_min

        if self.follow_side is None:
            self.follow_side = "L" if left_min <= right_min else "R"
        else:
            margin = 0.10
            if self.follow_side == "R" and left_min + margin < right_min:
                self.follow_side = "L"
            elif self.follow_side == "L" and right_min + margin < left_min:
                self.follow_side = "R"

        side = self.follow_side
        wall_dist = right_min if side == "R" else left_min

        if front_min < 0.50:
            if side == "R":
                return {"left_motor": 0.20 * MAX_SPEED, "right_motor": 0.85 * MAX_SPEED}  # turn left
            else:
                return {"left_motor": 0.85 * MAX_SPEED, "right_motor": 0.20 * MAX_SPEED}  # turn right

        error = DESIRED_DISTANCE - wall_dist

        alpha = 0.2
        self.err_f = (1 - alpha) * self.err_f + alpha * error
        e = self.err_f

        side_sign = +1.0 if side == "R" else -1.0

        if self.control_law == "bang-bang":
            deadband = 0.04
            if e > deadband:
                u = +0.55 * MAX_SPEED
            elif e < -deadband:
                u = -0.55 * MAX_SPEED
            else:
                u = 0.0
            u *= side_sign

        elif self.control_law == "P":
            u = side_sign * KP * e * MAX_SPEED

        else:  # PID
            dt = TIME_STEP / 1000.0

            self.integral_error += e * dt
            self.integral_error = float(np.clip(self.integral_error, -1.0, 1.0))

            derr = (e - self.prev_error) / dt
            self.prev_error = e

            u = side_sign * (KP * e + KI * self.integral_error + KD * derr) * MAX_SPEED

        base_fast = 0.55 * MAX_SPEED
        base_slow = 0.25 * MAX_SPEED
        base = base_slow if front_min < 0.8 else base_fast

        left = float(np.clip(base - u, -MAX_SPEED, MAX_SPEED))
        right = float(np.clip(base + u, -MAX_SPEED, MAX_SPEED))

        return {"left_motor": left, "right_motor": right}
