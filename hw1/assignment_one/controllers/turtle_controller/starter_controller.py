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
    
    ## helpers: 
    def _safe_min(self, arr, ignore_zeros=True):
        """Min that optionally ignores zeros (since inf gets mapped to 0)."""
        if ignore_zeros:
            arr = [v for v in arr if v > 1e-6]
        return min(arr) if arr else float("inf")

    def _get_directional_distances(self, lidar):
        """
        Returns approximate distances in key directions in robot frame.
        Assumes 360-ish lidar where index 0 front, 90 right, 180 back, 270 left.
        If your lidar length differs, this converts degrees -> index.
        """
        n = len(lidar)

        def idx(deg):
            return int((deg % 360) * n / 360)

        front = lidar[idx(0)]
        right = lidar[idx(90)]
        left  = lidar[idx(270)]
        front_left = lidar[idx(315)]  # 45° to the left of front

        closest = self._safe_min(lidar, ignore_zeros=True)
        return front, right, left, closest, front_left



def step(self, sensors):
    """
    Compute robot control as a function of sensors.

    Input:
    sensors: dict, contains current sensor values.

    Output:
    control_dict: dict, contains control for "left_motor" and "right_motor"
    """
    control_dict = {"left_motor": 0.0, "right_motor": 0.0}

    # Print control law ONCE (prevents console spam)
    if not hasattr(self, "_printed_control_law"):
        print("control_law:", self.control_law)
        self._printed_control_law = True

    # NOTE: This check is effectively dead-code because previous_lidar_data starts as []
    # but leaving it doesn't hurt.
    if self.previous_lidar_data is None:
        num_lidar_rays = len(sensors["lidar"])
        self.previous_lidar_data = [[] for _ in range(num_lidar_rays)]

    # Read sensors
    lidar_data = sensors["lidar"]

    # Apply filtering
    filtered_lidar = self.get_filtered_lidar_reading(lidar_data)
    heading = self.get_filtered_heading(sensors["heading"])

    # Pose estimate (ok to keep even if not used)
    x, y, heading = self.compute_pose(heading, filtered_lidar)

    # Directional distances (your helper must return 5 values)
    front, right, left, closest, front_left = self._get_directional_distances(filtered_lidar)

    # --- Local tuning variables (do NOT replace TURN_THRESHOLD globally) ---
    corner_threshold = 0.15            # "panic" distance for corners
    corner_exit_threshold = 0.35       # hysteresis for exiting corner mode

    # Follow LEFT wall (counter-clockwise)
    wall_dist = left
    error = wall_dist - DESIRED_DISTANCE

    # Base forward speed and turn command
    base = 0.5 * MAX_SPEED
    turn = 0.0

    # Validate readings (0 can mean "inf mapped to 0")
    front_ok = front > 1e-6
    fl_ok = front_left > 1e-6
    left_ok = left > 1e-6

    # Corner detection: use AND to avoid triggering too early
    approaching_corner = (front_ok and fl_ok and (front < corner_threshold) and (front_left < corner_threshold))

    # Bang-bang logic (for now you’re testing bang-bang behavior even if harness sets "P")
    if self.control_law in ["bang-bang", "P"]:
        bang_turn = 0.7  # tune 0.5–1.2

        # Enter corner mode
        if approaching_corner:
            self.in_corner_turn = True

        # Corner escape mode (adds hysteresis so it doesn't flip-flop)
        if getattr(self, "in_corner_turn", False):
            # For LEFT-wall following, turning RIGHT helps round the corner
            turn = -1.0
            base = 0.2 * MAX_SPEED

            # Exit when front opens up again (hysteresis)
            if front_ok and front > corner_exit_threshold:
                self.in_corner_turn = False
        else:
            # Standard bang-bang wall following (LEFT wall)
            if error < 0:
                # too close to left wall -> steer away (turn right)
                turn = -bang_turn
            else:
                # too far from left wall -> steer toward (turn left)
                turn = +bang_turn

    # Convert (base, turn) -> wheel speeds
    left_speed = base + turn
    right_speed = base - turn

    # Clamp to motor limits
    left_speed = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
    right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))

    control_dict["left_motor"] = left_speed
    control_dict["right_motor"] = right_speed

    return control_dict
