import math
import numpy as np
from controller import Robot, DistanceSensor, Motor, Compass, GPS
from controller import Supervisor
from starter_controller import StudentController


MIN_VELOCITY = -6.25
MAX_VELOCITY = 6.25
MAX_BOXES = 10


# Define the robot class
class TurtleBotController:
    def __init__(self):
        # Initialize the robot and get basic time step
        # self.robot = Robot()

        # Create the Supervisor instance
        self.robot = Supervisor()
        self.time_step = int(self.robot.getBasicTimeStep())
        self.robot_node = self.robot.getFromDef("MY_ROBOT")
        self.box_nodes = [self.robot.getFromDef(f"BOX_{d}") for d in range(MAX_BOXES)]

        # Get motors (assuming left and right motors are available)
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")

        # Set motor velocity to maximum (adjust as needed)
        self.left_motor.setPosition(float("inf"))  # Infinity means continuous rotation
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Initialize Lidar (Laser Distance Sensor)
        self.lidar = self.robot.getDevice("LDS-01")
        self.lidar.enable(self.time_step)

        # Initialize Compass (for orientation)
        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.time_step)

        # Sensor and control noise
        self._lidar_noise = 0.15
        self._detection_range = 1.0
        self._control_noise_pct = 0.0
        self._heading_noise = 0 * (math.pi / 180)
        self._odometry_noise = 0.005

        # Odometry variables
        self.prev_position = self.true_pose()

        self.reveal_correspondences = True
        self.student_controller = StudentController(filter="ekf")

    def provide_lidar(self):
        """Get lidar range image and lower detect range below default 3.5."""
        lidar_image = self.lidar.getRangeImage()
        for i in range(len(lidar_image)):
            lidar_image[i] = (
                float("inf")
                if lidar_image[i] >= self._detection_range
                else lidar_image[i]
            )
        noise = np.random.normal(0, self._lidar_noise, size=len(lidar_image))
        lidar_image += noise
        return lidar_image

    def provide_compass(self):
        """Get a noisy compass reading."""
        values = self.compass.getValues()
        heading = math.atan2(values[0], values[1])
        # noise = np.random.normal(0, self._heading_noise)
        # return heading + noise
        return heading

    def provide_odometry(self):
        """Compute a noisy odometry estimate."""
        position = self.true_pose()
        heading = self.provide_compass()
        delta_forward = (position[0] - self.prev_position[0]) ** 2 + (
            position[1] - self.prev_position[1]
        ) ** 2
        delta_forward = np.sqrt(delta_forward)
        delta_heading = np.arctan2(
            np.sin(heading - self.prev_heading),
            np.cos(heading - self.prev_heading)
        )
        self.prev_position = position
        self.prev_heading = heading
        odom_vector = np.array([delta_forward, delta_heading])
        noise = np.random.normal(0, self._odometry_noise, size=2)
        return odom_vector + noise

    def clip_control(self, control):
        """Non-linear behavior in controls."""
        if abs(control) < 0.05:
            control = 0.0
        control = min(control, max(control, MIN_VELOCITY), MAX_VELOCITY)
        return control

    def true_pose(self):
        position = self.robot_node.getField("translation").getSFVec3f()

        # ------ added for correct heading -----#
        values = self.compass.getValues()
        heading = math.atan2(values[0], values[1])

        return position[:2] + [heading]
        # --------------------------------------#

    def close_to_robot(self, pose):
        """Detect if a pose is close to robot based on a threshold."""
        if pose is None:
            return False
        robot_coord = np.array(self.true_pose())  # grab x,y
        # print(pose, robot_coord)
        distance = np.linalg.norm(robot_coord[:2] - pose[:2])
        return distance < self._detection_range, distance

    def provide_map(self):
        landmark_positions = {}
        for idx, box in enumerate(self.box_nodes):
            if box is None:
                continue
            position = box.getField("translation").getSFVec3f()
            landmark_positions[f"BOX_{idx}"] = position
        return landmark_positions

    def get_landmark_observations(self, known_correspondences=True):
        """Get a list of landmark identifiers for close landmarks."""

        close_landmarks = {}
        # close_landmarks = []
        count = 0

        for idx, box in enumerate(self.box_nodes):
            if box is None:
                continue
            position = box.getField("translation").getSFVec3f()
            close, distance = self.close_to_robot(position)
            if close:

                # get bearing
                pose = self.true_pose()
                bearing = (
                    np.arctan2(position[1] - pose[1], position[0] - pose[0]) - pose[2]
                )
                # Normalize angle
                # bearing = (bearing + math.pi) % (2 * math.pi) - math.pi

                # Add noise to observation
                distance += np.random.normal(0, 0.1)
                bearing += np.random.normal(0, 0.05)

                if known_correspondences:
                    close_landmarks[f"BOX_{idx}"] = (distance, bearing)
                else:
                    close_landmarks[f"Unknown_{count}"] = (distance, bearing)
                count += 1

        return close_landmarks

    def run(self):
        """
        The main loop that controls the robot's behavior.
        """

        arena_map = self.provide_map()
        first_step = True

        while self.robot.step(self.time_step) != -1:

            if first_step:
                self.prev_heading = self.provide_compass()
                first_step = False

            # Pack sensor values for student controller
            landmarks = self.get_landmark_observations(
                known_correspondences=self.reveal_correspondences
            )
            odometry = self.provide_odometry()

            sensors = {
                "map": arena_map,
                "observed_landmarks": landmarks,
                "odometry": odometry,
            }
            # for name in landmarks:
            #     distance, bearing = landmarks[name]
            #     print(name, distance, bearing * 180 / np.pi)

            # Get control values from student controller
            controls, robot_pose = self.student_controller.step(sensors)
            lwhl = controls.get("left_motor", 0.0)
            rwhl = controls.get("right_motor", 0.0)

            # Apply noise to control and clip to remain in bounds
            lwhl += np.random.normal(0, self._control_noise_pct * abs(lwhl))
            rwhl += np.random.normal(0, self._control_noise_pct * abs(rwhl))
            lwhl = self.clip_control(lwhl)
            rwhl = self.clip_control(rwhl)

            # Set control
            self.left_motor.setVelocity(lwhl)
            self.right_motor.setVelocity(rwhl)


# Create a controller instance and run it
controller = TurtleBotController()

controller.run()
