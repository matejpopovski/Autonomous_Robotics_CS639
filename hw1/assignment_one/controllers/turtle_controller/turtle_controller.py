import math
import numpy as np
from controller import Robot, DistanceSensor, Motor, Compass, GPS

from starter_controller import StudentController


MIN_VELOCITY = -6.25
MAX_VELOCITY = 6.25


# Define the robot class
class TurtleBotController:
    def __init__(self):
        # Initialize the robot and get basic time step
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())

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
        # lidar_values = self.lidar.getRangeImage()
        # print(lidar_values)

        # Initialize Compass (for orientation)
        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.time_step)

        # Sensor and control noise
        self._lidar_noise = 0.0
        self._control_noise_pct = 0.0
        self._heading_noise = 0 * (math.pi / 180)

        self.student_controller = StudentController(control_law="P")

    def provide_lidar(self):
        lidar_image = self.lidar.getRangeImage()
        noise = np.random.normal(0, self._lidar_noise, size=len(lidar_image))
        lidar_image += noise
        return lidar_image

    def provide_compass(self):
        values = self.compass.getValues()
        heading = math.atan2(values[0], values[1])
        noise = np.random.normal(0, self._heading_noise)
        return heading + noise

    def clip_control(self, control):
        """Non-linear behavior in controls."""
        if abs(control) < 0.05:
            control = 0.0
        control = min(control, max(control, MIN_VELOCITY), MAX_VELOCITY)
        return control

    def run(self):
        """
        The main loop that controls the robot's behavior.
        """

        # get sensor values
        while self.robot.step(self.time_step) != -1:
            # Pack sensor values for student controller
            sensors = {"lidar": self.provide_lidar(), "heading": self.provide_compass()}

            # Get control values from student controller
            controls = self.student_controller.step(sensors)
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
