"""student_controller controller."""

import math
import numpy as np


class StudentController:
    def __init__(self, filter="ekf"):
        self._filter_type = filter

    def step(self, sensors):
        """
        Compute robot control as a function of sensors.

        Input:
        sensors: dict, contains current sensor values.

        Output:
        control_dict:   dict, contains control for "left_motor" and "right_motor"
        estimated_pose: list, contains float values representing the robot's pose,
                        (x,y,orientation).
                        The pose should be given using a right-handed coordinate
                        system: positive x is the right side of the arena, positive
                        y is the top side of the arena, theta increases as the
                        robot turns counter-clockwise.
        """
        control_dict = {"left_motor": 0.0, "right_motor": 0.0}

        # TODO: add your controllers here.
        control_dict["left_motor"] = 3.0
        control_dict["right_motor"] = 3.0

        estimated_pose = [0, 0, 0]

        return control_dict, estimated_pose
