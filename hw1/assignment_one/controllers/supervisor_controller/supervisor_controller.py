from controller import Supervisor

import numpy as np

# Create the Supervisor instance
supervisor = Supervisor()

# Get the timestep
timestep = int(supervisor.getBasicTimeStep())
t = 0

xs, ys = [], []
points = []

# Main control loop
while supervisor.step(timestep) != -1:
    # Example: Get a reference to a robot in the simulation
    robot_node = supervisor.getFromDef("MY_ROBOT")  # Replace with your robot's DEF name

    if robot_node:
        # Get the robot's position
        position = robot_node.getField("translation").getSFVec3f()
        # print(f"Robot position: {position}")
        # print(robot_node.getDevice("compass"))
        # xs.append(position[0])
        # ys.append(position[1])
        points.append([position[0], position[1]])

    t += 1
    if t >= 6000:
        break

# End of controller
print("Saving data from run.")
points = np.array(points)
np.save('points.npy', points)
