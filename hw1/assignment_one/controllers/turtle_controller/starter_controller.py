"""student_controller controller."""

import numpy as np
from pathlib import Path

# Constants
TIME_STEP = 32
MAX_SPEED = 6.28
DESIRED_DISTANCE = 0.25  # Desired distance from the wall (meters)
TURN_THRESHOLD = 0.25    # Distance threshold to detect corners (meters)

# Controller gains (tune as you like)
KP = 0.10
KI = 0.01
KD = 0.02

# Debug
DEBUG_PRINT = True
DEBUG_EVERY_N_STEPS = 30  # ~ once per second


class StudentController:
    def __init__(self, control_law="bang-bang"):
        # reverse mode: move backward (toward -y initially since robot faces +y)
        self.REVERSE = True

        # small startup straight reverse to ensure it goes "down" immediately
        self._startup_steps = 0
        self._startup_steps_max = 25  # ~0.8s (25*32ms)

        # --- GUI-safe override: read control law from a text file ---
        try:
            cfg_path = Path(__file__).with_name("control_law.txt")
            print(f"[LAWCFG] looking for: {cfg_path}")
            print(f"[LAWCFG] exists? {cfg_path.exists()}")

            if cfg_path.exists():
                file_law = cfg_path.read_text(encoding="utf-8").strip()
                print(f"[LAWCFG] read: {repr(file_law)}")
                if file_law:
                    control_law = file_law
        except Exception as e:
            print(f"[LAWCFG] error reading control_law.txt: {e}")

        assert control_law in ["bang-bang", "P", "PID"], f"Unknown control_law: {control_law}"
        self.control_law = control_law

        print(f"[StudentController] Running with control law: {self.control_law}")

    def _idx_robot(self, deg, n):
        base = int((deg % 360) * n / 360)
        return (n // 2 - base + n) % n

    def step(self, sensors):
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
            return min(vals) if vals else 10.0

        # --- Sense distances (robot frame) ---
        front_min = arc_min(0, 15)
        left_min  = arc_min(270, 10)
        right_min = arc_min(90, 10)
        back_min  = arc_min(180, 15)

        # --- Startup: go straight in reverse (down/south) ---
        if self.REVERSE and self._startup_steps < self._startup_steps_max:
            self._startup_steps += 1
            v = -0.55 * MAX_SPEED
            return {"left_motor": v, "right_motor": v}

        # --- CLOCKWISE while driving backward => follow LEFT wall ---
        # (because motion direction is opposite heading)
        side = "L"
        wall_dist = left_min

        error = DESIRED_DISTANCE - wall_dist

        # --- Collision avoidance for REVERSE ---
        # When driving backward, the "front of motion" is behind the robot
        avoid_thresh = max(TURN_THRESHOLD, 0.45)
        motion_front = back_min if self.REVERSE else front_min

        if motion_front < avoid_thresh:
            # Turn so that while reversing we steer away.
            # To yaw right while reversing: left wheel less negative, right wheel more negative.
            if self.REVERSE:
                return {"left_motor": -0.10 * MAX_SPEED, "right_motor": -0.80 * MAX_SPEED}
            else:
                return {"left_motor": 0.80 * MAX_SPEED, "right_motor": 0.10 * MAX_SPEED}

        # --- Controller ---
        u = 0.0

        if self.control_law == "bang-bang":
            deadband = 0.04
            if error > deadband:
                u = +0.55 * MAX_SPEED
            elif error < -deadband:
                u = -0.55 * MAX_SPEED
            else:
                u = 0.0

        elif self.control_law == "P":
            u = KP * error * MAX_SPEED

        elif self.control_law == "PID":
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

        # --- DEBUG PRINT ---
        if DEBUG_PRINT:
            if not hasattr(self, "_dbg_i"):
                self._dbg_i = 0
            self._dbg_i += 1

            if self._dbg_i % DEBUG_EVERY_N_STEPS == 0:
                print(
                    f"[DIST] front={front_min:0.3f}  back={back_min:0.3f}  left={left_min:0.3f}  right={right_min:0.3f}  follow={side}:{wall_dist:0.3f}"
                )
                print(
                    f"[CTRL] law={self.control_law:>8}  err={error:+0.3f}  u={u:+0.3f}  reverse={self.REVERSE}"
                )

        # --- Speed scheduling ---
        base_fast = 0.55 * MAX_SPEED
        base_slow = 0.25 * MAX_SPEED
        base_fwd = base_slow if (front_min < 0.8) else base_fast

        base = -base_fwd if self.REVERSE else base_fwd

        # Differential drive mapping: omega depends on (right-left), works fine with negative base
        left = float(np.clip(base - u, -MAX_SPEED, MAX_SPEED))
        right = float(np.clip(base + u, -MAX_SPEED, MAX_SPEED))

        return {"left_motor": left, "right_motor": right}
