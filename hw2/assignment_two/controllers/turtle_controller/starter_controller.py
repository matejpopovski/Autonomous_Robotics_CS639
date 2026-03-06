# """student_controller controller."""
# import math
# import numpy as np

# def wrap(a):
#     return math.atan2(math.sin(a), math.cos(a))

# class StudentController:
#     def __init__(self, filter="ekf"):
#         self._filter  = filter.lower()
#         self.sr, self.sb = 0.10, 0.05       
#         self.sf, self.st = 0.04, 0.04       
#         self.mu, self.Sigma, self._ekf_ok = np.zeros(3), np.eye(3), False
#         self.N, self.particles, self.weights, self._pf_ok = 1000, None, None, False
#         self._last_est = None
#         self.xmin, self.xmax, self.ymin, self.ymax = -2.5, 2.5, -2.5, 2.5
#         self._bounds_ok = False
#         self._dx, self._dy, self._dth = 0.0, 0.0, 0.0
#         self._esc_steps, self._esc_dir = 0, 0.0
#         self._nav, self._nav_left = "fwd", np.random.randint(60, 120)
#         self._nav_turn_dir = 1.0
#         self._no_obs = 0
#         self.t = 0

#     def _bounds(self, amap):
#         if not amap or self._bounds_ok:
#             return
#         xs = [p[0] for p in amap.values()]; ys = [p[1] for p in amap.values()]
#         pad = 0.8
#         self.xmin, self.xmax = min(xs)-pad, max(xs)+pad
#         self.ymin, self.ymax = min(ys)-pad, max(ys)+pad
#         self._bounds_ok = True

#     def _predict_obs(self, x, lm):
#         dx, dy = lm[0]-x[0], lm[1]-x[1]
#         return np.array([math.sqrt(dx*dx+dy*dy), wrap(math.atan2(dy,dx)-x[2])])

#     def _known(self, obs, amap):
#         return (not obs) or (next(iter(obs)) in amap)

#     # ekf
#     def _ekf_step(self, sensors):
#         amap, obs, odom = sensors["map"], sensors["observed_landmarks"], sensors["odometry"]
#         self._bounds(amap)
#         if not self._ekf_ok:
#             cx, cy = (self.xmin+self.xmax)/2, (self.ymin+self.ymax)/2
#             self.mu = np.array([cx, cy, 0.0])
#             hw, hh  = (self.xmax-self.xmin)/2, (self.ymax-self.ymin)/2
#             self.Sigma = np.diag([hw**2, hh**2, math.pi**2])
#             self._ekf_ok = True

#         # predict
#         df, dth = float(odom[0]), float(odom[1])
#         th = self.mu[2]
#         G = np.array([[1,0,-df*math.sin(th)],[0,1,df*math.cos(th)],[0,0,1]])
#         self.mu    = np.array([self.mu[0]+df*math.cos(th), self.mu[1]+df*math.sin(th), wrap(th+dth)])
#         self.Sigma = G @ self.Sigma @ G.T + np.diag([self.sf**2, self.sf**2, self.st**2])
#         if self._bounds_ok:
#             self.mu[:2] = np.clip(self.mu[:2], [self.xmin,self.ymin], [self.xmax,self.ymax])

#         # update
#         known = self._known(obs, amap)
#         Q = np.diag([self.sr**2, self.sb**2])
#         for name, (dist, bear) in obs.items():
#             z = np.array([float(dist), float(bear)])
#             if known:
#                 lm = amap.get(name); lm = lm if lm is not None else None
#             else:
#                 lm, best = None, float('inf')
#                 for lname, pos in amap.items():
#                     inn = np.array([z[0]-self._predict_obs(self.mu,pos)[0],
#                                     wrap(z[1]-self._predict_obs(self.mu,pos)[1])])
#                     c = float(inn @ np.linalg.inv(Q) @ inn)
#                     if c < best and abs(inn[0]) < 1.0 and abs(inn[1]) < 0.5:
#                         best, lm = c, pos
#             if lm is None:
#                 continue
#             zhat = self._predict_obs(self.mu, lm)
#             y  = np.array([z[0]-zhat[0], wrap(z[1]-zhat[1])])
#             dx2, dy2 = lm[0]-self.mu[0], lm[1]-self.mu[1]
#             q, r = dx2**2+dy2**2, math.sqrt(dx2**2+dy2**2)
#             if r < 1e-9: continue
#             H = np.array([[-dx2/r,-dy2/r,0],[dy2/q,-dx2/q,-1]])
#             S = H @ self.Sigma @ H.T + Q
#             try:    K = self.Sigma @ H.T @ np.linalg.inv(S)
#             except: continue
#             self.mu    = self.mu + K @ y
#             self.mu[2] = wrap(self.mu[2])
#             self.Sigma = (np.eye(3) - K @ H) @ self.Sigma
#         return self.mu.copy()

#     # pf
#     def _pf_init(self):
#         N = self.N
#         self.particles = np.column_stack([
#             np.random.uniform(self.xmin, self.xmax, N),
#             np.random.uniform(self.ymin, self.ymax, N),
#             np.random.uniform(-math.pi, math.pi, N)])
#         self.weights = np.ones(N) / N
#         self._pf_ok  = True

#     def _pf_step(self, sensors):
#         amap, obs, odom = sensors["map"], sensors["observed_landmarks"], sensors["odometry"]
#         self._bounds(amap)
#         if not self._pf_ok: self._pf_init()

#         # predict
#         N = self.N
#         df, dth = float(odom[0]), float(odom[1])
#         nf  = np.random.normal(0, self.sf,  N)
#         nth = np.random.normal(0, self.st, N)
#         th  = self.particles[:,2]
#         self.particles[:,0] += (df+nf)*np.cos(th)
#         self.particles[:,1] += (df+nf)*np.sin(th)
#         self.particles[:,2]  = np.arctan2(np.sin(th+dth+nth), np.cos(th+dth+nth))
#         self.particles[:,0]  = np.clip(self.particles[:,0], self.xmin, self.xmax)
#         self.particles[:,1]  = np.clip(self.particles[:,1], self.ymin, self.ymax)

#         # update
#         if obs:
#             known = self._known(obs, amap)
#             px, py, pth = self.particles[:,0], self.particles[:,1], self.particles[:,2]
#             lm_xy = np.array([[v[0],v[1]] for v in amap.values()])
#             sr, sb = self.sr*3.0, self.sb*3.0 
#             log_w  = np.zeros(N)
#             for name, (dist, bear) in obs.items():
#                 zr, zb = float(dist), float(bear)
#                 if known and name in amap:
#                     lp = amap[name]
#                     dx2, dy2 = lp[0]-px, lp[1]-py
#                     pr = np.sqrt(dx2**2+dy2**2)
#                     pb = np.arctan2(np.sin(np.arctan2(dy2,dx2)-pth), np.cos(np.arctan2(dy2,dx2)-pth))
#                     log_w += -0.5*((zr-pr)**2/sr**2 + np.arctan2(np.sin(zb-pb),np.cos(zb-pb))**2/sb**2)
#                 else:
#                     dx2 = lm_xy[:,0][None,:]-px[:,None]; dy2 = lm_xy[:,1][None,:]-py[:,None]
#                     pr  = np.sqrt(dx2**2+dy2**2)
#                     pb  = np.arctan2(dy2,dx2) - pth[:,None]
#                     pb  = np.arctan2(np.sin(pb), np.cos(pb))
#                     eb  = np.arctan2(np.sin(zb-pb), np.cos(zb-pb))
#                     log_w += np.max(-0.5*((zr-pr)**2/sr**2 + eb**2/sb**2), axis=1)
#             log_w -= log_w.max()
#             w = np.exp(log_w); s = w.sum()
#             self.weights = w/s if s > 1e-300 else (self._pf_init() or self.weights)

#         neff = 1.0 / float(np.sum(self.weights**2))
#         if neff < N * 0.40:
#             pos = (np.arange(N) + np.random.uniform()) / N
#             cum = np.cumsum(self.weights); i = j = 0
#             idx = np.zeros(N, int)
#             while i < N:
#                 if pos[i] < cum[j]: idx[i]=j; i+=1
#                 else: j+=1
#             self.particles = self.particles[idx]
#             self.weights   = np.ones(N)/N
#             nr  = max(5, int(N*0.02))
#             ri  = np.random.choice(N, nr, replace=False)
#             self.particles[ri] = np.column_stack([
#                 np.random.uniform(self.xmin, self.xmax, nr),
#                 np.random.uniform(self.ymin, self.ymax, nr),
#                 np.random.uniform(-math.pi, math.pi, nr)])

#         ws, pts = self.weights, self.particles
#         best_score, best_mask = -1.0, None
#         for idx in np.argsort(ws)[-5:][::-1]:
#             seed = pts[idx]
#             mask = np.sqrt((pts[:,0]-seed[0])**2+(pts[:,1]-seed[1])**2) < 0.8
#             cw   = float(ws[mask].sum())
#             cont = math.exp(-0.5*(math.sqrt((seed[0]-self._last_est[0])**2+(seed[1]-self._last_est[1])**2))**2) \
#                    if self._last_est is not None else 1.0
#             if cw*cont > best_score:
#                 best_score, best_mask = cw*cont, mask
#         if best_mask is None or best_mask.sum() < max(10, int(0.05*N)):
#             best_mask = np.ones(N, bool)
#         cw = ws[best_mask]; cw /= cw.sum(); cp = pts[best_mask]
#         result = np.array([float(np.sum(cw*cp[:,0])), float(np.sum(cw*cp[:,1])),
#                            math.atan2(float(np.sum(cw*np.sin(cp[:,2]))), float(np.sum(cw*np.cos(cp[:,2]))))])
#         self._last_est = result
#         return result

#     def _control(self, sensors):
#         fwd = spin = 3.5
#         odom = sensors["odometry"]
#         df, dth = float(odom[0]), float(odom[1])
#         self._dx  += df*math.cos(self._dth); self._dy += df*math.sin(self._dth)
#         self._dth  = wrap(self._dth+dth)
#         if self._bounds_ok:
#             self._dx = float(np.clip(self._dx, self.xmin, self.xmax))
#             self._dy = float(np.clip(self._dy, self.ymin, self.ymax))

#         amap, obs = sensors["map"], sensors["observed_landmarks"]
#         self._no_obs = 0 if obs else self._no_obs+1

#         m = 0.6
#         near_wall = self._bounds_ok and (
#             self._dx<self.xmin+m or self._dx>self.xmax-m or
#             self._dy<self.ymin+m or self._dy>self.ymax-m)
        
#         near_box, esc_dir = False, 0.0
#         for pos in amap.values():
#             dx2, dy2 = self._dx-pos[0], self._dy-pos[1]
#             if math.sqrt(dx2*dx2+dy2*dy2) < 0.55:
#                 near_box, esc_dir = True, math.atan2(dy2, dx2); break

#         def steer(desired):
#             err = wrap(desired - self._dth)
#             if abs(err) > 0.15:
#                 return {"left_motor": -spin if err>0 else spin,
#                         "right_motor":  spin if err>0 else -spin}
#             return {"left_motor": fwd, "right_motor": fwd}

#         if near_wall or near_box or self._esc_steps > 0:
#             if (near_wall or near_box) and self._esc_steps <= 0:
#                 if near_box:
#                     self._esc_dir = esc_dir + np.random.uniform(-0.5, 0.5)
#                 else:
#                     cx, cy = (self.xmin+self.xmax)/2, (self.ymin+self.ymax)/2
#                     self._esc_dir = math.atan2(cy-self._dy, cx-self._dx) + np.random.uniform(-0.4, 0.4)
#                 self._esc_steps = 70
#             self._esc_steps -= 1
#             return steer(self._esc_dir)

#         if self._no_obs > 150 and amap:
#             best_d, best_pos = float('inf'), None
#             for pos in amap.values():
#                 d = math.sqrt((pos[0]-self._dx)**2+(pos[1]-self._dy)**2)
#                 if d < best_d: best_d, best_pos = d, pos
#             if best_pos:
#                 return steer(math.atan2(best_pos[1]-self._dy, best_pos[0]-self._dx))

#         self._nav_left -= 1
#         if self._nav_left <= 0:
#             if self._nav == "fwd":
#                 self._nav, self._nav_left = "turn", np.random.randint(10, 40)
#                 self._nav_turn_dir = float(np.random.choice([-1,1]))
#             else:
#                 self._nav, self._nav_left = "fwd", np.random.randint(30, 80)
#         if self._nav == "fwd":
#             return {"left_motor": fwd, "right_motor": fwd}
#         d = self._nav_turn_dir
#         return {"left_motor": -spin*d, "right_motor": spin*d}

#     def step(self, sensors):
#         self.t += 1
#         try:
#             est = self._ekf_step(sensors) if self._filter == "ekf" else self._pf_step(sensors)
#             ctrl = self._control(sensors)
#             if self.t % 30 == 0:
#                 ex, ey, eth = est
#                 obs = sensors["observed_landmarks"]
#                 if self._filter == "ekf":
#                     sx,sy = math.sqrt(self.Sigma[0,0]), math.sqrt(self.Sigma[1,1])
#                     sth   = math.sqrt(self.Sigma[2,2])
#                     # print(f"[t={self.t:04d}][EKF] est=({ex:+.3f},{ey:+.3f},{math.degrees(eth):+.1f}d)"
#                     #       f" std=({sx:.3f},{sy:.3f},{math.degrees(sth):.1f}d) lm={len(obs)}")
#                 else:
#                     neff = 1.0/float(np.sum(self.weights**2))
#                     # print(f"[t={self.t:04d}][PF]  est=({ex:+.3f},{ey:+.3f},{math.degrees(eth):+.1f}d)"
#                     #       f" Neff={neff:.0f}/{self.N} lm={len(obs)}")
#             return ctrl, [float(est[0]), float(est[1]), float(est[2])]
#         except Exception as e:
#             import traceback; traceback.print_exc()
#             return {"left_motor": 0.0, "right_motor": 0.0}, [0.0, 0.0, 0.0]


"""student_controller controller."""

import math
import numpy as np


# ─────────────────────────────────────────────
# Noise / model constants (inferred from turtle_controller.py)
# ─────────────────────────────────────────────
ODOMETRY_NOISE_STD  = 0.005   # std of odometry noise (forward & heading)
OBS_DIST_STD        = 0.1     # std of distance observation noise
OBS_BEAR_STD        = 0.05    # std of bearing observation noise

# Observation covariance
Q = np.diag([OBS_DIST_STD**2, OBS_BEAR_STD**2])

# Motion model covariance (for EKF)
R = np.diag([ODOMETRY_NOISE_STD**2, ODOMETRY_NOISE_STD**2, ODOMETRY_NOISE_STD**2])


def normalize_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


# ─────────────────────────────────────────────
# EKF Localizer
# ─────────────────────────────────────────────
class EKFLocalizer:
    """
    Extended Kalman Filter for robot localization.

    State: [x, y, theta]

    Motion model (odometry):
        x'     = x + delta_forward * cos(theta)
        y'     = y + delta_forward * sin(theta)
        theta' = theta + delta_heading

    Observation model for landmark lm at (lx, ly):
        expected_distance = sqrt((lx-x)^2 + (ly-y)^2)
        expected_bearing  = atan2(ly-y, lx-x) - theta
    """

    def __init__(self, init_pose=None, init_cov=None):
        if init_pose is None:
            init_pose = np.array([0.0, 0.0, 0.0])
        if init_cov is None:
            init_cov = np.eye(3) * 1.0

        self.mu    = np.array(init_pose, dtype=float)
        self.Sigma = np.array(init_cov,  dtype=float)

    # ── predict ──────────────────────────────
    def predict(self, odometry):
        """odometry: [delta_forward, delta_heading]"""
        df, dh = odometry
        theta = self.mu[2]

        # State update
        self.mu[0] += df * math.cos(theta)
        self.mu[1] += df * math.sin(theta)
        self.mu[2]  = normalize_angle(self.mu[2] + dh)

        # Jacobian of motion model w.r.t. state
        G = np.eye(3)
        G[0, 2] = -df * math.sin(theta)
        G[1, 2] =  df * math.cos(theta)

        # Jacobian of motion model w.r.t. control noise [df, dh]
        V = np.zeros((3, 2))
        V[0, 0] = math.cos(theta)
        V[1, 0] = math.sin(theta)
        V[2, 1] = 1.0

        M = np.diag([ODOMETRY_NOISE_STD**2, ODOMETRY_NOISE_STD**2])

        self.Sigma = G @ self.Sigma @ G.T + V @ M @ V.T

    # ── update with a single landmark ────────
    def update(self, distance, bearing, lm_pos):
        lx, ly = float(lm_pos[0]), float(lm_pos[1])
        x, y, theta = self.mu

        dx = lx - x
        dy = ly - y
        q  = dx**2 + dy**2
        sq = math.sqrt(q + 1e-9)

        z_hat = np.array([sq,
                          normalize_angle(math.atan2(dy, dx) - theta)])

        z     = np.array([distance, bearing])
        innov = np.array([z[0] - z_hat[0],
                          normalize_angle(z[1] - z_hat[1])])

        # Jacobian of observation model w.r.t. state
        H = np.array([
            [-dx/sq, -dy/sq,  0.0],
            [ dy/q,  -dx/q,  -1.0]
        ])

        S = H @ self.Sigma @ H.T + Q
        K = self.Sigma @ H.T @ np.linalg.inv(S)

        self.mu    = self.mu + K @ innov
        self.mu[2] = normalize_angle(self.mu[2])
        self.Sigma = (np.eye(3) - K @ H) @ self.Sigma

    def get_pose(self):
        return self.mu.copy()

    # ── data association (nearest neighbour) ─
    @staticmethod
    def nearest_landmark(distance, bearing, robot_pose, arena_map):
        """
        For unknown correspondences: pick landmark whose predicted observation
        is closest (in L2 of obs-space) to the actual observation.
        """
        x, y, theta = robot_pose
        best_key  = None
        best_err  = float('inf')
        for key, pos in arena_map.items():
            lx, ly = float(pos[0]), float(pos[1])
            dx = lx - x;  dy = ly - y
            sq = math.sqrt(dx**2 + dy**2 + 1e-9)
            pred_d = sq
            pred_b = normalize_angle(math.atan2(dy, dx) - theta)
            err = math.sqrt(((distance - pred_d)/OBS_DIST_STD)**2 +
                            (normalize_angle(bearing - pred_b)/OBS_BEAR_STD)**2)
            if err < best_err:
                best_err = err
                best_key = key
        return best_key, arena_map[best_key]


# ─────────────────────────────────────────────
# Particle Filter Localizer
# ─────────────────────────────────────────────
class ParticleFilter:
    """
    Particle Filter for robot localization.
    Supports kidnap recovery via random particle injection.

    Each particle: [x, y, theta]
    """

    ARENA_HALF_X = 1.6
    ARENA_HALF_Y = 1.1

    def __init__(self, n_particles=500, init_pose=None):
        self.N = n_particles

        if init_pose is not None:
            self.particles = np.zeros((self.N, 3))
            self.particles[:, 0] = init_pose[0] + np.random.normal(0, 0.2, self.N)
            self.particles[:, 1] = init_pose[1] + np.random.normal(0, 0.2, self.N)
            self.particles[:, 2] = init_pose[2] + np.random.normal(0, 0.2, self.N)
        else:
            self.particles = self._random_particles(self.N)

        self.weights = np.ones(self.N) / self.N

    def _random_particles(self, n):
        p = np.zeros((n, 3))
        p[:, 0] = np.random.uniform(-self.ARENA_HALF_X, self.ARENA_HALF_X, n)
        p[:, 1] = np.random.uniform(-self.ARENA_HALF_Y, self.ARENA_HALF_Y, n)
        p[:, 2] = np.random.uniform(-math.pi, math.pi, n)
        return p

    # ── predict ──────────────────────────────
    def predict(self, odometry):
        df, dh = odometry
        # Add process noise scaled slightly with motion magnitude
        noise_std_df = ODOMETRY_NOISE_STD + abs(df) * 0.05
        noise_std_dh = ODOMETRY_NOISE_STD + abs(dh) * 0.05
        noise_df = np.random.normal(0, noise_std_df, self.N)
        noise_dh = np.random.normal(0, noise_std_dh, self.N)
        theta = self.particles[:, 2]
        self.particles[:, 0] += (df + noise_df) * np.cos(theta)
        self.particles[:, 1] += (df + noise_df) * np.sin(theta)
        self.particles[:, 2]  = (theta + dh + noise_dh + math.pi) % (2*math.pi) - math.pi

    # ── update weights ────────────────────────
    def update(self, observations, arena_map, known_correspondences):
        if len(observations) == 0:
            return

        log_w = np.zeros(self.N)
        landmark_list = list(arena_map.items())   # [(name, pos), ...]

        for obs_name, (obs_d, obs_b) in observations.items():
            if known_correspondences and obs_name in arena_map:
                # Single known landmark
                lm = arena_map[obs_name]
                lx, ly = float(lm[0]), float(lm[1])
                dx = self.particles[:, 0] - lx   # shape (N,)  – note: lx - px below
                dy = self.particles[:, 1] - ly
                # Negate: we want (lx-px)
                dx = -dx;  dy = -dy
                pred_d = np.sqrt(dx**2 + dy**2 + 1e-9)
                pred_b = np.arctan2(dy, dx) - self.particles[:, 2]
                pred_b = (pred_b + math.pi) % (2*math.pi) - math.pi
                err_d  = obs_d - pred_d
                err_b  = obs_b - pred_b
                err_b  = (err_b + math.pi) % (2*math.pi) - math.pi
                log_w += -0.5*(err_d/OBS_DIST_STD)**2 - 0.5*(err_b/OBS_BEAR_STD)**2
            else:
                # Unknown correspondences: for each particle pick best-matching landmark
                # (vectorised over particles, loop over landmarks)
                best_ll = np.full(self.N, -np.inf)
                for _, lm in landmark_list:
                    lx, ly = float(lm[0]), float(lm[1])
                    dx = lx - self.particles[:, 0]
                    dy = ly - self.particles[:, 1]
                    pred_d = np.sqrt(dx**2 + dy**2 + 1e-9)
                    pred_b = np.arctan2(dy, dx) - self.particles[:, 2]
                    pred_b = (pred_b + math.pi) % (2*math.pi) - math.pi
                    err_d  = obs_d - pred_d
                    err_b  = obs_b - pred_b
                    err_b  = (err_b + math.pi) % (2*math.pi) - math.pi
                    ll = -0.5*(err_d/OBS_DIST_STD)**2 - 0.5*(err_b/OBS_BEAR_STD)**2
                    best_ll = np.maximum(best_ll, ll)
                log_w += best_ll

        # Numerically stable weight update
        log_w -= log_w.max()
        self.weights = np.exp(log_w)
        total = self.weights.sum()
        if total > 0:
            self.weights /= total
        else:
            self.weights = np.ones(self.N) / self.N

    # ── low-variance resampling ───────────────
    def resample(self, inject_random_fraction=0.05):
        N = self.N
        n_random   = max(1, int(N * inject_random_fraction))
        n_resample = N - n_random

        # Low-variance (systematic) resampling
        indices = np.zeros(n_resample, dtype=int)
        cumsum  = np.cumsum(self.weights)
        r = np.random.uniform(0, 1.0/n_resample)
        i = 0
        for m in range(n_resample):
            U = r + m / n_resample
            while U > cumsum[i] and i < N-1:
                i += 1
            indices[m] = i

        new_particles = np.vstack([
            self.particles[indices],
            self._random_particles(n_random)
        ])
        self.particles = new_particles
        self.weights   = np.ones(N) / N

    # ── pose estimate ─────────────────────────
    def get_pose(self):
        """Weighted mean (circular mean for theta)."""
        w = self.weights
        x     = np.sum(w * self.particles[:, 0])
        y     = np.sum(w * self.particles[:, 1])
        sin_t = np.sum(w * np.sin(self.particles[:, 2]))
        cos_t = np.sum(w * np.cos(self.particles[:, 2]))
        theta = math.atan2(sin_t, cos_t)
        return np.array([x, y, theta])

    # ── kidnap detection ─────────────────────
    def effective_sample_size(self):
        return 1.0 / (np.sum(self.weights**2) + 1e-12)

    def is_kidnapped(self, threshold_fraction=0.05):
        """Low ESS → filter has collapsed → likely kidnapped."""
        return self.effective_sample_size() < threshold_fraction * self.N


# ─────────────────────────────────────────────
# Simple exploration / wandering controller
# ─────────────────────────────────────────────
class WanderController:
    """Drives the robot around to observe different landmarks."""
    def __init__(self):
        self._step    = 0
        self._phase_len = 100   # steps per phase

    def get_control(self):
        phase = (self._step // self._phase_len) % 4
        self._step += 1
        if phase == 0:
            return {"left_motor": 3.0, "right_motor": 3.0}   # straight
        elif phase == 1:
            return {"left_motor": -2.0, "right_motor": 2.0}  # turn left
        elif phase == 2:
            return {"left_motor": 3.0, "right_motor": 3.0}   # straight
        else:
            return {"left_motor": 2.0, "right_motor": -2.0}  # turn right


# ─────────────────────────────────────────────
# Student Controller  (main entry point)
# ─────────────────────────────────────────────
class StudentController:
    def __init__(self, filter="ekf"):
        self._filter_type = filter
        self._initialised  = False
        self._wander       = WanderController()
        self._ekf          = None
        self._pf           = None

    # ── lazy init ───────────────────────────
    def _init_filters(self):
        init_pose = np.array([0.0, 0.0, 0.0])
        init_cov  = np.diag([0.5, 0.5, 0.5])
        self._ekf = EKFLocalizer(init_pose=init_pose, init_cov=init_cov)
        self._pf  = ParticleFilter(n_particles=500)
        self._initialised = True

    # ── main step ───────────────────────────
    def step(self, sensors):
        """
        Input:
            sensors: dict with keys:
                "map"               – { landmark_id: (x, y, z) }
                "observed_landmarks"– { landmark_id or Unknown_N: (dist, bearing) }
                "odometry"          – np.array([delta_forward, delta_heading])

        Output:
            control_dict:   {"left_motor": float, "right_motor": float}
            estimated_pose: [x, y, theta]  (right-handed, CCW positive)
        """
        arena_map  = sensors["map"]
        landmarks  = sensors["observed_landmarks"]
        odometry   = sensors["odometry"]

        if not self._initialised:
            self._init_filters()

        # Determine if correspondences are known
        known = all(not k.startswith("Unknown") for k in landmarks)

        # Run chosen filter
        if self._filter_type == "ekf":
            estimated_pose = self._step_ekf(odometry, landmarks, arena_map, known)
        else:
            estimated_pose = self._step_pf(odometry, landmarks, arena_map, known)

        control_dict = self._wander.get_control()
        return control_dict, list(estimated_pose)

    # ── EKF step ────────────────────────────
    def _step_ekf(self, odometry, landmarks, arena_map, known_correspondences):
        ekf = self._ekf
        ekf.predict(odometry)

        for obs_name, (dist, bearing) in landmarks.items():
            if known_correspondences:
                if obs_name in arena_map:
                    ekf.update(dist, bearing, arena_map[obs_name])
            else:
                _, lm_pos = EKFLocalizer.nearest_landmark(
                    dist, bearing, ekf.get_pose(), arena_map
                )
                ekf.update(dist, bearing, lm_pos)

        return ekf.get_pose()

    # ── Particle Filter step ─────────────────
    def _step_pf(self, odometry, landmarks, arena_map, known_correspondences):
        pf = self._pf
        pf.predict(odometry)
        pf.update(landmarks, arena_map, known_correspondences)

        # Kidnap recovery: inject many random particles if filter has collapsed
        if pf.is_kidnapped(threshold_fraction=0.05):
            pf.resample(inject_random_fraction=0.5)
        else:
            pf.resample(inject_random_fraction=0.05)

        return pf.get_pose()