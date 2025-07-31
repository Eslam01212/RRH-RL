import os, random, math, cv2, torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict as SpaceDict
from gymnasium.spaces import Box, Dict

from controller import Supervisor, Motor, GPS, Compass, Receiver, Emitter, Lidar

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib  import RecurrentPPO
from gymnasium import Env
import scipy.ndimage

# ----------------------------------------------------------------------
# Global constants (robot-independent)
# ----------------------------------------------------------------------
TIME_STEP     = 32                 # [ms] Webots basic timestep
RHO_MAX    = 0.598                  # [m/s] after scaling
ALPHA_MAX   = 3.63                  # [rad/s] after scaling
WHEEL_RADIUS = 0.09525  # meters
WHEELBASE = 0.33        # meters
MAX_SPEED = 6.28

SEED          = 42
_goal_reached = 1
LIDAR_ANGLES  = 9
MAX_DIST      = 5
max_steps     = 100000


CM_SIZE       = 40
map_dim       = (200, 200)
map_res       = .1
map_num_rays = 180

_near_obstacle = .5
lamda_progress   = 100
lamda_lidar      = -1/50
lamda_heading    = -1/40
lamda_ang        = -1/10
lamda_goal       = 10
lamda_obs        = -10

random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)
# ======================================================================
# Environment definition
# ======================================================================
class HarshEnv(Env):
    metadata = {"render_modes": []}

    def __init__(self, robot, seed: int = SEED, train: bool = True, showMap: bool = False, BP: bool = False, HORIZON:int = 0 ):
        super().__init__()
        self.horizon = HORIZON
        self.train = train
        self.showMap = showMap
        self.BP = BP

        # === Webots devices ===
        self.robot = robot
        self.robot_node = self.robot.getSelf()

        self.gps = self.robot.getDevice("gps")
        self.gps.enable(TIME_STEP)

        self.compass = self.robot.getDevice("compass")
        self.compass.enable(TIME_STEP)

        self.receiver = self.robot.getDevice("receiver")
        self.receiver.enable(TIME_STEP)
        self.receiver.setChannel(1)

        self.emitter = self.robot.getDevice("emitter")
        self.emitter.setChannel(2)

        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(TIME_STEP)

    
        # LIDAR parameters
        self.num_rays2 = int(self.lidar.getHorizontalResolution())
        self.num_rays = LIDAR_ANGLES
        self.fov = float(self.lidar.getFov())
        self.max_range = float(self.lidar.getMaxRange())
        self.ang_per_ray = self.fov / (self.num_rays - 1)

        # Wheels
        wheel_names = ['front left wheel', 'front right wheel',
                       'back left wheel', 'back right wheel']
        self.wheels = [self.robot.getDevice(n) for n in wheel_names]
        for w in self.wheels:
            w.setPosition(float('inf'))
            w.setVelocity(0.0)
            w.enableTorqueFeedback(TIME_STEP)

        # Global map metadata
        self.map_res = map_res
        self.map_ang_per_ray = self.fov / (map_num_rays - 1)

        # === Gym spaces ===
        act_dim = 2 * self.horizon
        flat_dim = LIDAR_ANGLES + 1 + 1 + 2 * (self.horizon - 1)

        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(act_dim,), dtype=np.float32)

        self.observation_space = Dict({
            "flat": Box(low=-1, high=1, shape=(flat_dim,), dtype=np.float32),
            "costmap": Box(low=-1, high=1, shape=(CM_SIZE, CM_SIZE), dtype=np.float32)
        })

        self.max_steps = max_steps
        self.human_position = [0.0, 0.0]

        self.seed(seed)  # Seed environment-level RNGs

    # ------------------------------------------------------------------
    def _init_episode_vars(self):
        self.steps = 0
        self.energy_consumed = 0.0
        self.total_distance = 0.0
        self.total_turn = 0.0
        self.prev_pos = np.array([0.0, 0.0])
        self.yaw_prev = 0.0
        self.linear_vel_log = []
        self.angular_vel_log = []

        self.torque_logs = [[] for _ in range(4)]
        self.velocity_logs = [[] for _ in range(4)]
        self.power_logs = [[] for _ in range(4)]
        
        self.prev_dist_to_goal = None
        self.reached_goal = False

        self.global_map = 0.5 * np.ones(map_dim, dtype=np.float32)
        self.log_odds_map = np.zeros(map_dim, dtype=np.float32)
        self.prev_map = 0.5 * np.ones(map_dim, dtype=np.float32)
         
        self.lidar_ranges = np.zeros(LIDAR_ANGLES, dtype=np.float32)
        self.dist_goal = 0
        self.ang_diff = 0
        self.entropy_score=self.smoothness_score=self.occ_confidence=self.free_confidence = 0.0

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            self.seed(seed)

        self._init_episode_vars()

        self.emitter.send("reset".encode())
        for _ in range(5):
            self.robot.step(TIME_STEP)  # Wait for reset

        self._read_human_position()

        obs = self._get_obs(np.zeros(2 * (self.horizon - 1)))
        self.prev_pos = np.array(self.gps.getValues()[:2])
        self.yaw_prev = self._yaw()

        return obs, {}


    def step(self, action_horizon):
        #action_horizon = np.zeros(2*self.horizon)
        self.steps += 1
        actions         = [action_horizon[i:i+2] for i in range(0, len(action_horizon), 2)]
        curr_action     = actions[0]
        self.future_actions = actions[1:]

        # --- Apply current action -----------------------------------
        self._apply_velocity(curr_action[0], curr_action[1])
        for _ in range(5): self.robot.step(TIME_STEP)
        # Compute power = sum(torque × angular velocity)
        power = 0.0
        for i, w in enumerate(self.wheels):
            torque = w.getTorqueFeedback()
            velocity = w.getVelocity()
            wheel_power = torque * velocity
            
            self.torque_logs[i].append(torque)
            self.velocity_logs[i].append(velocity)
            self.power_logs[i].append(wheel_power)
        
            power += wheel_power
        
        # Convert TIME_STEP from ms to seconds
        dt = TIME_STEP / 1000.0
        self.energy_consumed += power * dt
        # --- Distance and Smoothness Tracking ---
        curr_pos = np.array(self.gps.getValues()[:2])
        self.total_distance += np.linalg.norm(curr_pos - self.prev_pos)
        self.prev_pos = curr_pos
        
        yaw_now = self._yaw()
        delta_yaw = abs((yaw_now - self.yaw_prev + np.pi) % (2 * np.pi) - np.pi)  # wrapped difference
        self.total_turn += delta_yaw
        self.yaw_prev = yaw_now

        self._read_human_position()

        # --- Observation -------------------------------------------
        flat_future = np.array([v for pair in self.future_actions for v in pair])
        obs         = self._get_obs(flat_future)

        # --- Future-reward look-ahead ------------------------------
        future_reward = 0.0
        if self.train and self.horizon > 1:
            state = self._save_robot_state()
            future_reward, terminated = self._evaluate_future_actions()
            self._restore_robot_state(state)

        base_reward, terminated = self._compute_reward(curr_action)
        reward    = base_reward + future_reward* lamda_future
        truncated = self.steps >= self.max_steps
        
        self._build_costmap()
        if self.steps % 1 == 0 and self.showMap == True:
            self.show_map()
        return obs, float(reward), bool(terminated), bool(truncated), {}

    def close(self): self._apply_velocity(-1, 0)
    def seed(self, seed=None):
        np.random.seed(seed); random.seed(seed)

    # ------------------------------------------------------------------
    # Observation / reward helpers
    # ------------------------------------------------------------------
    def _yaw(self):
        return math.atan2(*self.compass.getValues()[0:2])
        
    def _get_obs(self, future_actions):
        lidar_ranges = np.clip(self._get_lidar(), 0, MAX_DIST)               
        pos_robot  = np.array(self.gps.getValues()[:2])
        yaw_robot  = self._yaw()

        pos_goal   = np.array(self.human_position)
        vec_goal   = pos_goal - pos_robot
        dist_goal  = np.linalg.norm(vec_goal)
        ang_goal   = math.atan2(vec_goal[1], vec_goal[0])
        ang_diff   = (ang_goal - yaw_robot + math.pi) % (2*math.pi) - math.pi

        self.lidar_ranges = lidar_ranges
        self.dist_goal = dist_goal
        self.ang_diff = ang_diff
                            
        flat_obs = np.concatenate((lidar_ranges / self.max_range,
                                   [dist_goal / 100, ang_diff / math.pi],
                                   future_actions)).astype(np.float32)
                                   
        local_patch = self._get_local_patch()  # shape (CM_SIZE, CM_SIZE
        return {
            "flat": flat_obs,
            "costmap": local_patch
        }
        
    def _compute_reward(self, action):
        lidar, dist_goal, ang_diff = self.lidar_ranges, self.dist_goal, self.ang_diff
        
        progress = (self.prev_dist_to_goal - dist_goal)*lamda_progress if self.prev_dist_to_goal is not None else 0.0
        self.prev_dist_to_goal = dist_goal

        near_obstacle = np.any(lidar[:]< _near_obstacle)
        lidar_penalty = np.sum((self.max_range - lidar)**2) * lamda_lidar
        ang_penalty   = abs(action[1]) * lamda_ang
        head_penalty  = abs(ang_diff) * lamda_heading
        goal_bonus    = lamda_goal if dist_goal < 1.0 else 0.0

        reward = progress + lidar_penalty + ang_penalty + head_penalty + goal_bonus 
        if near_obstacle: reward += lamda_obs

        self.reached_goal = dist_goal < _goal_reached
        terminated = self.reached_goal or near_obstacle

        return reward, terminated

    # ------------------------------------------------------------------
    # Future-reward evaluation
    # ------------------------------------------------------------------
    def _evaluate_future_actions(self):
        total_reward = 0.0
        prev_dist = self.prev_dist_to_goal
        scan0 = self._get_lidar()
        prev_pos = np.array(self.gps.getValues()[:2])
        prev_yaw = self._yaw()
        
        for i, (lin, ang) in enumerate(self.future_actions):
            # Apply action
            self._apply_velocity(lin, ang)
            for _ in range(5): self.robot.step(TIME_STEP)
    
            # Get actual new position and orientation

            new_pos = np.array(self.gps.getValues()[:2])
            new_yaw = self._yaw()
            
            # Predict lidar at new pose
            debug = False
            if i == len(self.future_actions)-1: debug = False
            pred_lidar = self.estimate_lidar(prev_pos, prev_yaw, new_pos, new_yaw, scan0, debug)
            pred_lidar = np.clip(pred_lidar, 0, MAX_DIST)               

            # Goal-related calculations
            vec_goal = np.array(self.human_position) - new_pos
            dist_goal = np.linalg.norm(vec_goal)
            ang_goal = math.atan2(vec_goal[1], vec_goal[0])
            ang_diff = (ang_goal - new_yaw + math.pi) % (2 * math.pi) - math.pi
    
            # Rewards
            progress_r = (prev_dist - dist_goal) * lamda_progress if prev_dist is not None else 0.0
            prev_dist = dist_goal
    
            near_obs = np.any(pred_lidar < 0.5)
            lidar_r = np.sum((self.max_range - pred_lidar)**2) * lamda_lidar 
            ang_r = abs(ang) * lamda_ang
            head_r = abs(ang_diff) * lamda_heading
            goal_r = lamda_goal if dist_goal < _goal_reached else 0.0
            obs_r = lamda_obs if near_obs else 0.0
    
            step_reward = progress_r + lidar_r/ ((i+1)*2) + ang_r + head_r + goal_r + obs_r
            total_reward += step_reward / ((i+1)*2)
    
            if dist_goal < _goal_reached or near_obs:
                break
    
        return total_reward, dist_goal < _goal_reached or near_obs


    # ------------------------------------------------------------------
    # LIDAR Exposure helper
    # ------------------------------------------------------------------
    def estimate_lidar(self, pos0, yaw0, pos1, yaw1, scan0, debug=False):
        ray_angles = np.linspace(-self.fov / 2, self.fov / 2, self.num_rays)
        d_pos = pos1 - pos0
        d_yaw = (yaw1 - yaw0 + np.pi) % (2 * np.pi) - np.pi
    
        est = []
        keep_indices = []
    
        for i, rel_a in enumerate(ray_angles):
            new_rel = rel_a - d_yaw
            if -self.fov / 2 <= new_rel <= self.fov / 2:
                global_a = yaw0 + rel_a
                ray_dir = np.array([np.cos(global_a), np.sin(global_a)])
                along = np.dot(d_pos, ray_dir)
                projected_dist = scan0[i] - along
                est.append(projected_dist)
                keep_indices.append(i)
    
        est = np.array(est, dtype=np.float32)
        if debug:
            plt.clf()
            plt.plot(scan0, label="Original Scan (scan0)",linewidth=3, linestyle="--", color='red')
            plt.plot(est, label="Estimated Scan (est)", color='green', linewidth=1.5)
            plt.title("LiDAR Projection Debug")
            plt.xlabel("Ray Index")
            plt.ylabel("Distance (m)")
            plt.ylim(0, self.max_range)
            plt.xlim(0, 180)

            plt.legend()
            plt.grid(True)
            plt.pause(0.01)
    
        return est
    


    
    # ------------------------------------------------------------------
    # Robot state save / restore (for look-ahead)
    # ------------------------------------------------------------------
    def _save_robot_state(self):
        # Get full 6D velocity
        velocity = self.robot_node.getVelocity()
    
        return {
            # Pose
            'position': list(self.robot_node.getField("translation").getSFVec3f()),
            'rotation': list(self.robot_node.getField("rotation").getSFRotation()),
    
            # Physics velocity
            'linear_velocity': list(velocity[:3]),
            'angular_velocity': list(velocity[3:]),
    
            # Motor targets
            'wheel_velocities': [w.getVelocity() for w in self.wheels],
    
            # Internal state
            'prev_dist_to_goal': self.prev_dist_to_goal,
            'reached_goal': self.reached_goal,
            'prev_pos': self.prev_pos.copy(),
            'yaw_prev': self.yaw_prev,
        }
    
    def _restore_robot_state(self, s):
        # Step 0: Stop everything before resetting
        for i, w in enumerate(self.wheels):
            w.setVelocity(0 if i % 2 == 0 else 0)
        self.robot_node.resetPhysics()

        for _ in range(2): self.robot.step(TIME_STEP)
    
        # Step 2: Restore pose
        self.robot_node.getField("translation").setSFVec3f(s['position'])
        self.robot_node.getField("rotation").setSFRotation(s['rotation'])
    
        for _ in range(2): self.robot.step(TIME_STEP)
    
        # Step 3: Restore 6D velocity (linear + angular)
        self.robot_node.setVelocity(s['linear_velocity'] + s['angular_velocity'])
    
        # Step 4: Restore motor target velocities
        #for i, w in enumerate(self.wheels):
            #w.setVelocity(s['wheel_velocities'][i])
    
        # Step 5: Restore internal environment state
        self.prev_dist_to_goal = s['prev_dist_to_goal']
        self.reached_goal = s['reached_goal']
        self.prev_pos = s['prev_pos']
        self.yaw_prev = s['yaw_prev']
        
        #for _ in range(5):  self.robot.step(TIME_STEP)  # Wait for reset

    # ------------------------------------------------------------------
    # low-level controller
    # ------------------------------------------------------------------
    def _apply_velocity(self, rho, alpha):
        v = ((rho + 1.0) / 2.0) * RHO_MAX
        omega = alpha * ALPHA_MAX
        self.linear_vel_log.append(v)
        self.angular_vel_log.append(omega)
        # Compute wheel speeds (rad/s)
        left_speed = (v - (omega * WHEELBASE / 2)) / WHEEL_RADIUS
        right_speed = (v + (omega * WHEELBASE / 2)) / WHEEL_RADIUS
        # Clip to motor limits
        left_speed = np.clip(left_speed, -MAX_SPEED, MAX_SPEED)
        right_speed = np.clip(right_speed, -MAX_SPEED, MAX_SPEED)
    
        for i, w in enumerate(self.wheels):
            w.setVelocity(left_speed if i % 2 == 0 else right_speed)
 
    # ------------------------------------------------------------------
    # Misc. helpers
    # ------------------------------------------------------------------
    def _get_lidar(self):
        scan = np.array(self.lidar.getRangeImage())
        lidar_ranges = np.interp(np.linspace(0, len(scan) - 1, LIDAR_ANGLES),
                         np.arange(len(scan)), scan)
        return lidar_ranges

    def _read_human_position(self):
        while self.receiver.getQueueLength() > 0:
            msg = self.receiver.getString()
            try:
                self.human_position = [float(x) for x in msg.split(',')[:2]]
            except ValueError:
                self.human_position = [0.0, 0.0]
            self.receiver.nextPacket()
       
    # ――――― cost‑map ―――――――――――――――――――――――――――――――――――――――――――――――――
    def belief_propagation_update(self,log_odds, iterations=2, alpha=0.8, beta=3.0):    
        # Define a 3×3 kernel for neighboring influence
        kernel = np.array([[0.05, 0.1, 0.05],
                           [0.1,  0.4, 0.1],
                           [0.05, 0.1, 0.05]])
    
        for _ in range(iterations):
            smoothed = scipy.ndimage.convolve(log_odds, kernel, mode='constant', cval=0.0)
            log_odds = alpha * log_odds + (1 - alpha) * smoothed
            log_odds = np.clip(log_odds, -beta, beta)
    
        return log_odds


    def _build_costmap(self):
        scan0 = np.array(self.lidar.getRangeImage())
        
        # Add Gaussian noise
        noise_mean = 0.0
        noise_std = 0.1  # or use np.sqrt(variance)
        gaussian_noise = np.random.normal(noise_mean, noise_std, size=scan0.shape)
        scan = scan0 + gaussian_noise
        
        # Ensure scan values remain within valid sensor limits
        scan = np.clip(scan, 0.01, self.max_range+1)

        robot_pos = np.array(self.gps.getValues()[:2])
        yaw = self._yaw()
    
        # Log-odds update parameters
        l_occ = 0.85
        l_free = -0.4
        l_min = -2.0
        l_max = 3.5
    
        for i, dist in enumerate(scan):
            angle_rel = self.fov / 2 - i * self.map_ang_per_ray  # Flip direction
            angle_global = yaw + angle_rel
    
            # Ray casting for free space
            num_steps = max(int(dist / self.map_res), 1)
            for step_frac in np.linspace(0, 1, num_steps, endpoint=False):
                x = robot_pos[0] + step_frac * dist * math.cos(angle_global)
                y = robot_pos[1] + step_frac * dist * math.sin(angle_global)
                mx = int(x / self.map_res)
                my = int(y / self.map_res)
    
                if 0 <= mx < map_dim[1] and 0 <= my < map_dim[0]:
                    self.log_odds_map[my, mx] = np.clip(self.log_odds_map[my, mx] + l_free, l_min, l_max)
    
            # Mark endpoint as occupied if not max range
            if dist < self.max_range - 1e-2:
                x = robot_pos[0] + dist * math.cos(angle_global)
                y = robot_pos[1] + dist * math.sin(angle_global)
                mx = int(x / self.map_res)
                my = int(y / self.map_res)
    
                if 0 <= mx < map_dim[1] and 0 <= my < map_dim[0]:
                    self.log_odds_map[my, mx] = np.clip(self.log_odds_map[my, mx] + l_occ, l_min, l_max)
    
        # Convert log-odds to probabilities in [0,1]
        if self.steps % 10 == 0 and self.BP == True:
            self.log_odds_map = self.belief_propagation_update(self.log_odds_map)
        self.global_map = 1 - 1 / (1 + np.exp(self.log_odds_map))  # sigmoid

              
    def _get_local_patch(self, CM_SIZE=CM_SIZE):
        robot_pos = np.array(self.gps.getValues()[:2])  # (x, y)
        yaw = self._yaw()
        half = CM_SIZE // 2
        patch = np.full((CM_SIZE, CM_SIZE), -1, dtype=np.float32)
    
        for i in range(CM_SIZE):
            for j in range(CM_SIZE):
                # Local grid centered at robot, (0,0) is front-middle
                dx_local = (j - half + 0.5) * self.map_res
                dy_local = (half - i - 0.5) * self.map_res
    
                # Rotate based on robot yaw
                dx_global = dx_local * math.cos(yaw) - dy_local * math.sin(yaw)
                dy_global = dx_local * math.sin(yaw) + dy_local * math.cos(yaw)
    
                x_global = robot_pos[0] + dx_global
                y_global = robot_pos[1] + dy_global
    
                mx = int(x_global / self.map_res)
                my = int(y_global / self.map_res)
    
                if 0 <= mx < self.global_map.shape[1] and 0 <= my < self.global_map.shape[0]:
                    patch[i, j] = self.global_map[my, mx]
    
        patch[np.isnan(patch)] = .5
        patch = (patch - 0.5) * 2  # Normalize to [-1, 1]
        patch = np.clip(patch, -1, 1)
        return patch.astype(np.float32)
       
    def show_map(self):    
        plt.figure("Global Map with Local Patch", figsize=(8, 8))
        plt.clf()
        plt.imshow(self.global_map, cmap='gray', origin='lower', vmin=0.0, vmax=1.0)

        plt.title(f"Global Map (step {self.steps})")
    
        # Get robot pose
        robot_pos = np.array(self.gps.getValues()[:2])
        yaw = self._yaw()
        mx = int(robot_pos[0] / self.map_res)
        my = int(robot_pos[1] / self.map_res)
    
        # Plot robot center
        plt.plot(mx, my, 'ro', markersize=4, label='Robot')
    
        # Draw orientation arrow
        arrow_length = 4
        dx = arrow_length * math.cos(yaw)
        dy = arrow_length * math.sin(yaw)
        plt.arrow(mx, my, dx, dy, head_width=1, head_length=1,
                  fc='red', ec='red', linewidth=1.5, zorder=15)
    
        # Get local patch (already aligned)
        patch = self._get_local_patch()
        half = CM_SIZE // 2
    
        # Overlay the local patch cells with color-coded rectangles (with rotation)
        for dy in range(CM_SIZE):
            for dx in range(CM_SIZE):
                # Convert (dx, dy) to local coordinates centered at (0,0)
                dx_local = (dx - half + 0.5) * self.map_res
                dy_local = (half - dy - 0.5) * self.map_res
    
                # Rotate to global
                dx_global = dx_local * math.cos(yaw) - dy_local * math.sin(yaw)
                dy_global = dx_local * math.sin(yaw) + dy_local * math.cos(yaw)
    
                x_global = robot_pos[0] + dx_global
                y_global = robot_pos[1] + dy_global
    
                gx = int(x_global / self.map_res)
                gy = int(y_global / self.map_res)
    
                if 0 <= gx < self.global_map.shape[1] and 0 <= gy < self.global_map.shape[0]:
                    val = patch[dy, dx]
                    if val > 0.65:
                        color = 'darkgreen'  # likely obstacle
                    elif val < -0.35:
                        color = 'lightgreen'  # likely free
                    elif val < .65 and val > -0.35:
                        color = 'blue'  # uncertain
    
                    plt.gca().add_patch(plt.Rectangle((gx - 0.5, gy - 0.5), 1, 1,
                                                      color=color, alpha=0.6))
    
        plt.axis("equal")
        plt.grid(False)
        plt.pause(0.01)
    
    def plot_wheel_data(self):    
        wheel_names = ['Front Left', 'Front Right', 'Back Left', 'Back Right']
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Torque subplot
        for i in range(4):
            axes[0].plot(self.torque_logs[i], label=wheel_names[i])
        axes[0].set_title("Torque Feedback for All Wheels")
        axes[0].set_ylabel("Torque (Nm)")
        axes[0].legend()
        axes[0].grid(True)
    
        # Velocity subplot
        for i in range(2):
            axes[1].plot(self.velocity_logs[i], label=wheel_names[i])
        axes[1].set_title("Wheel Velocities for All Wheels")
        axes[1].set_ylabel("Velocity (rad/s)")
        axes[1].legend()
        axes[1].grid(True)
    
        # Power subplot
        """for i in range(4):
            axes[2].plot(self.power_logs[i], label=wheel_names[i])
        axes[2].set_title("Wheel Power Consumption for All Wheels")
        axes[2].set_xlabel("Time Step")
        axes[2].set_ylabel("Power (W)")
        axes[2].legend()
        axes[2].grid(True)"""
    
        plt.tight_layout()
        plt.show()
    
    def plot_robot_velocities(self):
    
        plt.figure(figsize=(10, 4))
        plt.plot(self.linear_vel_log, label="Linear Velocity (m/s)")
        plt.plot(self.angular_vel_log, label="Angular Velocity (rad/s)")
        plt.title("Robot Linear and Angular Velocities")
        plt.xlabel("Time Step")
        plt.ylabel("Velocity")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# ======================================================================
# Training / evaluation entry-point
# ======================================================================
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import torch.nn.functional as F

class CostmapEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim=64):
        super().__init__(observation_space, features_dim)

        cm_size = observation_space["costmap"].shape[-1]
        flat_dim = observation_space["flat"].shape[0]

        self.cnn_output_dim = 32

        # Costmap CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * cm_size * cm_size, self.cnn_output_dim),
            nn.ReLU()
        )

        self.attn_dim = flat_dim + self.cnn_output_dim
        self.query = nn.Linear(self.attn_dim, self.attn_dim)
        self.key = nn.Linear(self.attn_dim, self.attn_dim)
        self.value = nn.Linear(self.attn_dim, self.attn_dim)

        # Output MLP
        self.linear = nn.Sequential(
            nn.Linear(self.attn_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        flat = observations["flat"]                              # (B, F)
        costmap = observations["costmap"].unsqueeze(1)           # (B, 1, H, W)

        cost_feat = self.cnn(costmap)                            # (B, C)
        merged = th.cat([flat, cost_feat], dim=1)                # (B, F+C)

        # Gated Attention: Q, K, V from merged
        Q = self.query(merged)                                   # (B, D)
        K = self.key(merged)                                     # (B, D)
        V = self.value(merged)                                   # (B, D)

        attn_scores = th.bmm(Q.unsqueeze(1), K.unsqueeze(2)) / (self.attn_dim ** 0.5)  # (B,1,1)
        attn_weights = th.sigmoid(attn_scores)                   # (B,1,1), gating not softmax
        attn_out = attn_weights.squeeze(-1) * V                  # (B,D), elementwise

        return self.linear(attn_out)                             # (B, features_dim)

# ───────── Training Parameters ─────────────────────────────
import glob
import re
robot_global = Supervisor()

if __name__ == "__main__":
    import torch as th
    import pandas as pd
    from stable_baselines3 import PPO
    from stable_baselines3.ppo import MlpPolicy
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    TIMESTEPS = 400_000
    NUM_EPISODES = 50 
    Train = True
    device = "cpu"
    LOG_DIR = "./logs_ppo"

    results = []

    for HORIZON in range(2, 21, 1):
        print(f"\n🌟 Starting HORIZON = {HORIZON}")
        lamda_future = 1 / HORIZON
        MODEL_PATH = f"ppo_harsh_{HORIZON}.zip"

        # Pass robot_global only once!
        env = Monitor(HarshEnv(robot=robot_global, seed=SEED, train=Train, BP=True,
                               HORIZON=HORIZON))

        policy_kwargs = dict(
            features_extractor_class=CostmapEncoder,
            features_extractor_kwargs=dict(features_dim=64)
        )

        cb = CheckpointCallback(
            save_freq=50_000,
            save_path="./checkpoints",
            name_prefix=f"ppo_harsh_{HORIZON}"
        )

        model = PPO(
            policy=MlpPolicy,
            env=env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=LOG_DIR,
            verbose=0,
            device=device,
            seed=SEED
        )
        
        if Train:
            model.learn(total_timesteps=TIMESTEPS, callback=cb)
            model.save(MODEL_PATH)

        print(f"🎯 Evaluating HORIZON = {HORIZON}")
        best_model_path = None
        best_success = -1
        best_metrics = {}
        
        pattern = f"./checkpoints/ppo_harsh_{HORIZON}_*steps.zip"
        checkpoint_files = sorted(glob.glob(pattern), key=lambda x: int(re.findall(r"(\d+)_steps", x)[0]))
        
        for ckpt_path in checkpoint_files[6:]:
            print(f"🔍 Evaluating checkpoint: {ckpt_path}")
            model = PPO.load(ckpt_path, env=env, device=device)
        
            steps = energy = length = smooth = 0
            success = 1        
            for _ in range(NUM_EPISODES):
                obs, _ = env.reset(seed=SEED)
                done = truncated = False
                while not (done or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, r, done, truncated, _ = env.step(action)
                    #env.env.show_map()

                success += int(env.env.reached_goal)
                steps   += env.env.steps
                length  += env.env.total_distance
                smooth  += env.env.total_turn
                energy  += env.env.energy_consumed
                #env.env.plot_wheel_data()
                #env.env.plot_robot_velocities()
                
                success_rate = success / NUM_EPISODES * 100
            if success_rate > best_success:
                best_success = success_rate
                best_model_path = ckpt_path
                best_metrics = {
                    "Horizon": HORIZON,
                    "Success Rate (%)": round(success_rate, 2),
                    "Avg Length (m)": round(length / success, 2),
                    "Avg Smoothness (rad)": round(smooth / success, 2),
                    "Avg Energy (J)": round(energy / success, 2)
                }
    
        print(f"✅ Best Model for HORIZON={HORIZON}: {best_model_path}")
        print(f"📈 Metrics: {best_metrics}")
        results.append(best_metrics)
        df = pd.DataFrame(results)
        df.to_excel("results.xlsx", index=False)
