

import os, sys
import time
import random
import queue
from collections import deque
import numpy as np 
import math
from scipy.special import logsumexp
from enum import Enum 
import gym
import carla
from dm_control.utils import rewards

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CUR_DIR)
from render import CarlaRender
from constants import *
from benchmark import DrivingTask


class CarlaEnv(gym.Env):
    """
        - State: 
        - Action: [steer, throttle, brake] 3-dim float 
        - Reward:  
        - Info:
    """
    def __init__(self, is_render=False, random_seed=2022, task_name="straight", perturb_spec={}, port=2000, domain_random=False): 

        super(CarlaEnv, self).__init__()

        # 1. connect carla server
        self.client = carla.Client("localhost", port)
        self.client.set_timeout(20.0)
        self.vehicle, self.world, self.town = None, None, "town_name"
        self.is_render = is_render

        self.pygame_render = CarlaRender() if is_render else None

        # 2. define running task
        self.task = DrivingTask(task_name, domain_random)

        # 3. perturb on parameters
        self.perturb_spec = perturb_spec

        # 4. gym style for RL
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32) 
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32) # steer, throttle, brake

        self.steps = 0
        self.time = 0
        self.metadata = {}


    def _setup_world(self, start_id=None):

        # ========= Select a set of parameters ===========
        # randomly sample a world parameters
        param_dict = self.task.sample_train_setting()
        # modify it if needed
        for k, v in self.perturb_spec.items():
            if k in param_dict:
                param_dict[k] = v
        print("- World Params:", param_dict)

        # ========= Start set up parameters ===========
        # town spawing point
        town, pose = param_dict.get("towns_poses", ("Town01", [36, 40]))
        if town != self.town: # only reload world when town is changed # only reload world when town is changed # only reload world when town is changed # only reload world when town is changed
            self.town = town
            # M: try connection 10 times for more robustness
            connections = 0
            while connections < 10:
                try:
                    self.world = self.client.load_world(town)
                    break
                except:
                    connections += 1
                    print(f"[Warning] Try to connect for {connections} times")
            self.world.apply_settings(carla.WorldSettings(
                no_rendering_mode=not self.is_render, # False for debug
                synchronous_mode=True,
                fixed_delta_seconds=DT_))
            self.map = self.world.get_map()
            self.blueprint_library = self.world.get_blueprint_library()

        # # road friction
        # road_friction = param_dict.get("road_friction", 3.5)
        # friction_bp = self.world.get_blueprint_library().find('static.trigger.friction')
        # extent = carla.Location(1000000.0, 1000000.0, 1000000.0)
        # friction_bp.set_attribute('friction', str(road_friction))
        # friction_bp.set_attribute('extent_x', str(extent.x))
        # friction_bp.set_attribute('extent_y', str(extent.y))
        # friction_bp.set_attribute('extent_z', str(extent.z))
        # transform = carla.Transform()
        # transform.location = carla.Location(-10000.0, -10000.0, 0.0)
        # self.world.spawn_actor(friction_bp, transform)

        # starting point
        self.spawn_points = self.world.get_map().get_spawn_points()

        # car type
        car_type  = param_dict.get("car_type", "model3")
        self.car_model = self.blueprint_library.filter(car_type)[0]

        spawn_flag = True 
        while spawn_flag:
            try:
                if start_id is not None:
                    self.spawn_point = self.spawn_points[start_id%len(self.spawn_points)]
                else:
                    # self.spawn_point = random.choice(self.spawn_points)
                    index = np.random.randint(len(self.spawn_points))
                    self.spawn_point = self.spawn_points[index]

                self.vehicle = self.world.spawn_actor(self.car_model, self.spawn_point)
                spawn_flag = False 
            except:
                continue

        physics_control = self.vehicle.get_physics_control()
        
        # car_type will override other perturbations
        if "car_type" not in self.perturb_spec:
            wheels = physics_control.wheels 
            # tire_friction 
            tire_friction = param_dict.get("tire_friction", 3.5)
            wheels[0].tire_friction = tire_friction
            wheels[1].tire_friction = tire_friction
            wheels[2].tire_friction = tire_friction
            wheels[3].tire_friction = tire_friction
            # damping rate
            damping_rate = param_dict.get("damping_rate", 0.25)
            wheels[0].damping_rate = damping_rate
            wheels[1].damping_rate = damping_rate
            wheels[2].damping_rate = damping_rate
            wheels[3].damping_rate = damping_rate
            # radius 
            radius = param_dict.get("radius ", 37)
            wheels[0].radius = radius
            wheels[1].radius = radius
            wheels[2].radius = radius
            wheels[3].radius = radius
            physics_control.wheels = wheels

            # car_mass
            car_mass = param_dict.get("car_mass", 1845.0)
            physics_control.mass = car_mass
            # drag coefficient
            drag_coefficient = param_dict.get("drag_coefficient", 0.15)
            physics_control.drag_coefficient = drag_coefficient
            # moi
            moi = param_dict.get("moi", 1.0)
            physics_control.moi = moi 
            # final ratio 
            final_ratio = param_dict.get("final_ratio", 4.0)
            physics_control.final_ratio = final_ratio

            self.vehicle.apply_physics_control(physics_control)


        # render the car
        if self.pygame_render:
            # _add_accessories
            # - colsensor
            transform = carla.Transform(carla.Location(x=2.5, z=0.7)) # transform for collision sensor attachment
            colsensor = self.blueprint_library.find("sensor.other.collision")
            self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
            self.colsensor.listen(lambda event: self.collision_data(event)) # what's the mechanism?
            # - camera
            blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
            blueprint.set_attribute('image_size_x', '{}'.format(RES_X))
            blueprint.set_attribute('image_size_y', '{}'.format(RES_Y))
            # blueprint.set_attribute('fov', '110')
            self.camera = self.world.spawn_actor(blueprint, carla.Transform(carla.Location(x=-11.0, z=5.6), carla.Rotation(pitch=-15)), attach_to=self.vehicle) # -5.5, 2.8, -15
            self.camera.image_size_x = RES_X
            self.camera.image_size_y = RES_Y
            self.image_queue = queue.Queue()
            self.camera.listen(self.image_queue.put)


    def reset(self, start_id=None): 

        print("\nResetting...")
        print(f"- Last Episode Steps: {self.steps}")
        try:
            self.close()
        except:
            print("Closed Error")

        self.time = 0
        self.steps = 0
        self.collision_hist = []

        # 1. reset env parameters
        t0 = time.time()
        self._setup_world(start_id)
        t1 = time.time()

        # 2. car warmup: speedup for several seconds
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.5))
        START_TIME = 3.0
        # print(f"Initialzing the car for {START_TIME} s...")
        for i in range(int( START_TIME / DT_)):
            self.world.tick()
            if self.pygame_render:  
                self.image_queue.get()
        t2 = time.time()
        print(f"- Carla Env Reset: setup_world {t1-t0:.2f} s | warmup_car {t2-t1:.2f} s")

        # 3. udpate waypoints
        self.start_point = self.vehicle.get_location()
        self.waypoint_buffer = deque(maxlen=WAYPOINT_BUFFER_LEN)
        self.update_waypoint_buffer(given_loc=self.start_point)
        self.end_point = self.waypoint_buffer[-1].transform.location
        print(f"- Start Point: {self.start_point} | End Point: {self.end_point}")
        self.last_goal_error = self.end_point.distance(self.start_point)
        return self.get_state(), {"waypoints": self.get_waypoint(), "car_loc": self.start_point}


    def collision_data(self, event):
        self.collision_hist.append(event)


    def update_waypoint_buffer(self, given_loc=None):
        if given_loc is not None:
            car_loc = given_loc
        else:
            car_loc = self.vehicle.get_location()

        if (len(self.waypoint_buffer) == 0):
            self.waypoint_buffer.append(self.map.get_waypoint(car_loc))

        # select closed waypoint to the car
        self.min_distance = np.inf
        min_distance_index = 0
        for i in range(len(self.waypoint_buffer)):
            curr_distance = self.waypoint_buffer[i].transform.location.distance(car_loc)
            if curr_distance < self.min_distance:
                self.min_distance = curr_distance
                min_distance_index = i

        # TODO: how waypoints generated, supplement waypoints to BUFFER_LEN
        # num_waypoints_to_be_added = max(0, min_distance_index - WAYPOINT_BUFFER_MID_INDEX)
        # num_waypoints_to_be_added = max(num_waypoints_to_be_added, WAYPOINT_BUFFER_LEN - len(self.waypoint_buffer))
        num_waypoints_to_be_added = max(0, min_distance_index + WAYPOINT_BUFFER_LEN - len(self.waypoint_buffer))

        for _ in range(num_waypoints_to_be_added):
            frontier = self.waypoint_buffer[-1]
            next_waypoints = list(frontier.next(WAYPOINT_INTERVAL))
            if len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                road_options_list = _retrieve_options(
                    next_waypoints, frontier)
                # TODO: randomly select route now (straight, turn left, turn right)
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(
                    road_option)]
            self.waypoint_buffer.append(next_waypoint)

        # M: should also update min_distance_index !
        # min_distance_index = 0
        # self.min_distance_index = WAYPOINT_BUFFER_MID_INDEX if min_distance_index > WAYPOINT_BUFFER_MID_INDEX else min_distance_index
    

    def calc_reward(self, ):
        """
            ! Always call this after update_waypoint_buffer
        """
        # route error
        car_loc = self.vehicle.get_location()
        self.route_error = self.waypoint_buffer[0].transform.location.distance(car_loc)
        # goal error
        self.goal_error = self.end_point.distance(car_loc)
        goal_progress = self.last_goal_error - self.goal_error # > 0
        self.last_goal_error = self.goal_error
        # vel error
        self.velocity = self.vehicle.get_velocity()
        vx = self.velocity.x
        vy = self.velocity.y
        self.vel_error =  np.sqrt((TARGET_SPEED - np.sqrt(vx**2+vy**2) ) ** 2)

        # REWARD TYPE
        # 1. route_error + goal_error (unnormalized)
        reward = - self.route_error + goal_progress - 1.0
        if len(self.collision_hist) != 0:
            reward -= 100

        # 2. route_error + goal_error (normalized)
        reward = np.exp(-self.route_error) * np.exp(-self.goal_error/ max(101 - self.steps, 1))
        if len(self.collision_hist) != 0:
            reward = 0

        # 3. reward: route_error +  speed_error 
        # control_cost = np.linalg.norm(self.action, ord=2)
        route_error_bound = 1.0 # [m]
        route_reward = rewards.tolerance(
                               -self.route_error,
                               bounds=(-route_error_bound, 0),
                               margin=5 * route_error_bound,
                               value_at_margin=0,
                               sigmoid='linear')
        vel_error_bound = 1.0 # [m/s]
        vel_reward = rewards.tolerance(
                               -self.vel_error,
                               bounds=(-vel_error_bound, 0),
                               margin=5 * vel_error_bound,
                               value_at_margin=0,
                               sigmoid='linear')
        reward = route_reward * vel_reward
        if len(self.collision_hist) != 0:
            reward = 0
        return reward

    def calc_done(self, ):
        """
            Always call this after calc_reward
        """
        done = False
        if len(self.collision_hist) != 0:
            done = True
        if self.steps == MAX_STEPS: done = True
        # if self.goal_error < DISTANCE_TO_SUCCESS:
        #     done = True
        return done

    def get_state(self,):
        # collect information
        self.location = self.vehicle.get_location()
        self.location_ = np.array([self.location.x, self.location.y, self.location.z])

        self.transform = self.vehicle.get_transform()
        # self.yaw = np.array(self.transform.rotation.yaw) # float, only yaw: only along z axis # check https://d26ilriwvtzlb.cloudfront.net/8/83/BRMC_9.jpg 
        phi = self.transform.rotation.yaw*np.pi/180 # phi is yaw

        self.velocity = self.vehicle.get_velocity()
        vx = self.velocity.x
        vy = self.velocity.y

        beta_candidate = np.arctan2(vy, vx) - phi + np.pi*np.array([-2,-1,0,1,2])
        local_diff = np.abs(beta_candidate - 0)
        min_index = np.argmin(local_diff)
        beta = beta_candidate[min_index]

        # state = [self.velocity.x, self.velocity.y, self.yaw, self.angular_velocity.z]
        state = [
                    self.location.x, # x
                    self.location.y, # y
                    np.sqrt(vx**2 + vy**2), # v
                    phi, # phi
                    beta, # beta
                ]
        return np.array(state)

    def get_waypoint(self,):
        """
            Always call this after update_waypoint_buffer
        """
        waypoints = []
        for i in range(FUTURE_WAYPOINTS_AS_STATE):
            waypoint_location = self.waypoint_buffer[i].transform.location
            waypoints.append([waypoint_location.x, waypoint_location.y])
        return np.array(waypoints)

    def step(self, action, action_info={}): # 0:steer; 1:throttle; 2:brake; np array shape = (3,); range:[-1,1]
        assert len(action) == 3

        steer_, throttle_, brake_ = action
        throttle_ = (throttle_ + 1 ) * 0.5 # [0,1]
        brake_ = (brake_+ 1 ) * 0.5 # [0,1] 
        assert steer_ >= -1 and steer_ <= 1 and throttle_ <= 1 and throttle_ >= 0 and  brake_ <= 1 and brake_ >= 0

        self.action = np.array([steer_, throttle_, brake_])

        # move a controller step (contain N_DT DTs)
        self.vehicle.apply_control(carla.VehicleControl(throttle=float(throttle_), steer=float(steer_), brake=float(brake_)))

        self.steps += 1

        for i in range(N_DT):
            self.world.tick() 
            self.time += DT_

            if self.pygame_render:  
                draw_info = {
                        "image": self.image_queue.get(),
                        "velocity": self.vehicle.get_velocity(),
                        "action": action,
                        "world": self.world,
                        "x_trj": action_info.get("x_trj", []),
                        "location": self.location_,
                        "waypoints": self.waypoint_buffer,
                        "min_distance_index": 0,
                        }
                self.pygame_render.tick(draw_info, i) 

        # next state
        car_loc = self.vehicle.get_location()
        next_state = self.get_state()

        # update waypoints to recalculate nearest point
        self.update_waypoint_buffer()
        waypoints = self.get_waypoint()

        # reward, done
        reward = self.calc_reward() 
        done = self.calc_done()

        success = 1.0 if self.goal_error < DISTANCE_TO_SUCCESS else 0.0

        return next_state, reward, done, {"waypoints": waypoints, "route_error": self.route_error, "goal_error": self.goal_error, "success": success, "control": np.array([steer_, throttle_, brake_]), "car_loc": car_loc, "vel_error": self.vel_error}


    def render(self, mode="human"):
        if self.pygame_render:
            return self.pygame_render.get_images()
        else:
            return []

    def close(self):
        if hasattr(self, 'camera') and self.camera:
            self.camera.destroy()
        if hasattr(self, 'colsensor') and self.colsensor:
            self.colsensor.destroy()
        if hasattr(self, 'vehicle') and self.vehicle:
            self.vehicle.destroy()
        return

class RoadOption(Enum): # for waypoint setting
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6

# two functions for waypoint selection
def _retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options

def _compute_connection(current_waypoint, next_waypoint, threshold=35):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < threshold or diff_angle > (180 - threshold):
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT

def sldist(c1, c2):
    return math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)


# ==============================================================================
# -- Testing ---------------------------------------------------------
# ==============================================================================

if __name__ == "__main__":

    env = CarlaEnv(is_render=False, random_seed=2022, task_name="turn")

    for ep in range(10):
        print(f"=== Episode: {ep} ===")
        state, info = env.reset()
        times = []
        for i in range(100):
           # action = f(current_State)
           action = np.random.rand(3)
           action[0] = (action[0]-0.5)*2
           action[1] = 1
           action[2] = 0
           t = time.time()
           next_state, reward, done, info = env.step(action)
           # times.append(time.time()-t)
           times.append(info["time"])

           # print(f"=== Step {i} ===")
           # print(f"state: {state} action: {action} reward: {reward} next_state: {next_state}") 
           # images = env.render()
           # print(len(images))

           state = next_state

        print(f"AVG_TIME/STEP: {np.mean(times)} FPS: {1.0/np.mean(times)} MAX_TIME: {np.max(times)}")

    env.close()



