
import os, sys
import numpy as np
import math
import itertools

# set up pygame
import carla
import pygame
os.environ['SDL_VIDEODRIVER'] = 'dummy'
pygame.init()

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CUR_DIR)
from constants import *

class CarlaRender(object):
    def __init__(self):
        pygame.display.set_mode((1,1))
        self.display = pygame.Surface((RES_X, RES_Y), pygame.SRCALPHA, 32)
        self.font = get_font()
        self.clock = pygame.time.Clock()
        self.render_images = []

    def tick(self, draw_info, dt_cnt):

        self.clock.tick()
        if dt_cnt == 0:
            self.render_images = []
            draw_planned_trj(draw_info["world"], np.array(draw_info["x_trj"]), draw_info["location"][2], color=(0, 223, 222))
            past_wp = list(itertools.islice(draw_info["waypoints"], 0, draw_info["min_distance_index"]))
            future_wp = list(itertools.islice(draw_info["waypoints"], draw_info["min_distance_index"]+1, len(draw_info["waypoints"])))
            draw_waypoints(draw_info["world"], future_wp, z=0.5, color=(255,0,0))
            draw_waypoints(draw_info["world"], past_wp, z=0.5, color=(0,255,0))
            draw_waypoints(draw_info["world"], [draw_info["waypoints"][draw_info["min_distance_index"]]], z=0.5, color=(0,0,255))

        # 1. draw image 
        draw_image(self.display, draw_info["image"])
        vel = draw_info["velocity"]
        self.display.blit(
                self.font.render('Velocity = {0:.2f} m/s'.format(math.sqrt(vel.x**2 + vel.y**2)), True, (255, 255, 255)),
                (8, 10))

        steer_, throttle_, brake_ = draw_info["action"]
        v_offset = 25
        bar_h_offset = 75
        bar_width = 100
        for key, value in {"steering":steer_, "throttle":throttle_, "brake":brake_}.items():
            rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
            pygame.draw.rect(self.display, (255, 255, 255), rect_border, 1)
            if key == "steering":
                rect = pygame.Rect((bar_h_offset + (1+value) * (bar_width)/2, v_offset + 8), (6, 6))
            else:
                rect = pygame.Rect((bar_h_offset + value * (bar_width), v_offset + 8), (6, 6))
            pygame.draw.rect(self.display, (255, 255, 255), rect)
            self.display.blit(
                self.font.render(key, True, (255, 255, 255)), (8, v_offset+3))
            v_offset += 18

        pygame.display.flip()

        # save image
        # - pygame.error: pygame_Blit: Surfaces must not be locked during blit
        # image = pygame.surfarray.pixels2d(self.display)
        string_image = pygame.image.tostring(self.display, 'RGB')
        temp_surf = pygame.image.fromstring(string_image,(RES_X, RES_Y), 'RGB')
        image = pygame.surfarray.array3d(temp_surf)
        self.render_images.append(np.transpose(image, (1,0,2)))


    def get_images(self):
        return self.render_images

def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def draw_waypoints(world, waypoints, z=0.5, color=(255,0,0)): # from carla/agents/tools/misc.py
    """
    Draw a list of waypoints at a certain height given in z.

    :param world: carla.world object
    :param waypoints: list or iterable container with the waypoints to draw
    :param z: height in meters
    :return:
    """
    color = carla.Color(r=color[0],g=color[1],b=color[2],a=255)
    for w in waypoints:
        t = w.transform
        begin = t.location + carla.Location(z)
        # angle = math.radians(t.rotation.yaw)
        # end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_point(begin, size=0.05, color=color, life_time=0.1)

def draw_planned_trj(world, x_trj, car_z, color=(255,0,0)):
    color = carla.Color(r=color[0],g=color[1],b=color[2],a=255)
    thickness = 0.05
    length = x_trj.shape[0]
    if length == 0: return
    xx = x_trj[:,0]
    yy = x_trj[:,1]
    for i in range(1, length):
        begin = carla.Location(float(xx[i-1]), float(yy[i-1]), float(car_z+1))
        end = carla.Location(float(xx[i]), float(yy[i]), float(car_z+1))
        world.debug.draw_line(begin=begin, end=end, thickness=thickness, color=color, life_time=DT_*N_DT)
    return

def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)

