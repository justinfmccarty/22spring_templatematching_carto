import numpy as np
import gym
from gym import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common import env_checker
import seaborn as sns
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
from shapely.geometry import Polygon, LineString, Point
import random
import time
from rasterio.features import rasterize
from IPython import display
import geopandas as gpd
import pandas as pd
from skimage.feature import match_template
import utils


class TemplateMatchingEnv(gym.Env):
    def __init__(self, image_path, temp_loc, obs_shape_mult=(3,3),compare_method='SSD',sim_threshold=1000,print_st=False):
        super(TemplateMatchingEnv, self).__init__()

        self.print_st = print_st
        self.compare_method = compare_method
        self.sim_threshold = sim_threshold
        
        # load image
        self.image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        # self.image = self.image_color # (self.image_color/255).astype(float).copy()
        self.image_array = self.image.copy()
        self.image_shape = self.image_array.shape
        self.I_HEIGHT, self.I_WIDTH, self.I_DEPTH = self.image_shape

        # load template 
        # self.template = cv2.cvtColor(cv2.imread(template_path), cv2.COLOR_BGR2RGB)
        self.template = self.image_array[temp_loc[0]:temp_loc[0]+temp_loc[2],temp_loc[1]:temp_loc[1]+temp_loc[3]]
        self.template_array = self.template.copy()
        # self.template_array = np.ones(self.template_array.shape)
        self.T_HEIGHT, self.T_WIDTH, self.T_DEPTH = self.template_array.shape
        
        self.obs_shape_mult_height = obs_shape_mult[0]
        self.obs_shape_mult_width = obs_shape_mult[1]
        self.obs_space_height = self.T_HEIGHT * self.obs_shape_mult_height
        self.obs_space_width = self.T_WIDTH * self.obs_shape_mult_width
        
        # Define a 2-D observation space
        self.observation_shape = (self.obs_space_height,
                                  self.obs_space_width,
                                  3)
        self.observation_space = spaces.Box(low = 0,#np.zeros(self.observation_shape), 
                                            high = 255,#np.ones(self.observation_shape),
                                            shape = self.observation_shape,
                                            dtype = np.uint8)
        
        # Action to choose the next agent location 
        # 0 -> 8 corresponds to the list of neighbor frames 
        self.action_space = spaces.Discrete(8)
        
        # # Create a canvas to render the environment images upon 
        # self.canvas = np.ones(self.observation_shape) * 1
        
        # Define elements present inside the environment
        self.elements = []
        
        # Maximum moves the frame can make before resetting the episode
        self.max_moves = 8
        
        # store window origin, dependent on observation multipliers and tempalte size
        self.window_y =  int(np.floor(self.obs_shape_mult_height/2)) * self.T_HEIGHT
        self.window_x = int(np.floor(self.obs_shape_mult_width/2)) * self.T_WIDTH
        
    def reset(self):
        if self.print_st==True:
            print("Setting new episode.")
        self.canvas = np.ones(self.observation_shape)
        self.actions = np.arange(0,self.action_space.n)
        
        # reset the image
        self.image_array = self.image.copy()
        self.I_HEIGHT, self.I_WIDTH, self.I_DEPTH = self.image_shape
        
        # set action and movement tracking variables back to nulls
        self.previous_action = None
        self.moves_taken = 0
        self.moves_left = self.max_moves
        self.log_points = []
        self.log_moves = []
        self.log_res = []
        self.log_closed = False
        self.log_actions = np.arange(0,self.action_space.n)
        self.ep_return  = 0
        
        # intialise the frame
        # set some buffers to make sure the frame doesn't eventually extend past image
        self.height_buffer = self.T_HEIGHT*(self.obs_shape_mult_height+1)
        self.width_buffer = self.T_WIDTH*(self.obs_shape_mult_width+1)
        self.y_max = self.I_HEIGHT-(self.height_buffer)
        self.x_max = self.I_WIDTH-(self.width_buffer)
        
        # create y and x lists for possible point locations of points (this is a grid)
        # self.lin_y = np.linspace(0,self.y_max,int(np.floor(
        #     self.y_max/self.T_HEIGHT)),dtype="int")
        self.lin_y = utils.calc_increment_space(self.I_HEIGHT,self.T_HEIGHT)
        self.lin_x = utils.calc_increment_space(self.I_WIDTH,self.T_WIDTH)
        # self.lin_x = np.linspace(0,self.x_max,int(np.floor(
        #     self.x_max/self.T_WIDTH)),dtype="int")
        
        # create an origin for the frame for the episode
        self.random_y_origin = random.choice(self.lin_y[self.lin_y<self.y_max]) # random.randint(0, self.y_max)
        self.random_x_origin = random.choice(self.lin_x[self.lin_x<self.x_max]) # random.randint(0, self.x_max)
        self.episode_origin = (self.random_y_origin,self.random_x_origin)
        self.previous_point = self.episode_origin
        
        # append origin point to the points log
        self.log_points.append(self.episode_origin)
        self.log_moves.append("Origin")
        
        # subset the image array to create the observation frame
        self.frame_array = utils.crop_image(self.image_array,
                                            self.random_y_origin,
                                            self.random_x_origin,
                                            self.obs_space_height,
                                            self.obs_space_width)

        # Intialise the elements (for now this only includes the array)
        self.elements = [self.frame_array]#,self.window]
        
        # Draw elements on the canvas
        self.draw_elements_on_canvas()
        
        # return the observation
        return self.canvas 
        
    def draw_elements_on_canvas(self):
        self.canvas = self.frame_array
    
    def render(self, mode = "human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            plt.imshow(self.canvas)
        
        elif mode == "rgb_array":
            return self.canvas
        
    def close(self):
        cv2.destroyAllWindows()
    
    
    def gen_possible_actions(self):
        # log the previous point
        if self.moves_taken < 1:
            self.previous_point = self.episode_origin
        else:
            self.previous_point = self.updated_point
        
        # get possible actions and point
        self.possible_actions = self.actions
        self.possible_points = self.generate_possible_pts()
                
        # self.termination_points = calc_termination_point(self.previous_point,
        #                                                  (env.obs_shape_mult_height,env.obs_shape_mult_width),
        #                                                  (env.T_HEIGHT,env.T_WIDTH))
        
        remove_actions = []
        # evaluate possible points 
        for n,pt in enumerate(self.possible_points):
            t_pt = utils.calc_termination_point(pt,
                                          (self.obs_shape_mult_height,self.obs_shape_mult_width),
                                          (self.T_HEIGHT,self.T_WIDTH))
            if pt==self.episode_origin:
                if self.moves_taken < 2:
                    if self.print_st==True:
                        print('Early close attempt possible. Remove.')
                    remove_actions.append(n)
                else:
                    pass
            elif pt in self.log_points:
                if self.print_st==True:
                    print('Point already used')
                remove_actions.append(n)
            elif pt[0] < 0 or pt[0]  > self.y_max:
                if self.print_st==True:
                    print('Point is not on image')
                remove_actions.append(n)
            elif pt[1] < 0 or pt[1] > self.x_max:
                if self.print_st==True:
                    print('Point is not on image')
                remove_actions.append(n)
            elif pt[0] < 0+(2*self.T_HEIGHT) or pt[0]  > self.y_max-(2*self.T_HEIGHT):
                if self.print_st==True:
                    print("Point is within buffer edge of border")
                remove_actions.append(n)
            elif pt[1] < 0+(2*self.T_WIDTH) or pt[1]  > self.x_max-(2*self.T_WIDTH):
                if self.print_st==True:
                    print("Point is within buffer edge of border")
                remove_actions.append(n)
            elif t_pt[0] < 0 or t_pt[0]  > self.y_max:
                if self.print_st==True:
                    print("Termination point is outside of the border")
                remove_actions.append(n)
            elif t_pt[1] < 0 or t_pt[1]  > self.x_max:
                if self.print_st==True:
                    print("Termination point is outside of the border")
                remove_actions.append(n)
            else:
                if self.print_st==True:
                    print("Point never used")
                else:
                    pass
        
        self.possible_actions = np.delete(self.possible_actions,remove_actions)
        
        
    
    # def choose_action(self):
    #     self.gen_possible_actions()
    #     if len(self.possible_actions)>0:
    #         return random.choice(self.possible_actions)
    #     else:
    #         if self.print_st==True:
    #             print('No actions possible, next epsiode')
    #         return -1
            
    def step(self,action):
        info = {}
        # make sure episode is continuing
        done = False
        self.action = action
        self.gen_possible_actions()
        if self.action not in self.possible_actions:
            reward = self.ep_return
            if self.print_st==True:
                print("Action Not Possible")
            done = True
        else:
            # apply the action to the frame
            if self.print_st==True:
                print(f"Action is {self.action}, {utils.get_action_meanings()[self.action]}")
            if self.action == 0: # south
                self.movement = (self.T_HEIGHT,0)
            elif self.action == 1: # southwest
                self.movement = (self.T_HEIGHT,self.T_WIDTH*-1)
            elif self.action == 2: # west
                self.movement = (0,self.T_WIDTH*-1)
            elif self.action == 3: # northwest
                self.movement = (self.T_HEIGHT*-1,self.T_WIDTH*-1)
            elif self.action == 4: # north
                self.movement = (self.T_HEIGHT*-1,0)
            elif self.action == 5: # northeast
                self.movement = (self.T_HEIGHT*-1,self.T_WIDTH)
            elif self.action == 6: # east
                self.movement = (0,self.T_WIDTH)
            elif self.action == 7: # southeast
                self.movement = (self.T_HEIGHT,self.T_WIDTH)
            elif self.action == -1: # no moves
                done = True
                reward = self.ep_return
                return self.canvas, reward, done, info 
                
            
            # update canvas and logs
            # store some data
            self.previous_action = self.action
            self.updated_point = utils.tuple_addition(self.previous_point,self.movement)
            self.log_points.append(self.updated_point)
            self.log_moves.append(self.action)#utils.get_action_meanings()[self.action])
            # move the frame by the action movements (this is really a recropping)
            # if self.moves_taken==0:
            #     pass
            # else:
            self.frame_array = utils.crop_image(self.image_array,
                                        self.updated_point[0],#-self.window_y,#
                                        self.updated_point[1],#-self.window_x,#
                                        #   self.random_y_origin + self.movement[0],#self.frame_y
                                        #   self.random_x_origin + self.movement[1],#self.frame_x
                                        self.obs_space_height,
                                        self.obs_space_width)
                
            # Decrease the moves remaining 
            self.moves_left -= 1 
            self.moves_taken += 1 
            if self.print_st==True:
                print(f"Moves remaining, {self.moves_left}")
            
            # test the window and template
            if self.test_match() is True:
                if self.print_st==True:
                    print("Match")
                # Reward for a match.
                reward_a = 1
                # Reward for contiguous matches.
                reward_b = 1 * self.moves_taken
                reward = reward_a+reward_b
                
                if self.log_points[0] == self.log_points[-1]:
                    if self.print_st==True:
                        print('Episode origin point visited. Loop created. Next Episode')
                    self.log_closed = True
                    reward += self.moves_taken*2
                    
                    # Increment the episodic return
                    self.ep_return += reward

                    # Draw elements on the canvas
                    self.draw_elements_on_canvas()
                    self.moves_left = 0
                    done = True
                else:
                    # Reset the moves allowed
                    self.moves_left = self.max_moves
            else:
                if self.print_st==True:
                    print('No match, next epsiode')
                reward = 0
                done = True
            
            # Increment the episodic return
            self.ep_return += reward

            # Draw elements on the canvas
            self.draw_elements_on_canvas()

            # If out of moves, end the episode.
            if self.moves_left == 0:
                done = True
            
        return self.canvas, reward, done, info
    
    def generate_possible_pts(self):
        all_moves = [(self.T_HEIGHT,0),
                        (self.T_HEIGHT,self.T_WIDTH*-1),
                        (0,self.T_WIDTH*-1),
                        (self.T_HEIGHT*-1,self.T_WIDTH*-1),
                        (self.T_HEIGHT*-1,0),
                        (self.T_HEIGHT*-1,self.T_WIDTH),
                        (0,self.T_WIDTH),
                        (self.T_HEIGHT,self.T_WIDTH)]
        return [utils.tuple_addition(self.previous_point,pt) for pt in all_moves]
    
        
        
    def test_match(self):
        self.window_array = utils.crop_image(self.frame_array,
                                        self.window_y,
                                        self.window_x,
                                        self.T_HEIGHT,
                                        self.T_WIDTH)
        self.window_grey = utils.remove_white_hsv_meth(self.window_array).copy()
        self.template_grey = utils.remove_white_hsv_meth(self.template_array).copy()
        if self.compare_method == 'SSD':            
            res_list = []
            for i in [0,1,2]:
                res_list.append(utils.ssd(self.window_grey[:,:,i],
                                    self.template_grey[:,:,i]))
            res_mean = sum(res_list) / len(res_list)
            # result_.append(ssd_) # log results
            if self.print_st==True:
                print(f"{self.compare_method} result is {res_mean}")
            self.log_res.append(res_mean)
            if res_mean<self.sim_threshold:
                return True
            else:
                return False
            
        elif self.compare_method == 'NCC':
            res_list = []
            for i in [0,1,2]: 
                res_list.append(utils.Normalized_cross_correlation(self.window_grey[:,:,i],
                                                             self.template_grey[:,:,i]))
            res_mean = sum(res_list) / len(res_list)
            # result_.append(ssd_) # log results
            if self.print_st==True:
                print(f"{self.compare_method} result is {res_mean}")
            self.log_res.append(res_mean)
            if res_mean>0.8:
                return True
            else:
                return False