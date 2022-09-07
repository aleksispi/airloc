#!/bin/env python3


from sys import exit
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
from skimage.transform import resize
from PIL import Image
import os
from matplotlib import patches
from matplotlib.colors import from_levels_and_colors
from matplotlib.widgets import TextBox

parser = argparse.ArgumentParser()

# Change to default None later
group = parser.add_mutually_exclusive_group()

group.add_argument("--image" , type = str , help = "Path to a image that the game should use", default = None)
group.add_argument("--image-dir", type = str, help = "Path to a folder containing the images that the game should use", default = None)

parser.add_argument("--seed" , type = int , default = 0 , help = "Seet the seed for all random number generators.")
parser.add_argument("--randomize-locs" , type = bool, default = False)
parser.add_argument("--image-size" , type = int, default = (4*48*1.1+48), help = "Size of the image used as playing field")
parser.add_argument("--debug" , type = bool, default = False)
parser.add_argument("--image-scale" , type = int , help = "Resize images", default = 2)
parser.add_argument("--time-limit" , type = int , help = "Set time limit per round", default = 60)
parser.add_argument("--real" , type = bool, default = False, help = "Wether to run the game in real mode with validation images.")
parser.add_argument("--trace", type = str, choices = ["none", "rgb", "regular"] ,default = "rgb", help = "Fill in trace in the position map of where you have been.")
parser.add_argument("--number-steps" , type = int , default = 10, help = "Select the number of steps allowed.")

args = parser.parse_args()

matplotlib.use('TkAgg')

# Set seeds 
np.random.seed(args.seed)
random.seed(args.seed)

def sample_grid_locs( n_games , im_size , step_size, patch_size):
    """
        Utils function for sampling random start and goal grid locations.
        Returns:
                start_locs : n_games x 4 numpy array with start locations
                goal_locs : n_games x 4 numpy array with goal locations
    """
   
    start_locs = np.zeros((n_games, 4))
    goal_locs = np.zeros((n_games, 4))
    
    # Calculate grid size 
    grid_size = im_size // step_size + 1
     
    start_locs[:,0:2] = (np.random.randint(0, grid_size , size = (n_games, 2))  * step_size).astype(int)
    goal_locs[:,0:2] = (np.random.randint(0, grid_size , size = (n_games, 2)) * step_size).astype(int)
  
    start_locs[:,2:] = patch_size
    goal_locs[:,2:]  = patch_size

    return start_locs , goal_locs 

class Game():

    def __init__(self, images , locs , image_size, STATS_dir, DEBUG = False, image_scale = 2, time_limit = 60, real_mode = False, trace = "none", number_steps = 10):
        """
            Game class.
                images: list of image paths to play on.
        """

        self.DEBUG = DEBUG
        self.real_mode = real_mode
        self.trace = trace

        self.RED = np.array([1.,0.0,0.0])
        self.BLUE = np.array([0.0,0.5,0.7])

        self.image_scale = image_scale

        self.STATS_dir = STATS_dir

        # Check that images are valid 
        if len(images) <= 0:
            raise(Exception("No images supplied"))
        for img in images:
            if not os.path.exists(img):
                raise(Exception("Image given does not exist:\t%s" % img))
        self.images = images
        self.n_images = len(images)

        # TODO Add option for this
        self.patch_size = 48 * self.image_scale

        # TODO Add option for this
        self.step_size = int(48 * 1.1) * self.image_scale

        # Check size of playing image all images will be resized to these dimensions
        if image_size == 0:
            # TODO - Use size of loaded imag
            pass
        elif image_size > 0:
            self.image_size = int(image_size) * self.image_scale
        elif image_size < 0:
            raise(Exception("Illegal image size selected"))
        
        # Select start locations
        if locs is None:
            # use random grid locations
            self.all_start_locs , self.all_goal_locs = sample_grid_locs( self.n_images , self.image_size, self.step_size, self.patch_size)
        else:
            self.all_start_locs , self.all_goal_locs = locs[:,0:4] , locs[:,4:]


        # Set game variables
        self.game_indx = 0
        self.done = False
        self.steps = 0
        self.max_steps = number_steps 
        self.current_loc = [0,0] 
        self.max_time = time_limit
        self.time_elapsed = 0   
        self.prev_outside = False
        self.quit = False
        self.wait = False

        self.n_rounds = len(images)

        # Calculate size of player area. should be 3x3 field with spaces between patches
        self.player_area_size = int(2 * self.step_size + self.patch_size)

        # Main layout 3x3 with 2x2 playing area
        self.main_fig , self.axes = plt.subplots(3 , 3 ,figsize = (10,8), gridspec_kw = {
                'width_ratios' : [1,1,2],
                'height_ratios' : [3,3,1],
            })
       
        # Set title
        if self.real_mode:
            self.main_fig.suptitle("Drone Navigation Game")
        else:
            self.main_fig.suptitle("Drone Navigation Game - Warmup!")

        # Remove axes for playing area  
        gs_play = self.axes[0,0].get_gridspec()
        for x in range(2):
            for y in range(2):
                self.axes[x,y].remove()
        
        # Add new big subplot for playing area
        self.player_axes = self.main_fig.add_subplot(gs_play[0:2,0:2])
        self.player_axes.set_title("Player Area")

        # Fill it in black for now
        self.player_area_canvas = self.player_axes.imshow(np.zeros((self.player_area_size , self.player_area_size, 3)))
        self.player_axes.set_xticks([])
        self.player_axes.set_yticks([])

        # Add imshow for goal patch
        self.goal_ax = self.axes[0,2]
        self.goal_ax.imshow(np.zeros((self.patch_size, self.patch_size, 3)))
        self.goal_ax.set_title("Goal Patch")
        self.goal_ax.set_yticks([])
        self.goal_ax.set_xticks([])

        # Then add position map
        self.pos_ax = self.axes[1,2]
        self.map_canvas = self.pos_ax.imshow(np.zeros((self.patch_size, self.patch_size,3))) 
        self.pos_ax.set_title("Position")
        self.pos_ax.set_yticks([])
        self.pos_ax.set_xticks([])

        # Then add textbox for time and steps counter
        self.counter_box = TextBox(self.axes[2,2] , "", "Game:   0 / %d\nTime left:   %d\nSteps:   0 / %d" % ( self.n_images,self.max_time, self.max_steps),  color = 'white', hovercolor = 'white')

        # Add timer to update time left
        self.timer = self.main_fig.canvas.new_timer(interval = 1000)
        self.timer.add_callback(self.timer_callback)

        # Add information area
        gs_info = self.axes[2,0].get_gridspec()
        for i in range(2):
            self.axes[2,i].remove()

        self.info_ax = self.main_fig.add_subplot(gs_info[2,0:2])
        self.info_ax.set_title("")
        
        self.info_box = TextBox(self.info_ax , "", "Welcome to the Drone Navigation Game!\nPress anywhere in the player area to start.",  color = 'white', hovercolor = 'white')

        self.main_fig.canvas.mpl_connect("button_press_event" , self.onClick)

        for ix in range(3):
            for iy in range(3):
                if ix == 1 and iy == 1:
                    continue
                self.player_axes.add_patch(patches.FancyBboxPatch(xy=(self.step_size * ix ,  self.step_size * iy),
                                                   width=self.patch_size,
                                                   height=self.patch_size,
                                                   linewidth=1.0,
                                                   boxstyle = patches.BoxStyle("Round",
                                                                               pad=0, 
                                                                               rounding_size=5),
                                                   edgecolor='none', alpha = 0.1, facecolor='#D0CEE2'))

        # Setup statistics for the games
        self.STATS_steps = np.zeros((len(self.images), 1))
        self.STATS_success = np.zeros((len(self.images), 1))
        self.STATS_final_distance = np.zeros((len(self.images), 1))
        self.STATS_time = np.zeros((len(self.images), 1))
        self.STATS_difficulty = np.zeros((len(self.images),1))


        self.started = False

    def timer_callback(self):
        if self.done:
            self.timer.stop()
        else:
            self.time_elapsed += 1
            self.counter_box.set_val("Game:   %d / %d\nTime left:   %d\nSteps:   %d / %d" % (self.game_indx+1,self.n_images,self.max_time - self.time_elapsed , self.steps, self.max_steps))

            if self.time_elapsed >= self.max_time:
                self.done = True

                self.info_box.set_val("You failed to reach the goal patch in time!")
                self.info_box.color = 'red'
                self.info_box.hovercolor = 'red'
                self.counter_box.color = 'red'
                self.counter_box.hovercolor = 'red'

                self.update_map_position(show_goal = True)

                # Save statistics for this game
                start_dist = np.abs(self.start_loc - self.goal_loc) // int(self.step_size)
                final_dist = np.abs(self.curr_loc - self.goal_loc) // int(self.step_size )
                difficulty = max( start_dist[0] , start_dist[1] ) # Diff == 1 means one step from start to goal
                final_distance = max(final_dist[0] , final_dist[1])

                self.STATS_steps[self.game_indx ] = self.steps
                self.STATS_success[self.game_indx ] = 1 if final_distance == 0 else 0 
                self.STATS_final_distance[self.game_indx ] = final_distance  
                self.STATS_time[self.game_indx] = self.time_elapsed  
                self.STATS_difficulty[self.game_indx] = difficulty

    def check_currently_outside(self):
        x , y = self.curr_loc[1] , self.curr_loc[0]
        return ( (x < 0 or x > self.image_size) or ( y < 0 or y > self.image_size))

    def update_player_canvas(self):
        temp_player_area = np.zeros((self.player_area_size, self.player_area_size, 3))
       

        # If current location is outside playing field only fill in black
        if not self.check_currently_outside():
            current_crop = self.image[self.curr_loc[0] : self.curr_loc[0]+self.curr_loc[2],
                                    self.curr_loc[1] : self.curr_loc[1] + self.curr_loc[3],:].copy()

            # Fill in middle of player canvas with current crop
            temp_player_area[self.step_size : self.step_size + self.patch_size ,self.step_size:self.step_size+self.patch_size,:] = current_crop
        
        self.player_area_canvas.set_data(temp_player_area) 

    def update_map_position(self , show_goal = False):
        temp_map = np.zeros((self.patch_size,self.patch_size,3))

        pos = np.round(self.curr_loc // int(self.step_size))
        size = self.patch_size // 5

        if self.trace != "none" and game.time_elapsed > 0 and not self.done:
            # Get all previous crops, resize them and put in map
            for s in range(self.steps + 1):
                if self.steps == self.max_steps:
                    break
                prev_loc = self.prev_locs[s, :].astype('int')
                x,y = prev_loc[1] , prev_loc[0]
                if ( (x < 0 or x > self.image_size) or ( y < 0 or y > self.image_size)):
                    continue

                # Get map position
                prev_pos = np.round(prev_loc // int(self.step_size)) 
                if self.trace == 'rgb':
                    crop = self.image[prev_loc[0]:prev_loc[0] + prev_loc[2],prev_loc[1]:prev_loc[1]+prev_loc[3],:]
                    crop = resize(crop , (size -2, size-2))
                    map_input = crop 
                elif self.trace == 'regular':
                    map_input = self.BLUE
                else:
                    raise(Exception("Unkown trace setting:\t%s" % self.trace))

                # Add to temp canvas
                temp_map[prev_pos[0] * size + 1: (prev_pos[0]+1) * size - 1, prev_pos[1] * size + 1 : (prev_pos[1]+1) * size - 1,:] = map_input

        if not self.check_currently_outside():
            temp_map[(pos[0]) * size  + 1: (pos[0]+1)*size -1, pos[1] * size + 1 : (pos[1] + 1) * size -1 , :] = 0.9
      
        if show_goal:
            goal_pos = self.goal_loc // self.step_size
            temp_map[(goal_pos[0]) * size:(goal_pos[0]+1) * size, goal_pos[1] * size : (goal_pos[1]+1) * size,:] = self.RED
        
        self.map_canvas.set_data(temp_map)

    def update_goal_patch(self):

        goal_crop = self.image[self.goal_loc[0]:self.goal_loc[0] + self.goal_loc[2],
                                self.goal_loc[1]:self.goal_loc[1] + self.goal_loc[3], :]

        self.goal_ax.imshow(goal_crop)

    def setup(self):
        """
            Setup the next game
        """
        
        self.done = False
        self.steps = 0
        self.time_elapsed = 0

        # Open image
        self.image = np.array(Image.open(self.images[self.game_indx]))
       
        # Resize image
        self.image = resize(self.image, (self.image_size, self.image_size))

        # Get start and goal locations
        self.start_loc = self.all_start_locs[self.game_indx , :].astype(int)
        self.goal_loc = self.all_goal_locs[self.game_indx , :].astype(int)
       
        self.curr_loc = self.start_loc.copy().astype(int)

        self.prev_locs = np.zeros((self.max_steps, 4))

        # Fill in crops in player area
        self.update_player_canvas()

        # Set goal patch image
        self.update_goal_patch() 

        self.update_map_position()

        self.info_box.color = 'white'
        self.info_box.hovercolor = 'white'
        self.counter_box.color = 'white'
        self.counter_box.hovercolor = 'white'

        self.info_box.set_val("A new game has started!\nClick in the player area to move.")
        self.counter_box.set_val("Game:   %d / %d\nTime left:   %d\nSteps:   0 / %d" % (self.game_indx+1,self.n_images, self.max_time, self.max_steps))

        self.timer.start()

    def run(self):
        plt.show()

    def update_current_loc(self, coords):

        # First find out what move was made 
        x_g , y_g = coords[0] // self.step_size, coords[1] // self.step_size 
        
        # Then map grid coords to a move
        if x_g == 1 and y_g == 0:
            move = np.array((-1,0)) # Up
        elif x_g == 2 and y_g == 0:
            move = np.array((-1,1)) # Up right
        elif x_g == 2 and y_g == 1:
            move = np.array((0,1)) # Right
        elif x_g == 2 and y_g == 2:
            move = np.array((1,1)) # Down right
        elif x_g == 1 and y_g == 2:
            move = np.array((1,0)) # Down
        elif x_g == 0 and y_g == 2:
            move = np.array((1,-1)) # Down left
        elif x_g == 0 and y_g == 1:
            move = np.array((0,-1)) # left
        elif x_g == 0 and y_g == 0:
            move = np.array((-1,-1))
        else:
            raise(Exception("Unknown move:\t(%d,%d)" % (x_g,y_g)))
        
        # Add this location to the previously visited locs
        self.prev_locs[self.steps , :] = self.curr_loc.copy()

        # Calculate new coordinates
        self.curr_loc[0:2] += move * int(self.step_size)
                   
    def check_valid_move(self, x,y):
        return ( x % self.step_size <= self.patch_size and y % self.step_size <= self.patch_size) and not (x // self.step_size == 1 and y // self.step_size == 1)

    def check_if_done(self):
        # USe L2 norm to determine if coords are equal
        return ( np.linalg.norm( self.curr_loc[0:2] - self.goal_loc[0:2])**2 < 1) or (self.steps >= self.max_steps)

    def save(self):
        """
            Save all generated statistics about this game.
        """
        # Only save results if we are in real mode
        if self.real_mode:

            # Create statistics directory
            os.makedirs(self.STATS_dir, exist_ok = True)

            # Find name of current run
            temp = 1
            self.STATS_dir_name = "run_%d" % temp
            while os.path.exists(os.path.join(self.STATS_dir, self.STATS_dir_name)):
                temp += 1
                self.STATS_dir_name = "run_%d" % temp
            
            self.STATS_dir_name = os.path.join(self.STATS_dir, self.STATS_dir_name)
            # Create save folder
            os.makedirs(self.STATS_dir_name)

            # Save statistics about human performance
            np.savez(os.path.join(self.STATS_dir_name, "statistics.npz") , 
                    steps = self.STATS_steps , success = self.STATS_success, 
                    final_distance = self.STATS_final_distance, 
                    time = self.STATS_time, difficulty = self.STATS_difficulty)


            # Save image names in separate file
            with open(os.path.join(self.STATS_dir_name, "image_files.txt"), 'w') as io:
                    for image in self.images:
                        print(image, file = io)


    def onClick(self , event):

        # Start game after first click
        if not self.started:
            self.started = True
            self.setup()

            self.info_box.set_val("The game has started!") 
            return
        
        if self.quit:
            plt.close('all')
            return
        
        if self.wait:
            self.wait = False
            self.setup()
            return


        if event.inaxes is not None and event.inaxes.title.get_text() == "Player Area" and not self.done:
            # Player clicked inside the player area.
            # See if valid move and if so update playing board
            # valid moves are in any of the 8 adjecent patches 

            if not self.check_valid_move(event.xdata, event.ydata):
                return

            self.info_box.set_val("")

            # move is valid calculate new location
            self.update_current_loc([event.xdata , event.ydata])

            # Update player area
            self.update_player_canvas() 
           
            self.update_map_position()

            self.steps += 1

            self.done = self.check_if_done() 
            
            if self.check_currently_outside(): 
                # We are outside update message
                self.prev_outside = True
                self.info_box.set_val("You have moved outside the image!")
                
            # If we are done display message
            if self.done:
                if ( np.linalg.norm( self.curr_loc[0:2] - self.goal_loc[0:2])**2 < 1):
                    self.info_box.set_val("You reached the goal patch!")
                    self.info_box.color = 'green'
                    self.info_box.hovercolor = 'green'
                    self.counter_box.color = 'green'
                    self.counter_box.hovercolor = 'green'
                else:
                    self.info_box.set_val("You reached the maximum number of steps!\n")
                    self.info_box.color = 'red'
                    self.info_box.hovercolor = 'red'
                    self.counter_box.color = 'red'
                    self.counter_box.hovercolor = 'red'
                    
                    self.update_map_position(show_goal = True)

            # Update number of steps
            self.counter_box.set_val("Game:   %d / %d\nTime left:   %d\nSteps:   %d / %d" % (self.game_indx+1,self.n_images,self.max_time - self.time_elapsed , self.steps, self.max_steps))

            plt.draw()

        elif self.done:

            # Save statistics for this game
            start_dist = np.abs(self.start_loc - self.goal_loc) // int(self.step_size)
            final_dist = np.abs(self.curr_loc - self.goal_loc) // int(self.step_size )
            difficulty = max( start_dist[0] , start_dist[1] ) # Diff == 1 means one step from start to goal
            final_distance = max(final_dist[0] , final_dist[1])

            self.STATS_steps[self.game_indx ] = self.steps
            self.STATS_success[self.game_indx] = 1 if final_distance == 0 else 0 
            self.STATS_final_distance[self.game_indx] = final_distance  
            self.STATS_time[self.game_indx ] = self.time_elapsed  
            self.STATS_difficulty[self.game_indx] = difficulty

            if self.DEBUG:
                print("Round has ended.")

                print("Start loc:\t%s" % self.start_loc)
                print("Goal loc:\t%s" % self.goal_loc)
                print("difficulty:\t%d" % difficulty)
                print("Step size:\t%d" % self.step_size)

                print("STATS_steps:\t%s" % self.STATS_steps[self.game_indx])
                print("STATS_success:\t%s" % self.STATS_success[self.game_indx])
                print("STATS_final_distance:\t%s" % self.STATS_final_distance[self.game_indx])
                print("STATS_time:\t%s" % self.STATS_time[self.game_indx])



            self.game_indx += 1
            if self.game_indx >= len(self.images):

                self.save()

                self.info_box.color = 'white'
                self.info_box.hovercolor = 'white'
                self.counter_box.color = 'white'
                self.counter_box.hovercolor = 'white'
                self.info_box.set_val("You have reached the end of the dataset.\nThank you for playing, press anywhere to quit!")

                mean_success = self.STATS_success.mean()
                mean_steps = self.STATS_steps.mean()

                self.counter_box.set_val("Mean success ratio:   %2.2f%%\nMean steps taken:   %2.2f" % (mean_success * 100, mean_steps))
                self.quit = True


                return  

            self.wait = True
            self.info_box.set_val("Press anywhere to start next game!") 


        else:
            # Pressed somewhere other than the playing area, ignore
            pass

if __name__ == '__main__':

    if args.image is None:
        # Use image folder
        locs = None         
        
        # If the given image_dir is None look for default folder
        if args.image_dir is None:
            image_dir = os.path.join("GameFiles")
        else:
            image_dir = args.image_dir
            #image_dir =         

        # Check that it exists
        if not os.path.exists(image_dir):
            print("Image folder does not exist:\t%s" % image_dir)
            exit(1)

        # Check if we are in warmup or real mode
        if args.real:
            split_file_path = os.path.join(image_dir, "split_file.csv")
        else:
            split_file_path = os.path.join(image_dir, "warmup_split_file.csv")
        
        stats_dir = os.path.join(image_dir, "statistics")

        # Check if split file available
        if os.path.exists(split_file_path):
            # Read images and locs from the split file
            split_file = pd.read_csv(split_file_path)
            
            # When running later shuffle order of images
            split_file = split_file.sample(frac = 1)

            images = list(split_file['image_id'])
            if args.real:
                images = [os.path.join(image_dir, "images" , x) for x in images]
            else:
                images = [os.path.join(image_dir, "warmup_images" , x)  for x in images]

            # TODO - read locs
            start_locs = np.concatenate((split_file[['start_y','start_x']] , 48 * np.ones((split_file.shape[0],2))), axis = 1 ) * args.image_scale
            goal_locs = np.concatenate((split_file[['goal_y','goal_x']] , 48 * np.ones((split_file.shape[0],2)) ), axis = 1) * args.image_scale

            locs = np.concatenate(( start_locs , goal_locs) , axis = 1)
        else:

            print("Split file does not exist:\t%s" % split_file_path)
            exit(1)

            """ 
            # Load all files ending with .jpg or .jpeg
            images = os.listdir(os.path.join(image_dir, "images"))
            images = list(filter(lambda f: f.endswith(".jpg") or f.endswith(".jpeg"), images))
            
            # Append path to the images
            images = [os.path.join(image_dir,"images",x) for x in images]
            """
        # Select start and goal locs
        if args.randomize_locs:
            # Randomize start and goal locations
            locs = None
        # If not locs has already been loaded from the split file 

    else:
        # Load a specific image
        images = [args.image]
    
    game = Game(images = images , locs = locs, image_size = args.image_size, 
            STATS_dir = stats_dir, DEBUG = args.debug, image_scale = args.image_scale, 
            time_limit = args.time_limit, real_mode = args.real, 
            trace = args.trace, number_steps = args.number_steps)
            
    game.run()





