# dependencies include gym, numpy, and pyglet
import gym
from gym import error, spaces, utils
import numpy as np
import pyglet as pt
from pyglet.gl import *
from datetime import datetime

# starting with a small game size seems a good idea
GAME_SIZE = (15,20)

# look up table
ACTION_MAP = {0:4,
              1:1,
              2:2,
              3:3,
              4:4,
              5:1}


class SnakeEnv(gym.Env):
    metadata = {'render.modes':['human','replay']}
    def __init__(self):
        '''
        Classic snake game, screen is small 15x20
        
        Food appears at a random available space
        everytime the previous one is eaten, extrahard
        mode include bonus food that's available for
        a limited number of time steps.

        Snake dies if running into border or into itself.
        Game is over if snake dies
        
        action space [-1,1]:
        0: keep straight
        1: turn clockwise
        -1: turn counterclockwise

        observation space 15x20 grid
        0: location is empty
        1: snake body, going left
        2: snake body, going up
        3: snake body, going right
        4: snake body, going down
        5: snake head, action pending
        -1: location is food

        Note: a special value (5) is assigned to the head
        so that we do not need to feed consecutive
        game states to the agent in order for it to
        know where the snake is going.
        '''

        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-1,5,GAME_SIZE,int)
        # default reward range is OK
        self.state = np.zeros(GAME_SIZE,int)
        self.snakelength = 0
        self.head = [0,0]
        self.tail = [0,0]
        self.direction = 0
        

        # objects dealing with rendering and saving replay
        self.window = None
        self.f = None


        # list of locations whose value changed
        # when rendering a new frame, we only
        # need to update places that has changed
        self.last_changed = []

        ####
        self.info = {'status':'Inializing',
                     'total_reward':0,
                     'episode_length':0,
                     }
        # since classic Snake is a complete information game
        # the observation_space alone is sufficient to represent
        # the game state.
    def step(self,action):
        '''
        Notes copied from gym.core.py
        input: "action" object provided by the environment
        returns:
        observation (object): agent's observation of the current environment
        reward (float): amount of reward
        done (boolean): whether the game has ended
        info (dict): contains auxiliary diagnostic information
        '''         
        # moves the snake, if one piece of food is eaten, another piece
        # is generated and placed in the environment.
        self.last_changed.clear()
        self.direction = ACTION_MAP[self.direction+action]
        self.state[self.head[0],self.head[1]]=self.direction
        self.last_changed.append((self.head[0],self.head[1],self.direction))

        # logic determining where the head should be next:
        if self.direction%2==1:
            # 1,3 are horizontal directions
            if self.direction==1: # going left one step
                self.head[1]-=1
            else: # going right one step
                self.head[1]+=1
        else:
            # 2,4 are vertical directions
            if self.direction==2: # going up one step
                self.head[0]-=1
            else: # going down one step
                self.head[0]+=1

        # inboard: whether snake remains on the board
        inboard = (self.head[0] in range(GAME_SIZE[0])
                      ) and (self.head[1] in range(GAME_SIZE[1]))

        if inboard and self.state[self.head[0],self.head[1]]<=0: # game continues
            
            if self.state[self.head[0],self.head[1]]<0:
                # snake eats food, grows by 1
                reward = 1
                self.state[self.head[0],self.head[1]]=5
                self.last_changed.append((self.head[0],self.head[1],5))
                self.place_food()
                self.snakelength+=1
                self.info['total_reward']+=1
            else:
                # snake does not reach food, shifts by 1
                reward = 0
                self.state[self.head[0],self.head[1]]=5
                self.last_changed.append((self.head[0],self.head[1],5))
                tail_direction = self.state[self.tail[0],self.tail[1]]
                self.state[self.tail[0],self.tail[1]]=0
                self.last_changed.append((self.tail[0],self.tail[1],0))
                if tail_direction%2==1:
                    if tail_direction==1:# left
                        self.tail[1]-=1
                    else: # right
                        self.tail[1]+=1
                    
                else:
                    if tail_direction==2:# up
                        self.tail[0]-=1
                    else: # down
                        self.tail[0]+=1
            
            self.info['status']='Ongoing'
            self.info['episode_length']+=1
            return self.state,reward,False,self.info
        else: # game ends, receives -5 for dying
            self.state[:,:]=0
            self.info['status']='Ended'
            self.close()
            return self.state,-5,True,self.info
                   
    def reset(self):
        # resets the state of the environment and returns
        # the initial observation
        self.place_snake()
        self.place_food()
        
        return self.state # we observe the entire state

    def place_snake(self):
        # The initial snake faces a random direction
        # has total body length of 5, including its head
        # The snake is randomly placed such that every
        # initial position is equally possible
        
        direction = np.random.randint(1,5)
        self.direction = direction
        if direction%2==1:
            if direction==1: # 1, going left
                x_loc = np.random.randint(0,GAME_SIZE[0])
                y_loc = np.random.randint(0,GAME_SIZE[1]-4)
                self.head = [x_loc,y_loc]
                self.tail = [x_loc,y_loc+4]
                self.state[x_loc,y_loc]=5
                self.state[x_loc,y_loc+1:y_loc+5]=direction
            else: # 3 going right
                x_loc = np.random.randint(0,GAME_SIZE[0])
                y_loc = np.random.randint(4,GAME_SIZE[1])
                self.head = [x_loc,y_loc]
                self.tail = [x_loc,y_loc-4]
                self.state[x_loc,y_loc]=5
                self.state[x_loc,y_loc-4:y_loc]=direction
        else:
            if direction==2: # 2, going up
                x_loc = np.random.randint(0,GAME_SIZE[0]-4)
                y_loc = np.random.randint(0,GAME_SIZE[1])
                self.head = [x_loc,y_loc]
                self.tail = [x_loc+4,y_loc]
                self.state[x_loc,y_loc]=5
                self.state[x_loc+1:x_loc+5,y_loc]=direction
            else: # 4, going down
                x_loc = np.random.randint(4,GAME_SIZE[0])
                y_loc = np.random.randint(0,GAME_SIZE[1])
                self.head = [x_loc,y_loc]
                self.tail = [x_loc-4,y_loc]
                self.state[x_loc,y_loc]=5
                self.state[x_loc-4:x_loc,y_loc]=direction

        self.snakelength = 5
        
    def place_food(self):
        # Randomly drops a piece of food into the game
        empty_spaces = (self.state==0).nonzero()
        if empty_spaces[0].shape[0]: # check if the board is full
            new_loc = np.random.randint(empty_spaces[0].shape[0])
            self.state[empty_spaces[0][new_loc],empty_spaces[1][new_loc]]=-1
            self.last_changed.append((empty_spaces[0][new_loc],empty_spaces[1][new_loc],-1))
                
    def render(self, mode='human'):
        # currently having trouble with rendering
        # just gonna steal gym.classic_control.rendering for this project
        if mode == 'human':
            if self.window==None:
                self.window = SnakeWindow(initial_state=self.state)
                self.window.draw()
                self.window.dispatch_events()
            else:
                #self.window.dispatch_events()
                self.window.update(self.last_changed)
                self.window.draw()
                self.window.dispatch_events()
                self.window.action_input=0
        elif mode == 'replay':
            if self.f==None:
                filename = datetime.now().strftime("%Y%b%d_%H_%M_%S")
                self.f = open(filename,'w+')
                self.f.write(self.state.__str__()+"\n")
            else:
                self.f.write(self.last_changed.__str__()+"\n")
                

    def close(self):
        if self.window:
            self.window.close()
            self.window.has_exit=True
        if self.f:
            self.f.close()
        
    def __str__(self):
        print(self.__name__)

class SnakeWindow(pt.window.Window):
    # a game window object useful
    # for rending in 'human' mode
    
    def __init__(self,
                 size=GAME_SIZE,
                 initial_state=np.zeros(GAME_SIZE,int)):
        super().__init__(size[1]*20,size[0]*20,caption='Snake v1.0 (gym env)')
        self.size=size
        self.batch = pt.graphics.Batch()
        self.vertices = [[None for _ in range(size[1])]
                         for _ in range(size[0])]
        #print(len(self.vertices),len(self.vertices[0]))
        for i in range(size[0]):
            for j in range(size[1]):                
                self.add_(i,j,initial_state[i,j])
        self.draw()
        self.dispatch_events()

        self.action_input=0
                

    def add_(self,i,j,val):
        # adds a GL_QUAD primitive to be rendered
        # as part of a pyglet.graphic.Batch object

        # the input arguments (i,j) refers to the
        # location of the primitive within the game state,
        # i=row, j=column, both are a 0-indexed, starting
        # from the upper left corner

        # val refers to the value of the game state at
        # the location
        
        if val: # we only bother with non empty places
            if val==5: # head is a blueish square block
                coordinates = (20*(j)+4,
                               20*(self.size[0]-i-1)+15,
                               20*(j)+15,
                               20*(self.size[0]-i-1)+15,
                               20*(j)+15,
                               20*(self.size[0]-i-1)+4,
                               20*(j)+4,
                               20*(self.size[0]-i-1)+4)
                color = (0,155,255)
            elif val<0: # food is a smallish red block
                coordinates = (20*(j)+6,
                               20*(self.size[0]-i-1)+13,
                               20*(j)+13,
                               20*(self.size[0]-i-1)+13,
                               20*(j)+13,
                               20*(self.size[0]-i-1)+6,
                               20*(j)+6,
                               20*(self.size[0]-i-1)+6)
                color = (255,100,100)
            elif val%2==1:
                if val==1: # left, body is green rectangle
                    coordinates = (20*(j)-5,
                                   20*(self.size[0]-i-1)+15,
                                   20*(j)+15,
                                   20*(self.size[0]-i-1)+15,
                                   20*(j)+15,
                                   20*(self.size[0]-i-1)+4,
                                   20*(j)-5,
                                   20*(self.size[0]-i-1)+4,)
                    color = (0,255,100)
                else: # right, green rectangle
                    coordinates = (20*(j)+4,
                                  20*(self.size[0]-i-1)+15,
                                  20*(j)+24,
                                  20*(self.size[0]-i-1)+15,
                                  20*(j)+24,
                                  20*(self.size[0]-i-1)+4,
                                  20*(j)+4,
                                  20*(self.size[0]-i-1)+4,)
                    color = (0,255,100)
            else:
                if val==2: # up, green rectangle
                    coordinates = (20*(j)+4,
                                  20*(self.size[0]-i-1)+24,
                                  20*(j)+15,
                                  20*(self.size[0]-i-1)+24,
                                  20*(j)+15,
                                  20*(self.size[0]-i-1)+4,
                                  20*(j)+4,
                                  20*(self.size[0]-i-1)+4,)
                    color = (0,255,100)
                else: # down, green rectangle
                    coordinates = (20*(j)+4,
                                  20*(self.size[0]-i-1)+15,
                                  20*(j)+15,
                                  20*(self.size[0]-i-1)+15,
                                  20*(j)+15,
                                  20*(self.size[0]-i-1)-5,
                                  20*(j)+4,
                                  20*(self.size[0]-i-1)-5,)
                    color = (0,255,100)
            #print(i,j,coordinates,color)
            
            self.vertices[i][j]=self.batch.add(4,pt.gl.GL_QUADS,None,('v2i',coordinates),('c3B',color*4))
        else:
            self.vertices[i][j]=None
            

    def update(self,last_changed):
        for point in last_changed:
            if self.vertices[point[0]][point[1]]:
                self.vertices[point[0]][point[1]].delete()
            self.add_(point[0],point[1],point[2])

    def draw(self,): 
        self.clear()
        ##
        self.batch.draw()
        ##
        self.flip()

    def on_key_release(self,symbol,modifier):
        if symbol==pt.window.key.LEFT:
            self.action_input=-1
        if symbol==pt.window.key.RIGHT:
            self.action_input=1
        


        
