import gym
import gym_snake
import sched,time


difficulties = ['Easy',
                'Normal',
                'Hard']

class SnakeGame:
    def __init__(self,env_name,difficulty='Normal'):
        self.env=gym.make(env_name)
        self.sch=sched.scheduler()
        self.done=False
        self.mode="human"
        self.diff=difficulty
        if difficulty=="Easy":
            self.rate=0.7
        elif difficulty=="Normal":
            self.rate=0.5
        else:
            self.rate=0.3
        self.action=0

    def start(self):
        print("Starting Snake, difficulty lvl: {}".format(self.diff))
        self.env.reset()
        self.sch.enter(0,1,self.do_once)
        self.sch.run()

    def do_once(self):
        self.env.render(self.mode)
        self.env.window.activate()
        self.sch.enter(self.rate,1,self.do_more)
            
        
    def do_more(self):
        self.env.window.dispatch_events()
        _,_,self.done,_=self.env.step(self.get_action()) 
        if self.done:
            self.end()
        else:            
            self.env.render(self.mode)            
            self.sch.enter(self.rate,1,self.do_more)

    def get_action(self):
        return self.env.window.action_input
        
        
    def end(self):
        self.env.close()
        for e in self.sch.queue:
            self.sch.cancel(e)
        



if __name__=="__main__":
    lvl=3
    mode="human"
    while lvl not in range(3):
        v = input('Enter difficulty lvl (1-easy, 2-normal, 3-hard): ')
        lvl=int(v)-1
    game = SnakeGame("snake-v0",difficulties[lvl])
    game.start()



    
