# gym-snake
Implementation of gym environment for classic Snake game

September 21st, 2018

Snake:

15 x 20 gridworld, a snake apears at the random initial position snake periodically takes an action chosen from:

0: no action, continue in the current direction
-1: turn left
1: turn right

Snake receives +1 reward when eats a piece of food, 0 if goes to an empty grid.
Upon death (happens when snake runs into its own body or runs into the boundaries), snake receives 
a penalty of -5 reward, which is designed to train the agent to avoid dying early on.

The game state is represented by a 15x20 numpy array that is updated every step.
Pyglet package is used to render the game in 'human' mode. The render mode does not
support video generation yet. But the 'replay' mode allows you to save the game (initial state + complete action history),
which can be used later to generate videos if desired.

To see what the environment does, setup the environment then run playSnake.py.
Left arrow key tells the snake to turn left, Right arrow key tells the snake to turn right.
The default action is to continue straight ahead. The game will have prompt you to choose from
3 different speeds at the beginning.

