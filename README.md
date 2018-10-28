# gym-snake
Implementation of gym environment for classic Snake game

gym-snake repo
September 21st, 2018


- Contains a Snake environment that uses OpenAI gym's common interface
- Implements the Classic game of Snake

Snake:

15 x 20 gridworld, a snake apears at the random initial position snake periodically takes an action chosen from:

0: no action, continue in the current direction
-1: turn left
1: turn right

Snake receives +1 reward when eats a piece of food, 0 if goes to an empty grid.
Upon death (happens when snake runs into its own body or runs into the boundaries), snake receives 
a penalty of -5 reward, which is designed to train the agent to avoid dying early on.

