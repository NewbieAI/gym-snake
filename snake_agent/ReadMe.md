# An agent to solve the Snake Environment

I don't want to use an off-the-shelf agent to solve this environment. The whole point of doing this is to learn how to implement an RL algorithm myself, building all the pieces and put them together.

The current idea is to use a Conv Net (using tensorflow) that takes the game state (15x20 numpy array) and outputs a discrete distribution representing a stochastic policy through a final softmax layer. The snake env is explicitly designed to have the Markov property, so I don't expect it to be hard to solve.

Based on the stochastic policy, the agent will then sample the snake environment, generating multiple trajectories. Using the collected data, the agent will (most likely) run vanilla policy gradient, 0-mean the advantage function using a baseline, then update the policy. Still need to put all the plumbings together. Honestly I have no idea how long it will take to train.
