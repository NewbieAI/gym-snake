from setuptools import setup

setup(name='gym_snake',
      version='0.0.1',
      url="http://github.com/NewbieAI/gym-snake",
      author="Mingzhi Tian",
      license="MIT",
      packages=['gym_snake','gym_snake.envs'],
      install_requires=['gym','numpy','pyglet']
      )
