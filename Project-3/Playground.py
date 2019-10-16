import environment as Env
from environment import Environment


env = Environment('BreakoutNoFrameskip-v4', "", atari_wrapper=True, test=False)
n = env.env.action_space
print("dsafda")

