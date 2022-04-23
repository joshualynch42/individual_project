from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
env = make_atari_env('PongNoFrameskip-v4', n_envs=8, seed=0)
# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)

model = PPO(policy='CnnPolicy', env=env, verbose=1,  n_steps= 128)
model.learn(total_timesteps=2500000)

# model.save("D:\Josh\github\individual_project\simulation\sim_agents\PPO_agent")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
