import torch
from dart_env import DartEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import os

def make_env(rank, seed=0):
    def _init():
        env = DartEnv(bvh_filename="bvh/walk1_subject1_short.bvh", use_gl=False)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def main():
    # TODO: mlp policy + height map
    policy_kwargs = {
        "activation_fn": torch.nn.ReLU,
        "net_arch": [{
            "pi": [256, 256],
            "vf": [256, 256],
        }],
    }

    # env = DartEnv(bvh_filename="bvh/walk1_subject1_short.bvh", use_gl=False)
    num_cpu = 4
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    # if os.path.isfile("walk.model.zip"):
    #     model = PPO.load("walk.model.zip", env)
    #     print("load from previous model...")
    # else:
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=2048000)
    model.save("walk.model.zip")

if __name__ == "__main__":
    main()
