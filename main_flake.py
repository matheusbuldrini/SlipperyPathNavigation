import gymnasium
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


# from BFSSolver import BFSSolver
from HumanSolver import HumanSolver

def init_env(map_size = 4, is_slippery = True, success_rate = 1.0, seed = 0):
    env = gymnasium.make(
            'FrozenLake-v1',
            render_mode="human",
            desc=generate_random_map(size=map_size, seed=seed),
            map_name=None,
            is_slippery=is_slippery,
            success_rate=success_rate,
            reward_schedule=(1, 0, 0)
            )
    env.reset(seed=seed)
    return env


def run(steps=1000, seed = 0):
    env = init_env(seed=seed)
    solver = HumanSolver(env)
    for _ in range(steps):
        env.render()
        observation, reward, terminated, truncated, info = env.step(solver.get_next_action())
        
        if float(reward) > 0:
            print("reward received!")
            break

        if terminated:
            print("terminated!")
            break

        if truncated:
            print("truncated!")
            break

    env.close()

run(seed=0)