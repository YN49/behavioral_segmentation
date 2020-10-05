from gym.envs.registration import register

register(
    id='driving_seg2_ddpg-v0',
    entry_point='driving_env_2nd_lay_ddpg.env:ENV'
)