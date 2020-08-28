from gym.envs.registration import register

register(
    id='driving_seg2-v0',
    entry_point='driving_env_2nd_lay.env:ENV'
)