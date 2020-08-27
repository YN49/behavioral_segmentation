from gym.envs.registration import register

register(
    id='driving_seg-v0',
    entry_point='driving_env.env:ENV'
)