import driving_env_2nd_lay_ddpg
import numpy as np
import gym
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

#https://www.it-swarm.dev/ja/tensorflow/tensorflow%EF%BC%9Ainternalerror%EF%BC%9Ablas-sgemm%E3%81%AE%E8%B5%B7%E5%8B%95%E3%81%AB%E5%A4%B1%E6%95%97%E3%81%97%E3%81%BE%E3%81%97%E3%81%9F/824534956/
if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


ENV_NAME = 'driving_seg2_ddpg-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
#離散的でなく連続的な行動空間を持つことを保証する
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]
# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))

actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('linear'))
print(actor.summary())
action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())
try:
    actor.load_weights('ddpg_{}_weights_actor.h5f'.format(ENV_NAME))
    print('----------------actor loading completed----------------')
except:
    pass
try:
    critic.load_weights('ddpg_{}_weights_critic.h5f'.format(ENV_NAME))
    print('----------------critic loading completed----------------')
except:
    pass
memory = SequentialMemory(limit=50000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=0.1, mu=0., sigma=0.4)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input, memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100, random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
agent.fit(env, nb_steps=50000, visualize=True, verbose=1)
agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
agent.test(env, nb_episodes=5, visualize=True)