import driving_env_2nd_lay_ddpg
import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import NAFAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor

import tensorflow as tf

class PendulumProcessor(Processor):
    def process_reward(self, reward):
        # The magnitude of the reward can be important. Since each step yields a relatively
        # high reward, we reduce the magnitude by two orders.
        return reward / 100.

#https://www.it-swarm.dev/ja/tensorflow/tensorflow%EF%BC%9Ainternalerror%EF%BC%9Ablas-sgemm%E3%81%AE%E8%B5%B7%E5%8B%95%E3%81%AB%E5%A4%B1%E6%95%97%E3%81%97%E3%81%BE%E3%81%97%E3%81%9F/824534956/
if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
"""
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


ENV_NAME = 'driving_seg2_ddpg-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
#離散的でなく連続的な行動空間を持つことを保証する
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]



# Build all necessary models: V, mu, and L networks.
V_model = Sequential()
V_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
V_model.add(Dense(16))
V_model.add(Activation('relu'))
V_model.add(Dense(16))
V_model.add(Activation('relu'))
V_model.add(Dense(16))
V_model.add(Activation('relu'))
V_model.add(Dense(1))
V_model.add(Activation('linear'))
print(V_model.summary())

mu_model = Sequential()
mu_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
mu_model.add(Dense(16))
mu_model.add(Activation('relu'))
mu_model.add(Dense(16))
mu_model.add(Activation('relu'))
mu_model.add(Dense(16))
mu_model.add(Activation('relu'))
mu_model.add(Dense(nb_actions))
mu_model.add(Activation('linear'))
print(mu_model.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
x = Concatenate()([action_input, Flatten()(observation_input)])
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(((nb_actions * nb_actions + nb_actions) // 2))(x)
x = Activation('linear')(x)
L_model = Model(inputs=[action_input, observation_input], outputs=x)
print(L_model.summary())


try:
    V_model.load_weights('naf_{}_weights_V_model.h5f'.format(ENV_NAME))
    print('----------------actor loading completed----------------')
except:
    pass
try:
    mu_model.load_weights('naf_{}_weights_mu_model.h5f'.format(ENV_NAME))
    print('----------------actor loading completed----------------')
except:
    pass
try:
    L_model.load_weights('naf_{}_weights_L_model.h5f'.format(ENV_NAME))
    print('----------------critic loading completed----------------')
except:
    pass

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
processor = PendulumProcessor()
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
agent = NAFAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                 memory=memory, nb_steps_warmup=100, random_process=random_process,
                 gamma=.99, target_model_update=1e-3, processor=processor)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
agent.fit(env, nb_steps=50000, visualize=True, verbose=1, nb_max_episode_steps=200)

# After training is done, we save the final weights.
agent.save_weights('naf_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=200)