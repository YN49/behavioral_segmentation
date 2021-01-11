import driving_env
import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, Reshape
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import tensorflow as tf

#https://www.it-swarm.dev/ja/tensorflow/tensorflow%EF%BC%9Ainternalerror%EF%BC%9Ablas-sgemm%E3%81%AE%E8%B5%B7%E5%8B%95%E3%81%AB%E5%A4%B1%E6%95%97%E3%81%97%E3%81%BE%E3%81%97%E3%81%9F/824534956/
if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()


#メモリ使用制限
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

ENV_NAME = 'driving_seg-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
# Next, we build a very simple model.
"""
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
#flatten_1 (Flatten)          (None, 4)                 0
#4だから((2, 2), input_shape=(4, ))/2だったら((2, 1), input_shape=(2, ))
model.add(Reshape((4, 1), input_shape=(4, )))
#input_shape=(2, 2)=前のと同じ
model.add(LSTM(50, input_shape=(4, 1), 
        return_sequences=False))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
"""


model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())


try:
    model.load_weights('dqn_test_{}_weights.h5f'.format(ENV_NAME))
    print('ロード')
except:
    pass
print("----------Completed construction of 1st layer----------")

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=5000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
            target_model_update=1e-2, policy=policy, enable_double_dqn=True)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])



#===========================1層目を学習します===========================

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
history = dqn.fit(env, nb_steps=200000, visualize=False, verbose=1)

# After training is done, we save the final weights.
dqn.save_weights('dqn_test_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

import matplotlib.pyplot as plt
import seaborn as sns
# 結果を表示

np.save('episode_reward_DQN_test',history.history["episode_reward"])
sns.set()
plt.subplot(1,1,1)
plt.plot(history.history["episode_reward"])
plt.xlabel("episode")
plt.ylabel("reward")

plt.show()  # windowが表示されます。

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=100, visualize=True)