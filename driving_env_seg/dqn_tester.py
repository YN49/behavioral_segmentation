import driving_env
import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, Reshape
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

#https://www.it-swarm.dev/ja/tensorflow/tensorflow%EF%BC%9Ainternalerror%EF%BC%9Ablas-sgemm%E3%81%AE%E8%B5%B7%E5%8B%95%E3%81%AB%E5%A4%B1%E6%95%97%E3%81%97%E3%81%BE%E3%81%97%E3%81%9F/824534956/
if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

ENV_NAME = 'driving_seg-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
# Next, we build a very simple model.

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
print(model.summary())

try:
    model.load_weights('dqn_{}_weights.h5f'.format(ENV_NAME))
    print('1層目をロード')
except:
    pass
print("----------Completed construction of 1st layer----------")

#ロードが完了したことを伝える
dqntes_main = np.fromfile('強化学習/行動細分化/driving_env/driving_env_seg/dqntes_main.npy', dtype="bool")
dqntes_main[0] = True
dqntes_main.tofile('強化学習/行動細分化/driving_env/driving_env_seg/dqntes_main.npy')

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=5000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
            target_model_update=1e-2, policy=policy, enable_double_dqn=True)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])


# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=10, visualize=True)