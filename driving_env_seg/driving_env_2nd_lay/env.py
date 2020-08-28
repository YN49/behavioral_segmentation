from keras import backend as K

import matplotlib.pyplot as plt
import argparse

from PIL import Image

import math

import os

import gym
import numpy as np
import gym.spaces

class ENV(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}
    MAX_STEPS = 50
    #感度、移動速度
    SENSITIVITY = 0.5

    #同期させてね
    VIEW_SIZE = (15,15)

    def __init__(self):
        super().__init__()
        # action_space, observation_space, reward_range を設定する
        self.action_space = gym.spaces.Discrete(5)  # 上下左右決定
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(2,)
        )
        self.reward_range = [-1., 100.]

        self._reset()

        self.viewer = None

    def _reset(self):
        # 諸々の変数を初期化する
        self.pos = np.array([0, 0])#画面の座標x,画面のディレクトリ
        #ターゲット格納
        self.target = np.array([0, 0])
        self.done = False
        self.steps = 0
        self.original_dim = self.VIEW_SIZE[0] * self.VIEW_SIZE[1]
        self.sync = np.array([False,False],dtype="bool")

        try:
            loaded_array = np.load('強化学習/行動細分化/driving_env/driving_env_seg/env1_env2_comu.npz')
            self.target = loaded_array['arr_0']
            self.obs = loaded_array['arr_1']
        except FileNotFoundError:
            self.target = np.array([0,0])
            self.obs = np.zeros(self.original_dim)
            ###env2-env1の通信ファイルを作成
            np.savez('強化学習/行動細分化/driving_env/driving_env_seg/env1_env2_comu.npy', self.target, self.obs)

        #同期ファイル確認読み込み
        try:
            self.sync = np.load('強化学習/行動細分化/driving_env/driving_env_seg/sync.npz')
        except FileNotFoundError:
            pass
        self.sync = np.array([False,False],dtype="bool")

        
        return self._observe()


    def _step(self, action):

        #同期を保存 ついでに初期化
        self.sync[1] = False
        np.save('強化学習/行動細分化/driving_env/driving_env_seg/sync',self.sync)

        #通信用ファイル読み込み
        loaded_array = np.load('強化学習/行動細分化/driving_env/driving_env_seg/env1_env2_comu.npz')
        self.target = loaded_array['arr_0']
        self.obs = loaded_array['arr_1']


        # 1ステップ進める処理を記述。戻り値は observation, reward, done(ゲーム終了したか), info(追加の情報の辞書)
        if action == 0:
            self.pos = self.pos + np.array([self.SENSITIVITY, 0])
        elif action == 1:
            self.pos = self.pos + np.array([-self.SENSITIVITY, 0])
        elif action == 2:
            self.pos = self.pos + np.array([0, self.SENSITIVITY])
        elif action == 3:
            self.pos = self.pos + np.array([0, -self.SENSITIVITY])
        else:#ターゲット決定
            self.target = self.pos

        self.action = action
        self.steps = self.steps + 1
        observation = self._observe()
        reward = self._get_reward()
        self.done = self._is_done()

        ###env2-env1の通信ファイルを作成
        np.savez('強化学習/行動細分化/driving_env/driving_env_seg/env1_env2_comu.npz', self.target, self.obs)

        #同期を保存 処理終了のためTrueに
        self.sync[1] = True
        np.save('強化学習/行動細分化/driving_env/driving_env_seg/sync',self.sync)
        while True:
            self.sync = np.load('強化学習/行動細分化/driving_env/driving_env_seg/sync.npy')
            #相手の処理が終了したらbreak
            if self.sync[0]:
                break


        return observation, reward, self.done, {}

    def _render(self, mode='human', close=False):
        # human の場合はコンソールに出力。ansiの場合は StringIO を返す
        os.system('cls')
        

    def _close(self):
        pass

    def _seed(self, seed=None):
        pass

    def _get_reward(self):
        # 報酬を返す。報酬の与え方が難しいが、ここでは
        # - ゴールにたどり着くと 100 ポイント
        # - ダメージはゴール時にまとめて計算
        # - 1ステップごとに-1ポイント(できるだけ短いステップでゴールにたどり着きたい)
        # とした
        return -1

    def _observe(self):#今の居場所の座標を人工知能に入力
        return self.pos
    
    def _is_done(self):
        # 今回は最大で self.MAX_STEPS までとした
        if self.steps > self.MAX_STEPS:
            return True
        else:
            return False
