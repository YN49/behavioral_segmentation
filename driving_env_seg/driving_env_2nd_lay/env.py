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
            shape=(4,)
        )
        self.reward_range = [-2000., 15000.]

        self._reset()

        self.viewer = None

    def _reset(self):
        # 諸々の変数を初期化する
        self.pos = np.array([0, 0],dtype="float64")#画面の座標x,画面のディレクトリ
        #ターゲット格納
        self.target = np.array([0, 0],dtype="float64")
        self.done = False
        self.steps = 0
        self.original_dim = self.VIEW_SIZE[0] * self.VIEW_SIZE[1]
        self.sync1_2 = np.array([False,False],dtype="bool")


        #同期ファイルを初期化
        self.sync1_2 = np.array([False],dtype="bool")
        self.sync1_2.tofile('強化学習/行動細分化/driving_env/driving_env_seg/sync1_2.npy')

        #終了伝達ファイルを初期化
        self.done_signal = np.array([False],dtype="bool")
        self.done_signal.tofile('強化学習/行動細分化/driving_env/driving_env_seg/done_signal.npy')
        
        return self._observe()


    def _step(self, action):

        if self.steps == 0:
            #準備ができたので準備完了信号を出す
            self.sync1_2[0] = True
            self.sync1_2.tofile('強化学習/行動細分化/driving_env/driving_env_seg/sync1_2.npy')
            while True:
                self.sync1_2 = np.fromfile('強化学習/行動細分化/driving_env/driving_env_seg/sync1_2.npy', dtype="bool")
                #Falseにenv1がしてきたのでそれを感知して処理開始
                #多分同時にファイル開かれるとサイズが0になっちゃうからそれを防止する
                try:
                    if not self.sync1_2[0]:
                        break
                except IndexError:
                    pass

        self.pre_target = self.target
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
            self.target[0] = self.pos[0]
            self.target[1] = self.pos[1]


        self.action = action
        self.steps = self.steps + 1
        observation = self._observe()

        reward = self._get_reward()
        self.done = self._is_done()

        ###ターゲットの受け渡し
        self.target.tofile('強化学習/行動細分化/driving_env/driving_env_seg/target.npy')

        #準備ができたので準備完了信号を出す
        self.sync1_2[0] = True
        self.sync1_2.tofile('強化学習/行動細分化/driving_env/driving_env_seg/sync1_2.npy')
        while True:
            self.sync1_2 = np.fromfile('強化学習/行動細分化/driving_env/driving_env_seg/sync1_2.npy', dtype="bool")
            #Falseにenv1がしてきたのでそれを感知して処理開始
            #多分同時にファイル開かれるとサイズが0になっちゃうからそれを防止する
            try:
                if not self.sync1_2[0]:
                    break
            except IndexError:
                pass

        #print(observation,"---------------------------------------------------------------")
        return observation, reward, self.done, {}

    def _render(self, mode='human', close=False):
        # human の場合はコンソールに出力。ansiの場合は StringIO を返す
        #os.system('cls')
        pass
        

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
        reward = np.fromfile('強化学習/行動細分化/driving_env/driving_env_seg/rew_signal.npy', dtype="int64")[0]
        #何もターゲットが設定されないのを避ける
        #if reward == -5:
        #    reward = -20
        """
        if self.action == 4 and not (self.pre_target[0] == self.target[0] and self.pre_target[1] == self.target[1]):
            reward = reward + 10"""
        return reward

    def _observe(self):#今の居場所の座標とVAEのOBSを人工知能に入力
        #通信用ファイル読み込み
        #VAEの結果を取得　視界を獲得
        return np.concatenate([np.fromfile('強化学習/行動細分化/driving_env/driving_env_seg/encoded_obs.npy',dtype="float64"),self.pos],0)
    
    def _is_done(self):
        #相手が終了したか取得する ENV1からの通信 ENV2からの通信
        done_signal = np.fromfile('強化学習/行動細分化/driving_env/driving_env_seg/done_signal.npy', dtype="bool")

        # 今回は最大で self.MAX_STEPS までとした
        if self.steps > self.MAX_STEPS:
            #MAXSTEPで完全に終了するため伝達を行う
            done_signal[0] = True
            done_signal.tofile('強化学習/行動細分化/driving_env/driving_env_seg/done_signal.npy')
            return True
        #終了伝達がきたら終了
        elif done_signal[0]:
            return True
        else:
            return False
