
from keras import backend as K

import matplotlib.pyplot as plt
import argparse


from PIL import Image

import math

import os

import gym
import numpy as np
import gym.spaces

import pygame
from pygame.locals import *




class PIC(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}
    MAX_STEPS = 50
    f_model = './model'

    PIC = np.array(Image.open('強化学習/pic_env_random/MAP_PIC.png').convert('L'))

    WIDTH = 950
    HEIGHT = 450


    def __init__(self):
        super().__init__()
        # action_space, observation_space, reward_range を設定する
        self.action_space = gym.spaces.Discrete(4)  # 東西南北設置破壊
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(4,)
        )
        self.reward_range = [-1., 100.]

        self._reset()

        self.viewer = None

    def _reset(self):
        # 諸々の変数を初期化する
        self.pos = np.array([8, 8], dtype=int)#画面の座標x,画面のディレクトリ
        self.done = False
        self.steps = 0

        self.reset_rend = False
        
        #self.TARGET = np.array([3,3])
        self.TARGET = np.random.randint(2,self.PIC.shape[0]-2,(2,))#ターゲットとりあえずランダム設定



        return self._observe()


        

    def _step(self, action):
        # 1ステップ進める処理を記述。戻り値は observation, reward, done(ゲーム終了したか), info(追加の情報の辞書)
        if action == 0:
            next_pos = self.pos + np.array([1, 0])
        elif action == 1:
            next_pos = self.pos + np.array([-1, 0])
        elif action == 2:
            next_pos = self.pos + np.array([0, 1])
        else:
            next_pos = self.pos + np.array([0, -1])
        
        if 2 <= next_pos[0] < self.PIC.shape[0] - 2 and 2 <= next_pos[1] < self.PIC.shape[1] - 2:
            self.pos = next_pos




        self.action = action
        self.steps = self.steps + 1
        observation = self._observe()
        reward = self._get_reward()
        self.done = self._is_done()

        return observation, reward, self.done, {}

    def _render(self, mode='human', close=False):
        # human の場合はコンソールに出力。ansiの場合は StringIO を返す
        os.system('cls')
        print(self.pos)
        #print(self.obs())
        print(self.TARGET)
        print(self._observe())
        #print(self.x_train)
        #print(self.train_data)
        
        if not self.reset_rend:#一度目の処理なので描画初期化
            # Pygameを初期化
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("pic-gym-v1")              # タイトルバーに表示する文字
            self.font = pygame.font.Font(None, 15) 
            self.font_item = pygame.font.Font(None, 30)               # フォントの設定(55px)
            self.screen.fill((0,0,0))

            self.reset_rend = True

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()



        self.screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(np.array([self.obs(),self.obs(),self.obs()]).transpose(1, 2, 0)), (self.obs().shape[0] * 24 , self.obs().shape[1] * 24)), (10, 10))

        pygame.display.update()  # 画面を更新

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
        #print(np.max(encoded_obs),'aaaaaaaaaaaa')
        #print(encoded_obs)
        ##########################################################################################
        if self.TARGET[0] == self.pos[0] and self.TARGET[1] == self.pos[1]:
            return 100
        else:
            return -1
        '''
        if not self._is_done():#終了してないとき
            return -1
        else:#終了時
            if self.TARGET[0] == self.pos[0] and self.TARGET[1] == self.pos[1]:
                return 100
            else:
                return -1'''

    def obs(self):#こっちは2D
        return self.PIC[self.pos[0]-2:self.pos[0]+3,self.pos[1]-2:self.pos[1]+3]

    '''
    def _observe(self):#こっちは1D+エンコード結果
        return np.concatenate([(np.ravel(self.PIC[self.pos[0]-2:self.pos[0]+3,self.pos[1]-2:self.pos[1]+3])) / 255,self.pos / 50],0)'''


    def _observe(self):#こっちは1D+エンコード結果
        return np.concatenate([self.TARGET / self.PIC.shape[0] ,self.pos / self.PIC.shape[1]],0)

    
    def _is_done(self):
        # 今回は最大で self.MAX_STEPS までとした
        if self.steps > self.MAX_STEPS:
            return True
        else:
            if self.TARGET[0] == self.pos[0] and self.TARGET[1] == self.pos[1]:#最後のディレクトリでボタンを押す
                return True
            else:
                return False


