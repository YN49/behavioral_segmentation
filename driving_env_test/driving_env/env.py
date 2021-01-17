from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras import backend as K

from tensorflow.keras.callbacks import EarlyStopping

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
import pygame.gfxdraw
import sys

#---------------------------------------------------------------------
import ctypes
def isPressed(key):
    return(bool(ctypes.windll.user32.GetAsyncKeyState(key)&0x8000))
#---------------------------------------------------------------------


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class ENV(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}
    
    f_model = './model'

    PIC = np.array(Image.open('強化学習/行動細分化/driving_env/driving_env_seg/MAP_PIC.png').convert('L'))

    WIDTH = 950
    HEIGHT = 450
    #ウインドウの場所とサイズ x y size(何倍か)
    """
    WINDOW_DATA = [[0,0,10],
                  [200,0,10],
                  [0,200,2],
                  [500,300,40]]"""
    WINDOW_DATA = [[0,0,5],
                  [200,0,5],
                  [0,200,2],
                  [500,300,40]]

    #車の加速度 1タイムステップどのくらいの速度加速するかOR減速するか
    ACCEL = 0.3*0.1
    #何度ハンドルを曲げられるか
    ANG_HNG = 10*0.45
    #スピードの上限
    #一定以上の速度で走れば報酬を与える
    SPEED_REW = 0.3
    SPEED_LIM = 0.4
    VIEW_SIZE = (20,20)
    #初期位置
    INI_POS = [20, 15]
    #初期のベクトル情報
    INI_VEC = [0,90]
    #居場所を保存するか
    ENABLE_SAVE_POS = False
    #移動ベクトルを保存するか
    ENABLE_SAVE_VEC = False

    #道路の外側の色の濃さ　この色の濃さのところを通るとマイナスの報酬が
    OUTSIDE = 0
    #ゴールの色の濃さ ここを通ると報酬が発生
    GOAL = 200
    #ゴールとか障害物の値の誤差範囲
    ERROR_OF_PIX_VAL = 5


    weights_filename = '強化学習/行動細分化/driving_env/driving_env_seg/vae.hdf5'

    #1層目と2層目の動作間隔 なお2層目のときは終了ステップは MAX_STEPS * INTERVAL
    INTERVAL = 4
    MAX_STEPS = 150

    #RANGE = 0.18#報酬やるときにどのくらいの距離だったら同じものだという認識に入るか
    RANGE = 0.21#本来はこれ

    #vaeのエポック数10
    epochs = 1
    #epochs = 15
    #何stepに一回学習するか200
    TRAIN_FREQ = 10000

    original_dim = VIEW_SIZE[0] * VIEW_SIZE[1]

    #VAEの学習を実行するか
    ENABLE_VAR = True
    #何ステップ分の教師データを保存するか
    X_TRAIN_RANGE = 80000

    # VAE parameters
    input_shape = (original_dim, )
    #中間層
    intermediate_dim = 128
    batch_size = 128
    #潜在変数
    latent_dim = 2

    ##############################################モデル構成##############################################

    # VAE model = encoder + decoder
    # エンコーダーモデルを構築
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')



    def __init__(self):
        super().__init__()
        # action_space, observation_space, reward_range を設定する
        self.action_space = gym.spaces.Discrete(5)  # 東西南北設置破壊
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(400,)
        )
        self.reward_range = [-500., 10000.]

        self._reset()

        self.viewer = None

    def _reset(self):
        # 諸々の変数を初期化する

        #初期位置

        #移動速度のベクトル情報格納 ベクトルの長さ,x軸との交差角θ
        self.move_vec = [0,0]
        self.done = False
        self.steps = 0

        self.pre_reward = np.zeros(2,dtype="int64")

        self.reset_rend = False

        self.out_train = np.zeros((2,self.original_dim))

        self.achievement_flg = False
        


        #保存されている位置を利用
        if self.ENABLE_SAVE_POS:
            self.pos = self.train_data[1:3]
        else:
            self.pos = self.INI_POS
        #保存されてるベクトル利用
        if self.ENABLE_SAVE_VEC:
            self.move_vec = [self.train_data[3],self.train_data[4]]
        else:
            self.move_vec[0] = self.INI_VEC[0]
            self.move_vec[1] = self.INI_VEC[1]
    

        try:
            self.vae.load_weights(os.path.join(self.weights_filename))
        except:
            pass

        self.int_pos = np.array(self.pos,dtype="int64")


        """
        if self.lear_method[0]:
            print("===========================ENV1:2層目の学習であることを認知しました===========================")
        else:
            print("===========================ENV1:1層目の学習であることを認知しました===========================")"""



        return self._observe()
        

    def _step(self, action):


        # 1ステップ進める処理を記述。戻り値は observation, reward, done(ゲーム終了したか), info(追加の情報の辞書)
        if action == 0:
            #加速
            self.move_vec[0] = self.move_vec[0] + self.ACCEL
            #リミッター超えたとき
            if self.SPEED_LIM < self.move_vec[0]:
                self.move_vec[0] = self.SPEED_LIM
        elif action == 1:
            #減速   ブレーキはアクセルより効きが良い
            self.move_vec[0] = self.move_vec[0] - self.ACCEL * 3
            if self.move_vec[0] < 0:
                self.move_vec[0] = 0
        elif action == 2:
            #左にハンドル曲げる
            self.move_vec[1] = self.move_vec[1] + self.ANG_HNG
        elif action == 3:
            #右にハンドル曲げる
            self.move_vec[1] = self.move_vec[1] - self.ANG_HNG
        else:#action4は何もしない
            pass
        
        
        a_1 = self.move_vec[0] * math.cos(math.radians(self.move_vec[1]))
        a_2 = self.move_vec[0] * math.cos(math.radians(90 - self.move_vec[1]))
        #移動方向のベクトル
        self.mov_dir_vec = np.array([a_1,a_2])

        next_pos = self.pos + self.mov_dir_vec
        self.pos = next_pos
        #視界のサイズにあわせて数字の大きさ変えないとね
        #端っこに衝突したら減点
        self.collusion_flg = False
        if self.PIC.shape[0] - math.ceil(self.VIEW_SIZE[0]/2) < next_pos[0]:
            self.pos[0] = self.PIC.shape[0] - math.ceil(self.VIEW_SIZE[0]/2)
            self.collusion_flg = True
        if next_pos[0] <= math.ceil(self.VIEW_SIZE[0]/2):
            self.pos[0] = math.ceil(self.VIEW_SIZE[0]/2)
            self.collusion_flg = True
        if self.PIC.shape[1] - math.ceil(self.VIEW_SIZE[1]/2) < next_pos[1]:
            self.pos[1] = self.PIC.shape[1] - math.ceil(self.VIEW_SIZE[1]/2)
            self.collusion_flg = True
        if  next_pos[1] <= math.ceil(self.VIEW_SIZE[1]/2):
            self.pos[1] = math.ceil(self.VIEW_SIZE[1]/2)
            self.collusion_flg = True

        self.int_pos = np.array(self.pos,dtype="int64")

        

        self.action = action
        self.steps = self.steps + 1
        observation = self._observe()
        reward = self._get_reward()
        self.done = self._is_done()

        return observation, reward, self.done, {}

    def _render(self, mode='human', close=False):
        # human の場合はコンソールに出力。ansiの場合は StringIO を返す
        #os.system('cls')
        #print(self.pos)
        #print(self._get_reward())
        #print(self.obs())
        #print(math.sqrt(np.sum((self.encoded_obs - self.TARGET) ** 2)),'差')
        #print('範囲',self.range_calcu,'x',math.sqrt(np.sum(self.TARGET**2)))
        #print(self.encoded_obs)
        #print(self.TARGET)
        #print(self.x_train)
        #print(self.train_data)
        
        if not self.reset_rend:#一度目の処理なので描画初期化
            # Pygameを初期化
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("pic-gym-v0")              # タイトルバーに表示する文字
            self.font = pygame.font.Font(None, 15) 
            self.font_item = pygame.font.Font(None, 30)               # フォントの設定(55px)
            self.screen.fill((0,0,0))


        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()


        #ID0
        self.screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(np.array([self.obs(),self.obs(),self.obs()]).transpose(1, 2, 0)), (self.obs().shape[0] * self.WINDOW_DATA[0][2] , self.obs().shape[1] * self.WINDOW_DATA[0][2])), (self.covX(0,0), self.covY(0,0)))
        #真ん中に車を描画
        pygame.draw.circle(self.screen, (255,255,0), (self.covX(0,self.VIEW_SIZE[0]/2),self.covY(0,self.VIEW_SIZE[1]/2)), 5)
        #移動ベクトル描画
        pygame.draw.line(self.screen, (255,0,0), (self.covX(0,self.VIEW_SIZE[0]/2),self.covY(0,self.VIEW_SIZE[1]/2)), (self.covX(0,self.VIEW_SIZE[0]/2+self.mov_dir_vec[0]*5),self.covY(0,self.VIEW_SIZE[1]/2+self.mov_dir_vec[1]*5)))
        
        #ID2
        #mapの下地
        self.screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(np.array([self.PIC,self.PIC,self.PIC]).transpose(1, 2, 0)), (int(self.PIC.shape[0] * self.WINDOW_DATA[2][2]) , int(self.PIC.shape[1] * self.WINDOW_DATA[2][2]))), (self.covX(2,0), self.covY(2,0)))
        #車を描画
        pygame.draw.circle(self.screen, (255,255,0), (self.covX(2,self.pos[0]),self.covY(2,self.pos[1])), 3)
        #移動ベクトル描画
        pygame.draw.line(self.screen, (255,0,0), (self.covX(2,self.pos[0]),self.covY(2,self.pos[1])), (self.covX(2,self.pos[0]+self.mov_dir_vec[0]*20),self.covY(2,self.pos[1]+self.mov_dir_vec[1]*20)))

        #ID3
        #軸
        pygame.draw.line(self.screen, (255,255,255), (self.covX(3,0),self.covY(3,3)), (self.covX(3,0),self.covY(3,-3)))
        pygame.draw.line(self.screen, (255,255,255), (self.covX(3,3),self.covY(3,0)), (self.covX(3,-3),self.covY(3,0)))



        pygame.display.update()  # 画面を更新

        if not self.reset_rend:
            self.reset_rend = True

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
        #円の中に入ったら終了

        #ゴールに着いたら高い報酬を与える
        #if self.GOAL - self.ERROR_OF_PIX_VAL < self.PIC[self.int_pos[0]][self.int_pos[1]] < self.GOAL + self.ERROR_OF_PIX_VAL and self.lear_method[0]:
        if self.GOAL - self.ERROR_OF_PIX_VAL < self.PIC[self.int_pos[0]][self.int_pos[1]] < self.GOAL + self.ERROR_OF_PIX_VAL:
            #pass
            return 800
        #壁にぶつかったら減点
        elif self.collusion_flg:
            return -2
        #外側走ったらダメだから減点
        elif self.OUTSIDE - self.ERROR_OF_PIX_VAL < self.PIC[self.int_pos[0]][self.int_pos[1]] < self.OUTSIDE + self.ERROR_OF_PIX_VAL:
            return -5#####-500###############################################################################################################################################
        #一定の速度で走れば報酬を増やす
        elif self.SPEED_REW < self.move_vec[0]:
            return 0
        #ステップ毎減点
        else:
            return -1.5

    def obs(self):#こっちは2D 画面に表示するやつ
        return self.PIC[self.int_pos[0]-math.ceil(self.VIEW_SIZE[0]/2):self.int_pos[0]+math.floor(self.VIEW_SIZE[0]/2), 
        self.int_pos[1]-math.ceil(self.VIEW_SIZE[1]/2):self.int_pos[1]+math.floor(self.VIEW_SIZE[1]/2)]
    

    def obs_encoder(self):#エンコーダへの入力
        return np.ravel(self.obs()).astype('float32') / 255.

    def _observe(self):#エンコード結果+ターゲット 環境の出力
        return self.obs().flatten()

    
    def _is_done(self):
        # 今回は最大で self.MAX_STEPS までとした ゴールについたら終了 最後のディレクトリでボタンを押す


        #maxstep超えたら終了
        if self.steps > self.MAX_STEPS:
            return True
        #ゴールに到着で終了 これは共有条件
        elif self.GOAL - self.ERROR_OF_PIX_VAL < self.PIC[self.int_pos[0]][self.int_pos[1]] < self.GOAL + self.ERROR_OF_PIX_VAL:
            #ゴールに着いたらベクトルと場所を初期化
            self.move_vec = self.INI_VEC
            self.pos = np.array(self.INI_POS)

            return True
        ####################################################################################################################################################
        
        else:
            return False

        """#######################################################################################################################################################
        elif self.OUTSIDE - self.ERROR_OF_PIX_VAL < self.PIC[self.int_pos[0]][self.int_pos[1]] < self.OUTSIDE + self.ERROR_OF_PIX_VAL:
            return True
        #円の中に入ったら終了
        elif (math.sqrt(np.sum((self.encoded_obs - self.TARGET) ** 2)) < self.range_calcu and not self.lear_method[0]):
            return True

        #障害物にぶつかったら終了
        elif self.OUTSIDE - self.ERROR_OF_PIX_VAL < self.PIC[self.int_pos[0]][self.int_pos[1]] < self.OUTSIDE + self.ERROR_OF_PIX_VAL:
            return True
        #壁にぶつかったら終了
        elif self.collusion_flg:
            return True
        """

    #描画のためにx座標y座標を変換
    def covX(self,winID,x):
        return int(x * self.WINDOW_DATA[winID][2] + self.WINDOW_DATA[winID][0])
    def covY(self,winID,y):
        return int(y * self.WINDOW_DATA[winID][2] + self.WINDOW_DATA[winID][1])
