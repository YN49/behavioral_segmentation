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
    ACCEL = 0.7*0.1
    #何度ハンドルを曲げられるか
    ANG_HNG = 10*0.6
    #スピードの上限
    #一定以上の速度で走れば報酬を与える
    SPEED_REW = 0.4
    SPEED_LIM = 1.6
    VIEW_SIZE = (30,30)
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
    GOAL = 64
    #ゴールとか障害物の値の誤差範囲
    ERROR_OF_PIX_VAL = 5


    weights_filename = '強化学習/行動細分化/driving_env/driving_env_seg/vae.hdf5'

    #1層目と2層目の動作間隔 なお2層目のときは終了ステップは MAX_STEPS * INTERVAL
    INTERVAL = 3
    MAX_STEPS = 50

    #RANGE = 0.18#報酬やるときにどのくらいの距離だったら同じものだという認識に入るか
    RANGE = 0.21#本来はこれ

    #vaeのエポック数10
    epochs = 100
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
            shape=(4,)
        )
        self.reward_range = [-300., 2000.]

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

        self.encoded_obs = np.zeros(self.latent_dim, dtype="float64")

        self.achievement_flg = False
        
        try:
            loaded_array = np.load('強化学習/行動細分化/driving_env/driving_env_seg/data.npz')
            self.x_train = loaded_array['arr_0']
            self.train_data = loaded_array['arr_1']
            
        except FileNotFoundError:
            self.x_train = np.zeros((2,self.original_dim))
            self.train_data = np.array([0,self.INI_POS[0],self.INI_POS[1],self.INI_VEC[0],self.INI_VEC[1]])


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


        #学習モード取得 Trueは2層目学習 Falseは1層目学習
        self.lear_method = np.fromfile('強化学習/行動細分化/driving_env/driving_env_seg/lear_method.npy', dtype="bool")

        #２層目のときはmaxを増やしてもいいので増やす
        if self.lear_method[0]:
            #1層目と2層目の動作間隔 なお2層目のときは終了ステップは MAX_STEPS * INTERVAL
            self.MAX_STEPS = self.MAX_STEPS * (self.INTERVAL - 1) + (self.INTERVAL - 1) * 2

        """
        if self.lear_method[0]:
            print("===========================ENV1:2層目の学習であることを認知しました===========================")
        else:
            print("===========================ENV1:1層目の学習であることを認知しました===========================")"""

        self.sync1_2 = np.array([False],dtype="bool")

        #終了伝達ファイルを初期化
        done_signal = np.array([False],dtype="bool")
        done_signal.tofile('強化学習/行動細分化/driving_env/driving_env_seg/done_signal.npy')

        #報酬伝達ファイルを初期化
        self.rew_signal = np.array([0],dtype="int64")
        self.rew_signal.tofile('強化学習/行動細分化/driving_env/driving_env_seg/rew_signal.npy')


        #一層目の学習はランダムでターゲット設定
        if not self.lear_method[0]:
            self.update_traget()#ターゲットとりあえずランダム設定
        else:
            #まだターゲットが決まっていないなら0にしておく
            self.TARGET = np.zeros(self.latent_dim)
            self.TARGET_PIC = self.decoder.predict(np.squeeze(self.TARGET)[np.newaxis,:]).reshape(self.VIEW_SIZE)*255
            #計算済みの範囲を格納
            self.range_calcu = 0.5


        return self._observe()
        

    def _step(self, action):

        #print(self.steps)
        self.pre_TARGET = self.TARGET
        self.pre_range_calcu = self.range_calcu

        #2層目学習時にはENV2のターゲット情報を獲得する
        if self.lear_method[0]:
            #1,5,9,13,17(INTERVAL=5の場合)の周期で受け取る
            if self.steps % (self.INTERVAL - 1) == 1:

                #決定されたターゲットの読み込み
                try:
                    self.TARGET = np.fromfile('強化学習/行動細分化/driving_env/driving_env_seg/target.npy')
                except FileNotFoundError:
                    self.TARGET.tofile('強化学習/行動細分化/driving_env/driving_env_seg/target.npy')

                self.TARGET_PIC = self.decoder.predict(np.squeeze(self.TARGET)[np.newaxis,:]).reshape(self.VIEW_SIZE)*255
                #計算済みの範囲を格納
                self.range_calcu = 0.5

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

        #エンコードするぞ
        self.encoded_obs, _, _ = self.encoder.predict(np.squeeze(self.obs_encoder())[np.newaxis,:])
        self.encoded_obs = self.encoded_obs.reshape(self.latent_dim, )

        self.action = action
        self.steps = self.steps + 1
        observation = self._observe()
        reward = self._get_reward()
        self.done = self._is_done()



        #どっちのモードでも描画用に達成したかを保存
        if math.sqrt(np.sum((self.encoded_obs - self.TARGET) ** 2)) < self.range_calcu:
            self.achievement_flg = True
        #一回達成したらターゲット更新 一層目学習のときのみ有効
        if (math.sqrt(np.sum((self.encoded_obs - self.TARGET) ** 2)) < self.range_calcu and not self.lear_method[0]):
            self.update_traget()#ターゲットとりあえずランダム設定


        #現ステップのobservationを教師データに格納
        self.out_train = np.insert(self.out_train, self.out_train.shape[0], self.obs_encoder(), axis=0)
        if self.done:
            
            #終了時に先端の２つの余計な配列を取り除く
            self.out_train = np.delete(self.out_train, 0, 0)
            self.out_train = np.delete(self.out_train, 0, 0)


            self.x_train = np.insert(self.out_train, self.out_train.shape[0], self.x_train, axis=0)
            if self.train_data[0] == 0:#一番最初の学習の場合
                #先端の２つの余計な配列を取り除く
                self.x_train = np.delete(self.x_train, 0, 0)
                self.x_train = np.delete(self.x_train, 0, 0)

            self.train_data[0] = self.train_data[0] + self.steps#ステップ数カウント

            self.x_train = self.x_train[max(0,int(self.x_train.shape[0] - self.X_TRAIN_RANGE)):,:]

            if self.train_data[0] >= self.TRAIN_FREQ:#TRAIN_FREQステップに一回学習
                self.train_data[0] = 0



                #VAE実行
                if self.ENABLE_VAR:
                    self.VAE()

                #ひとがくしゅう終わったからステップ数カウンターをを初期化する
                self.train_data[0] = 0

                #os.remove('強化学習/行動細分化/driving_env/driving_env_seg/data.npz')

            #居場所を格納
            if self.ENABLE_SAVE_POS:
                self.train_data[1:3] = self.pos
            #ベクトル保存
            if self.ENABLE_SAVE_VEC:
                self.train_data[3] = self.move_vec[0]
                self.train_data[4] = self.move_vec[1]
            #保存
            np.savez('強化学習/行動細分化/driving_env/driving_env_seg/data.npz', self.x_train, self.train_data, self.TARGET)

        ############ 同期 ############
        #2層目学習時にはENV2の処理を待つ
        if self.lear_method[0]:
            self.pre_reward[0] = self.pre_reward[0] + reward
            #1,5,9,13,17(INTERVAL=5の場合)の周期でストップさせる
            if self.steps % (self.INTERVAL - 1) == 1:

                #報酬を伝達
                self.pre_reward.tofile('強化学習/行動細分化/driving_env/driving_env_seg/rew_signal.npy')
                self.pre_reward[0] = 0

                #視界(VAEのエンコード結果)の情報をenv2に入力する
                np.array(self.encoded_obs,dtype="float64").tofile('強化学習/行動細分化/driving_env/driving_env_seg/encoded_obs.npy')

                #準備ができたので準備完了信号を出す
                self.sync1_2[0] = False
                self.sync1_2.tofile('強化学習/行動細分化/driving_env/driving_env_seg/sync1_2.npy')

                while True:
                    self.sync1_2 = np.fromfile('強化学習/行動細分化/driving_env/driving_env_seg/sync1_2.npy', dtype="bool")
                    #Trueにenv2がしてきたのでそれを感知して処理開始
                    #多分同時にファイル開かれるとサイズが0になっちゃうからそれを防止する
                    try:
                        if self.sync1_2[0]:
                            break
                    except IndexError:
                        pass

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

        self.TARGET2d = self.TARGET.reshape(int(self.latent_dim/2),2)*255
        self.encoded2d = self.encoded_obs.reshape(int(self.latent_dim/2),2)*255

        #ID0
        self.screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(np.array([self.obs(),self.obs(),self.obs()]).transpose(1, 2, 0)), (self.obs().shape[0] * self.WINDOW_DATA[0][2] , self.obs().shape[1] * self.WINDOW_DATA[0][2])), (self.covX(0,0), self.covY(0,0)))
        #真ん中に車を描画
        pygame.draw.circle(self.screen, (255,255,0), (self.covX(0,self.VIEW_SIZE[0]/2),self.covY(0,self.VIEW_SIZE[1]/2)), 5)
        #移動ベクトル描画
        pygame.draw.line(self.screen, (255,0,0), (self.covX(0,self.VIEW_SIZE[0]/2),self.covY(0,self.VIEW_SIZE[1]/2)), (self.covX(0,self.VIEW_SIZE[0]/2+self.mov_dir_vec[0]*5),self.covY(0,self.VIEW_SIZE[1]/2+self.mov_dir_vec[1]*5)))
        #ID1
        self.screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(np.array([self.TARGET_PIC,self.TARGET_PIC,self.TARGET_PIC]).transpose(1, 2, 0)), (self.TARGET_PIC.shape[0] * self.WINDOW_DATA[1][2] , self.TARGET_PIC.shape[1] * self.WINDOW_DATA[1][2])), (self.covX(1,0), self.covY(1,0)))
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

        #円の中心を点に
        pygame.gfxdraw.pixel(self.screen, self.covX(3,self.pre_TARGET[0]), self.covY(3,self.pre_TARGET[1]), (255,255,255))

        if not self.achievement_flg:
            pygame.draw.circle(self.screen, (255,255,255), (self.covX(3,self.pre_TARGET[0]), self.covY(3,self.pre_TARGET[1])), int(self.WINDOW_DATA[3][2]*self.pre_range_calcu), 1)
        else:#中に入ると塩が赤く
            pygame.draw.circle(self.screen, (255,0,0), (self.covX(3,self.pre_TARGET[0]), self.covY(3,self.pre_TARGET[1])), int(self.WINDOW_DATA[3][2]*self.pre_range_calcu), 2)
            self.achievement_flg = False

        if self.reset_rend:#一度目の処理じゃない場合
            pygame.draw.line(self.screen, (255,255,255), (self.covX(3,self.encoded_obs[0]), self.covY(3,self.encoded_obs[1])), (self.befo_pixpos[0],self.befo_pixpos[1]))

        self.befo_pixpos = [self.covX(3,self.encoded_obs[0]), self.covY(3,self.encoded_obs[1])]

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

        if math.sqrt(np.sum((self.encoded_obs - self.TARGET) ** 2)) < self.range_calcu and not self.lear_method[0]:
            return 300
        #ゴールに着いたら高い報酬を与える
        elif self.GOAL - self.ERROR_OF_PIX_VAL < self.PIC[self.int_pos[0]][self.int_pos[1]] < self.GOAL + self.ERROR_OF_PIX_VAL and self.lear_method[0]:
            return 2000
        #壁にぶつかったら減点
        elif self.collusion_flg:
            return -300
        #外側走ったらダメだから減点
        elif self.OUTSIDE - self.ERROR_OF_PIX_VAL < self.PIC[self.int_pos[0]][self.int_pos[1]] < self.OUTSIDE + self.ERROR_OF_PIX_VAL:
            return -300####################################################################################################################################################
        #一定の速度で走れば報酬を増やす
        elif self.SPEED_REW < self.move_vec[0]:
            return -1
        #ステップ毎減点
        else:
            return -8

    def obs(self):#こっちは2D 画面に表示するやつ
        return self.PIC[self.int_pos[0]-math.ceil(self.VIEW_SIZE[0]/2):self.int_pos[0]+math.floor(self.VIEW_SIZE[0]/2), 
        self.int_pos[1]-math.ceil(self.VIEW_SIZE[1]/2):self.int_pos[1]+math.floor(self.VIEW_SIZE[1]/2)]
    

    def obs_encoder(self):#エンコーダへの入力
        return np.ravel(self.obs()).astype('float32') / 255.

    def _observe(self):#エンコード結果+ターゲット 環境の出力
        return np.concatenate([self.encoded_obs,self.TARGET],0)

    
    def _is_done(self):
        # 今回は最大で self.MAX_STEPS までとした ゴールについたら終了 最後のディレクトリでボタンを押す

        #相手が終了したか取得する ENV1からの通信 ENV2からの通信
        done_signal = np.fromfile('強化学習/行動細分化/driving_env/driving_env_seg/done_signal.npy', dtype="bool")

        #maxstep超えたら終了
        if self.steps > self.MAX_STEPS and not self.lear_method[0]:
            return True
        #ゴールに到着で終了 これは共有条件
        elif self.GOAL - self.ERROR_OF_PIX_VAL < self.PIC[self.int_pos[0]][self.int_pos[1]] < self.GOAL + self.ERROR_OF_PIX_VAL:
            #ゴールに着いたらベクトルと場所を初期化
            self.move_vec = self.INI_VEC
            self.pos = np.array(self.INI_POS)

            done_signal[0] = True
            done_signal.tofile('強化学習/行動細分化/driving_env/driving_env_seg/done_signal.npy')

            return True
        #終了伝達がきたら終了
        elif done_signal[0]:
            return True
        ####################################################################################################################################################
        #障害物にぶつかったら終了
        elif self.OUTSIDE - self.ERROR_OF_PIX_VAL < self.PIC[self.int_pos[0]][self.int_pos[1]] < self.OUTSIDE + self.ERROR_OF_PIX_VAL:
            return True
        #壁にぶつかったら終了
        elif self.collusion_flg:
            return True
        
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

    #VAE
    def VAE(self):
        ##############################################学習開始##############################################
        #モデル読み込み
        #json_string = open(os.path.join(self.model_filename)).read()
        #autoencoder = model_from_json(json_string)

        
        
        parser = argparse.ArgumentParser()
        help_ = "Load h5 model trained weights"
        parser.add_argument("-w", "--weights", help=help_)
        help_ = "Use mse loss instead of binary cross entropy (default)"
        parser.add_argument("-m",
                            "--mse",
                            help=help_, action='store_true')
        args = parser.parse_args()
        models = (self.encoder, self.decoder)
        data = (self.x_train, 1)

        # VAE loss = mse_loss or xent_loss + kl_loss
        if args.mse:
            reconstruction_loss = mse(self.inputs, self.outputs)
        else:
            reconstruction_loss = binary_crossentropy(self.inputs,
                                                    self.outputs)
        #load vae
        '''
        try:
            self.vae.load_weights(os.path.join(self.weights_filename))
        except:
            pass'''

        reconstruction_loss *= self.original_dim
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam', loss=None)

        #自動終了
        early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
        # train the autoencoder
        history = self.vae.fit(self.x_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(self.x_train, None),
                callbacks=[early_stopping])

        #self.plot_results()

        self.vae.save_weights(os.path.join(self.weights_filename))

        '''
        loss = np.array(history.history['loss'])
        val_loss = np.array(history.history['val_loss'])
        #学習履歴データの前のを取り出して読んで保存
        try:
            vae_hist = np.load('強化学習/行動細分化/driving_env/driving_env_seg/vae_history.npz')
            loss = np.append(loss, vae_hist['arr_0'])
            val_loss = np.append(val_loss, vae_hist['arr_1'])
        except FileNotFoundError:
            pass

        np.savez('強化学習/行動細分化/driving_env/driving_env_seg/vae_history.npz', loss, val_loss)

        #学習のhistory結果をグラフ化
        self.compare_TV(loss,val_loss)'''



    def plot_results(self,
                    batch_size=128,
                    model_name="vae_mnist"):
        """Plots labels and MNIST digits as a function of the 2D latent vector
        # Arguments
            models (tuple): encoder and decoder models
            data (tuple): test data and label
            batch_size (int): prediction batch size
            model_name (string): which model is using this function
        """

        y_test = 1
        os.makedirs(model_name, exist_ok=True)

        # display a 2D plot of the digit classes in the latent space
        z_mean, _, _ = self.encoder.predict(self.x_train,
                                    batch_size=batch_size)
        plt.figure(figsize=(12, 10))
        plt.scatter(z_mean[:, 0], z_mean[:, 1])
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.show()

        # display a 30x30 2D manifold of digits
        n = 30
        digit_size = 5
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = self.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                    j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        start_range = digit_size // 2
        end_range = (n - 1) * digit_size + start_range + 1
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap='Greys_r')
        plt.show()

    



    def update_traget(self):#ターゲットアップデート
        self.TARGET = np.random.randn(self.latent_dim)#ターゲットとりあえずランダム設定
        self.TARGET_PIC = self.decoder.predict(np.squeeze(self.TARGET)[np.newaxis,:]).reshape(self.VIEW_SIZE)*255
        #計算済みの範囲を格納
        self.range_calcu = (self.RANGE*math.sqrt(self.latent_dim*(1/((1/math.sqrt(2*math.pi))*math.e**((np.sum(self.TARGET**2))/-2)))**2))/math.sqrt(self.latent_dim*2*math.pi)


    #描画のためにx座標y座標を変換
    def covX(self,winID,x):
        return int(x * self.WINDOW_DATA[winID][2] + self.WINDOW_DATA[winID][0])
    def covY(self,winID,y):
        return int(y * self.WINDOW_DATA[winID][2] + self.WINDOW_DATA[winID][1])
