import subprocess
import numpy as np


ENABLE_DDPG = True


#https://www.it-swarm.dev/ja/tensorflow/tensorflow%EF%BC%9Ainternalerror%EF%BC%9Ablas-sgemm%E3%81%AE%E8%B5%B7%E5%8B%95%E3%81%AB%E5%A4%B1%E6%95%97%E3%81%97%E3%81%BE%E3%81%97%E3%81%9F/824534956/
if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

#学習方式を伝達Trueは2層目学習 Falseは1層目学習
lear_method = np.array([True],dtype="bool")
lear_method.tofile('強化学習/行動細分化/driving_env/driving_env_seg/lear_method.npy')

#1層目のDQNが構築完了したことを伝えるためのファイルの初期化
dqntes_main = np.array([False],dtype="bool")
dqntes_main.tofile('強化学習/行動細分化/driving_env/driving_env_seg/dqntes_main.npy')

print("==========Start loading the 1st layer==========")
#1層目のDQNを読み込み (非同期で読み込む)
subprocess.Popen(["python","強化学習/行動細分化/driving_env/driving_env_seg/dqn_1st_lay.py"])

while True:
    dqntes_main = np.fromfile('強化学習/行動細分化/driving_env/driving_env_seg/dqntes_main.npy', dtype="bool")
    #ロード完了したらTrueになるのでそうなったら開放する
    #多分同時にファイル開かれるとサイズが0になっちゃうからそれを防止する
    try:
        if dqntes_main[0]:
            break
        
    except IndexError:
        pass
        

print("==========Start loading the 2nd layer==========")
#2層目のDQNを読み込み (非同期で読み込む)
if ENABLE_DDPG:
    subprocess.Popen(["python","強化学習/行動細分化/driving_env/driving_env_seg/ddpg_2nd_lay_learner.py"])
else:
    subprocess.Popen(["python","強化学習/行動細分化/driving_env/driving_env_seg/dqn_2nd_lay_learner.py"])