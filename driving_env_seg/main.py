import subprocess
import numpy as np

#https://www.it-swarm.dev/ja/tensorflow/tensorflow%EF%BC%9Ainternalerror%EF%BC%9Ablas-sgemm%E3%81%AE%E8%B5%B7%E5%8B%95%E3%81%AB%E5%A4%B1%E6%95%97%E3%81%97%E3%81%BE%E3%81%97%E3%81%9F/824534956/
if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

#1層目のDQNが構築完了したことを伝えるためのファイルの初期化
dqntes_main = np.array([False,False],dtype="bool")
dqntes_main.tofile('強化学習/行動細分化/driving_env/driving_env_seg/dqntes_main.npy')

print("==========Start loading the 1st layer==========")
#1層目のDQNを読み込み (非同期で読み込む)
subprocess.Popen(["python","強化学習/行動細分化/driving_env/driving_env_seg/dqn_tester.py"])

while True:
    dqntes_main = np.fromfile('強化学習/行動細分化/driving_env/driving_env_seg/dqntes_main.npy', dtype="bool")
    #ロード完了したらTrueになるのでそうなったら開放する
    if dqntes_main[0]:
        break

print("==========Start loading the 2nd layer==========")
#2層目のDQNを読み込み (非同期で読み込む)
subprocess.Popen(["python","強化学習/行動細分化/driving_env/driving_env_seg/dqn_2nd_lay_learner.py"])