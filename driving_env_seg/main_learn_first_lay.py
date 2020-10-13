#一層目を学習するときにはこっちから起動
import subprocess
import numpy as np

#https://www.it-swarm.dev/ja/tensorflow/tensorflow%EF%BC%9Ainternalerror%EF%BC%9Ablas-sgemm%E3%81%AE%E8%B5%B7%E5%8B%95%E3%81%AB%E5%A4%B1%E6%95%97%E3%81%97%E3%81%BE%E3%81%97%E3%81%9F/824534956/
if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

#学習方式を伝達Trueは2層目学習 Falseは1層目学習
lear_method = np.array([False],dtype="bool")
lear_method.tofile('強化学習/行動細分化/driving_env/driving_env_seg/lear_method.npy')

print("==========Start loading the 1st layer==========")
#1層目のDQNを読み込みc
#import dqn_1st_lay
#subprocess.Popen(["python","強化学習/行動細分化/driving_env/driving_env_seg/dqn_1st_lay.py"])
import agent57
#subprocess.Popen(["python","強化学習/行動細分化/driving_env/driving_env_seg/agent57.py"])