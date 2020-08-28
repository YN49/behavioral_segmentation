import subprocess
 
print("cmd.py start!!")
 
cmd = "Python3 dqn_tester.py"
#subprocess.Popen(cmd.split())
subprocess.Popen(["python","強化学習/行動細分化/driving_env/driving_env_seg/dqn_tester.py"])
 
print("cmd.py end!!")