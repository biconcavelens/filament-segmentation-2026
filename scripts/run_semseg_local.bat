@echo off
cd /d "C:\projects\kaggle sun thiing"
set PY=C:\Users\trish\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe
"%PY%" -u train_semseg_local.py > semseg_local_train.log 2>&1 && "%PY%" -u train_semseg_local.py predict > semseg_local_predict.log 2>&1
