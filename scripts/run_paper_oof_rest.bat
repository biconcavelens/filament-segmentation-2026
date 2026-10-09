@echo off
cd /d "C:\projects\kaggle sun thiing"
set PY=C:\Users\trish\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe
for %%R in (base4 plainS solo_A solo_B solo_C solo_L solo_S) do (
  "%PY%" -u paper_oof.py %%R > paper_oof\%%R.log 2> paper_oof\%%R.err
)
