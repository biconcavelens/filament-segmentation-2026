@echo off
cd /d "C:\projects\kaggle sun thiing"
set PY=C:\Users\trish\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe
for %%R in (raw_base4 lodo_Av lodo_B1280v lodo_Cv lodo_L1280v lead12_nofuse voteS lead12 fullS_v0.2 fullS_v0.4 fullS_v0.1 fullS_v0.5 fullS_v0.6) do (
  if not exist paper_oof\%%R.npz "%PY%" -u paper_oof.py %%R > paper_oof\%%R.log 2> paper_oof\%%R.err
)
