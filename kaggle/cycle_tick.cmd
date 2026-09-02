@echo off
REM Autonomous one-step chaining tick (D-023). Registered as a scheduled task.
cd /d "D:\Personal Projects\geom_aware_neural_operator"
echo ==== tick %DATE% %TIME% ==== >> logs\cycle.log
".venv\Scripts\python.exe" kaggle\cycle.py >> logs\cycle.log 2>&1
echo ---- tick end %DATE% %TIME% ---- >> logs\cycle.log
