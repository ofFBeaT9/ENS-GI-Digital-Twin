@echo off
REM ENS-GI Digital Twin - Environment Setup Script
REM This creates a proper conda environment with Python 3.10

echo ========================================
echo ENS-GI Digital Twin - Environment Setup
echo ========================================
echo.

echo [1/4] Creating conda environment 'ens-gi' with Python 3.10...
call conda create -n ens-gi python=3.10 -y

echo.
echo [2/4] Activating environment...
call conda activate ens-gi

echo.
echo [3/4] Installing required packages...
call pip install -r requirements.txt

echo.
echo [4/4] Running verification tests...
call python verify_installation.py

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To use the environment in the future:
echo   conda activate ens-gi
echo.
echo To run verification:
echo   python verify_installation.py
echo.
echo To run tests:
echo   pytest tests/ -v
echo.
pause
