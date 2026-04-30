@echo off
title Detector de Postura

echo.
echo ==========================================
echo    DETECTOR DE POSTURA - INICIANDO
echo ==========================================
echo.

py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo ERRO: Python 3.11 nao encontrado!
    echo Instale em: https://www.python.org/downloads/release/python-3119/
    echo Marque "Add Python to PATH" durante a instalacao.
    pause
    exit /b 1
)

echo [OK] Python 3.11 encontrado!

if exist "venv\" (
    echo Removendo ambiente virtual antigo...
    rmdir /s /q venv
)

echo.
echo [1/3] Criando ambiente virtual...
py -3.11 -m venv venv
echo [OK] Ambiente virtual criado!

echo.
echo [2/3] Instalando dependencias (aguarde)...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install mediapipe==0.10.13 opencv-python numpy --quiet
echo [OK] Dependencias instaladas!

echo.
echo [3/3] Iniciando o detector...
echo.
echo Pressione Q na janela da camera para encerrar.
echo.

python posture_detector.py

echo.
echo Programa encerrado.
pause
