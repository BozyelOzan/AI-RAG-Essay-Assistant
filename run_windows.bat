@echo off
setlocal

echo ==============================================
echo    🧬 AI Research Assistant - Setup & Run
echo ==============================================

if not exist "config\config.py" (
    echo [WARNING] 'config.py' not found!
    echo Copying 'config_example.py'...
    copy config\config_example.py config\config.py
    echo.
    echo [STOPPED] Please enter your API Key in 'config/config.py' and run the script again.
    pause
    exit /b
)

if not exist ".venv" (
    echo [INFO] Creating virtual environment (.venv)...
    python -m venv .venv
) else (
    echo [OK] Virtual environment exists.
)

call .venv\Scripts\activate

echo [INFO] Installing libraries...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] An error occurred during library installation.
    pause
    exit /b
)

echo.
echo [INFO] Creating vector database (Ingestion)...
python src\data_ingestion.py

echo.
echo [STARTING] Launching Streamlit interface...
echo ==============================================
streamlit run app.py

pause