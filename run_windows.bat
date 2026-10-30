@echo off
setlocal

echo ==============================================
echo    AI Research Assistant - Setup and Run
echo ==============================================

if not exist ".env" (
    echo [WARNING] '.env' not found!
    echo Copying '.env.example'...
    copy .env.example .env
    echo.
    echo [STOPPED] Please open '.env', set LLM_PROVIDER and the matching API key, and run the script again.
    pause
    exit /b
)

if not exist ".venv" (
    echo [INFO] Creating virtual environment .venv...
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
