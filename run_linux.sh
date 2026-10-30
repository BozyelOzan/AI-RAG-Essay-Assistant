#!/bin/bash

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN}   🧬 AI Research Assistant - Setup & Run  ${NC}"
echo -e "${GREEN}==============================================${NC}"

if [ ! -f ".env" ]; then
    echo -e "${YELLOW}[WARNING] '.env' file not found!${NC}"
    echo -e "${YELLOW}Copying '.env.example'...${NC}"
    cp .env.example .env
    echo -e "${RED}[STOPPED] Please open '.env', set LLM_PROVIDER and the matching API key (GROQ_API_KEY or OPENAI_API_KEY), and run the script again.${NC}"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}[INFO] Creating virtual environment (.venv)...${NC}"
    python3 -m venv .venv
else
    echo -e "${GREEN}[OK] Virtual environment exists.${NC}"
fi

source .venv/bin/activate

echo -e "${YELLOW}[INFO] Checking libraries/dependencies...${NC}"
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] An error occurred while installing libraries!${NC}"
    exit 1
fi

if [ -z "$(ls -A data/raw/*.pdf 2>/dev/null)" ]; then
    echo -e "${RED}[WARNING] No PDFs found in the 'data/raw' directory!${NC}"
    echo -e "${YELLOW}The system will run, but may not have content to query. Please add PDFs.${NC}"
else
    echo -e "${GREEN}[OK] PDF files detected.${NC}"
fi

echo -e "${YELLOW}[INFO] Creating/Updating database...${NC}"
python3 src/data_ingestion.py

echo -e "${GREEN}[STARTING] Launching Streamlit interface...${NC}"
echo -e "${GREEN}==============================================${NC}"
streamlit run app.py
