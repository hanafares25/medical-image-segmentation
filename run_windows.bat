@echo off
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_sample_data.py
python train.py --epochs 8
streamlit run streamlit_app.py
