\# medical-ml-platform



A minimal Python API for predicting a patient's relative medical risk from basic health indicators.



\## Quick start



```bash

python -m venv .venv

source .venv/bin/activate  # Windows: .venv\\\\Scripts\\\\activate

pip install -r requirements.txt

uvicorn app:app --reload

