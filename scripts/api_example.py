from __future__ import annotations

import json
from pathlib import Path
from urllib import request

import pandas as pd

history = pd.read_csv("data/sample_history.csv").tail(240)
future = pd.read_csv("data/sample_future.csv")

payload = {
    "records": history.to_dict(orient="records"),
    "future_records": future.to_dict(orient="records"),
    "prediction_length": len(future),
    "backend": "chronos2",
}

req = request.Request(
    "http://localhost:8000/forecast",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with request.urlopen(req, timeout=600) as res:
    data = json.loads(res.read().decode("utf-8"))

Path("outputs").mkdir(exist_ok=True)
Path("outputs/api_response.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print("outputs/api_response.json")
