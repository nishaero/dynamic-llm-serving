import os
from huggingface_hub import login
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, HTMLResponse, Response
import uvicorn
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.registry import fetch_model_info, switch_model
from prometheus_client import Counter, Histogram, generate_latest
from jose import jwt, JWTError
from fastapi.security import OAuth2PasswordBearer

app = FastAPI()

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

model = None
tokenizer = None
MODEL_INFO = None

SECRET_KEY = os.getenv("JWT_SECRET", "changeme")
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None

def load_model_from_info(model_info):
    global model, tokenizer, MODEL_INFO
    model_id = model_info.get("model_id")
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=HF_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=HF_TOKEN,
        device_map="cuda"
    )
    print(f"Model loaded on device: {model.device if hasattr(model, 'device') else 'cuda/device_map'})")
    MODEL_INFO = model_info

# Metrics
INFERENCE_LATENCY = Histogram('inference_latency_seconds', 'Inference request latency')
MODEL_SWITCHES = Counter('model_switches_total', 'Total number of model switches')

# Initial load
MODEL_INFO = fetch_model_info()
load_model_from_info(MODEL_INFO)

@app.post("/generate")
@INFERENCE_LATENCY.time()
async def generate(data: dict):
    prompt = data.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=128)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return JSONResponse({"response": response})

@app.post("/set-model")
async def set_model(data: dict, request: Request, token: str = Depends(oauth2_scheme)):
    if not verify_token(token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    new_model_id = data.get("model_id")
    if not new_model_id:
        raise HTTPException(status_code=400, detail="Missing model_id")
    switch_model(new_model_id)
    MODEL_SWITCHES.inc()
    # Fetch latest model info after switch
    model_info = fetch_model_info()
    load_model_from_info(model_info)
    return JSONResponse({"status": "ok", "new_model": new_model_id})

@app.get("/model-info")
async def model_info():
    return JSONResponse(MODEL_INFO)

@app.get("/healthz")
async def healthz():
    return JSONResponse({"status": "ok", "model_loaded": model is not None})

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/")
async def chat_page():
    html_content = """..."""  # unchanged, elided for brevity
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
