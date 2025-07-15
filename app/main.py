import os
from huggingface_hub import login
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from app.registry import fetch_model_info, switch_model

app = FastAPI()

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

model = None
tokenizer = None
MODEL_INFO = None


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
        token=HF_TOKEN
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on device: {device}")
    MODEL_INFO = model_info

# Initial load
MODEL_INFO = fetch_model_info()
load_model_from_info(MODEL_INFO)

@app.post("/generate")
async def generate(data: dict):
    prompt = data.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")
    device = model.device if hasattr(model, "device") else torch.device("cpu")
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=128)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return JSONResponse({"response": response})

@app.post("/set-model")
async def set_model(data: dict, request: Request):
    new_model_id = data.get("model_id")
    if not new_model_id:
        raise HTTPException(status_code=400, detail="Missing model_id")
    switch_model(new_model_id)
    # Fetch latest model info after switch
    model_info = fetch_model_info()
    load_model_from_info(model_info)
    return JSONResponse({"status": "ok", "new_model": new_model_id})

@app.get("/model-info")
async def model_info():
    return JSONResponse(MODEL_INFO)

@app.get("/")
async def chat_page():
    html_content = """
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <title>LLM Chat</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f4f4f4; }
            #chatbox { width: 100%; max-width: 600px; margin: 40px auto; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px #ccc; padding: 24px; }
            #messages { height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 12px; margin-bottom: 16px; background: #fafafa; }
            .user { color: #0078d7; font-weight: bold; }
            .bot { color: #333; }
            #prompt { width: 80%; padding: 8px; }
            #send { padding: 8px 16px; }
        </style>
    </head>
    <body>
        <div id='chatbox'>
            <h2>Chat with LLM</h2>
            <div id='messages'></div>
            <input type='text' id='prompt' placeholder='Type your message...' />
            <button id='send'>Send</button>
        </div>
        <script>
            const messages = document.getElementById('messages');
            const promptInput = document.getElementById('prompt');
            const sendBtn = document.getElementById('send');
            function addMessage(sender, text) {
                const div = document.createElement('div');
                div.innerHTML = `<span class='${sender}'>${sender === 'user' ? 'You' : 'Bot'}:</span> ${text}`;
                messages.appendChild(div);
                messages.scrollTop = messages.scrollHeight;
            }
            sendBtn.onclick = async function() {
                const prompt = promptInput.value;
                if (!prompt) return;
                addMessage('user', prompt);
                promptInput.value = '';
                const res = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt })
                });
                const data = await res.json();
                addMessage('bot', data.response);
            };
            promptInput.addEventListener('keydown', function(e) {
                if (e.key === 'Enter') sendBtn.click();
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
