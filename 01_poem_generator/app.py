import json
import re
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from reward import compute_reward  # функция оценки качества

# === Pydantic-схемы ===

class GenerateRequest(BaseModel):
    input_text: str
    max_new_tokens: int = 60
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 0.8

class Example(BaseModel):
    text: str
    score: float
    model: str

class GenerateResponse(BaseModel):
    examples: list[Example]


# === Инициализация FastAPI и шаблонов ===

app = FastAPI(title="Poetry Continuation API", version="0.1")
templates = Jinja2Templates(directory="templates")


# === Загрузка JSON-профилей ===

BASE_DIR = Path(__file__).parent
with open(BASE_DIR / "length_profiles.json", "r", encoding="utf-8") as f:
    length_profiles = json.load(f)
with open(BASE_DIR / "length_profiles_MLE.json", "r", encoding="utf-8") as f:
    length_profiles_MLE = json.load(f)


def pick_profile(text_len: int, profiles: dict) -> dict:
    if text_len < 50:
        key = "<50"
    elif text_len <= 80:
        key = "50-80"
    elif text_len <= 100:
        key = "80-100"
    else:
        key = ">100"
    return profiles[key]


# === Пути к моделям и устройство ===

EPOCH_MODEL_DIR = BASE_DIR / "epoch-3"
BEST_MODEL_DIR = BASE_DIR / "best_model_optuna_2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(str(EPOCH_MODEL_DIR), local_files_only=True)
model_epoch = AutoModelForSeq2SeqLM.from_pretrained(str(EPOCH_MODEL_DIR), local_files_only=True).eval().to(device)
model_best  = AutoModelForSeq2SeqLM.from_pretrained(str(BEST_MODEL_DIR), local_files_only=True).eval().to(device)


def clean_decode(tokens: torch.Tensor) -> str:
    text = tokenizer.decode(tokens[0], skip_special_tokens=False)
    text = re.sub(r"<extra_id_\d+>", "", text)
    text = text.replace(tokenizer.pad_token or "", "").replace(tokenizer.eos_token or "", "")
    return text.replace("<NL>", "\n").strip(" ,\n")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    text = req.input_text.strip()
    if not text:
        raise HTTPException(400, "Введите текст для продолжения.")

    prompt = "Продолжи стихотворение: " + text
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=64).to(device)
    text_len = len(text)

    def sample_and_score(model, model_name: str, top_n: int):
        profiles = length_profiles if "best_model" in model_name else length_profiles_MLE
        prof = pick_profile(text_len, profiles)

        # рассчитываем max_new_tokens с учётом пользовательского ввода
        user_req = req.max_new_tokens
        profile_max = prof["max_new_tokens"]
        chosen = min(user_req, profile_max)
        max_new = max(40, min(chosen, 200))

        top_k = prof["top_k"]
        top_p = prof["top_p"]
        temperature = prof["temperature"]
        repetition_penalty = prof.get("repetition_penalty", 1.0)

        # теперь 10 пробных генераций
        ids = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            num_return_sequences=20,
            no_repeat_ngram_size=2
        )
        texts = [clean_decode(ids[i : i + 1]) for i in range(10)]
        scored = [(t, compute_reward(t)) for t in texts]
        top = sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]
        return [Example(text=t, score=round(s, 4), model=model_name) for t, s in top]

    best_examples  = sample_and_score(model_best,  "best_model_optuna", 2)
    epoch_examples = sample_and_score(model_epoch, "epoch-3",       1)
    return GenerateResponse(examples=best_examples + epoch_examples)
