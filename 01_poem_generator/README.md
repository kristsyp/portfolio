markdown
# 🎭 Poem Generator

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-FFD21E)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> Лёгковесный сервис для генерации поэтических продолжений с помощью Seq2Seq-моделей из HuggingFace Transformers.  
> Предоставляет **REST API** и простой **веб-интерфейс** на Bootstrap.

---

## 🔧 Установка

### 1️⃣ Клонируйте репозиторий

```bash
git clone git@github.com:KsuZavyalova/poem_generator.git
cd poem_generator
2️⃣ Скачайте модели
Модели доступны по ссылке:
🔗 Google Drive

Скачайте и разместите папки epoch-3/ и best_model_optuna_2/ в корне проекта.

### 3️⃣ Создайте виртуальное окружение
Linux / macOS:

bash
python3 -m venv venv
source venv/bin/activate
Windows:

bash
python -m venv venv
venv\Scripts\activate
4️⃣ Установите зависимости
bash
pip install --upgrade pip
pip install -r requirements.txt
📁 Структура проекта
text
poem_generator/
│
├── app.py                          # Основной файл приложения (FastAPI)
├── reward.py                       # Функция оценки качества текста
├── requirements.txt                # Зависимости
│
├── length_profiles.json            # Профили семплинга (PPO-модель)
├── length_profiles_MLE.json        # Профили семплинга (MLE-модель)
│
├── templates/
│   └── index.html                  # Веб-интерфейс (Bootstrap)
│
├── epoch-3/                        # MLE-модель (3 эпохи)
├── best_model_optuna_2/            # PPO-модель (Optuna)
│
└── README.md
⚙️ Конфигурация
Файлы конфигурации:

length_profiles.json — оптимальные параметры семплинга для PPO-модели (зависят от длины продолжения)

length_profiles_MLE.json — оптимальные параметры семплинга для MLE-модели

Модели:

epoch-3/ — базовая модель, обученная методом MLE на отфильтрованном корпусе

best_model_optuna_2/ — дообученная модель (PPO + Optuna) с учётом метра, ритма и рифмы

⚠️ Важно: Директории моделей должны находиться в корне проекта рядом с app.py.

▶️ Запуск
bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
Параметры:

--reload — авто-перезагрузка при изменении кода (для разработки)

--host 0.0.0.0 — доступ с любого устройства в сети

--port 8000 — порт для запуска

После запуска откройте в браузере:

Веб-интерфейс: http://localhost:8000

Swagger UI: http://localhost:8000/docs

ReDoc: http://localhost:8000/redoc

🌐 Веб-интерфейс
Откройте http://localhost:8000

Введите начальные строки стихотворения

Укажите желаемую длину продолжения (в токенах)

Нажмите «Сгенерировать»

Результаты отобразятся в виде карточек:

Варианты от MLE-модели

Варианты от PPO-модели

Оценка качества (Reward Score)

Время выполнения

📚 API
Эндпоинты:

POST /generate — генерация поэтического продолжения

GET /health — проверка работоспособности сервиса

Документация:

Swagger UI: http://localhost:8000/docs

ReDoc: http://localhost:8000/redoc

🛠 Стек технологий
FastAPI — REST API фреймворк

Uvicorn — ASGI-сервер

Transformers — загрузка и инференс моделей ruT5

PyTorch — бэкенд для моделей

Bootstrap — веб-интерфейс

Jinja2 — шаблонизатор HTML

📝 Примечания
Сервис использует две модели: MLE (базовая) и PPO (дообученная)

Параметры семплинга (top_k, top_p, temperature) автоматически подбираются под длину продолжения

Reward Score рассчитывается на основе метра, ритма, рифмы и лексического разнообразия


