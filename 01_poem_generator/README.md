# Poem Generator

Лёгковесный сервис для генерации поэтических продолжений с помощью Seq2Seq-моделей из HuggingFace Transformers. Предоставляет REST API и простой веб-интерфейс на Bootstrap.

---

## Установка

### 1. Клонируйте репозиторий

git clone git@github.com:kristsyp/01_poem_generator.git
cd 01_poem_generator

### 2. Скачайте модели

Модели доступны по ссылке:
https://drive.google.com/drive/folders/1NlOr0GRGB3UwLl2_FOyVGOxSVxn0A-9x?usp=sharing

Скачайте и разместите папки epoch-3/ и best_model_optuna_2/ в корне проекта.

### 3. Создайте виртуальное окружение

Linux / macOS:

python3 -m venv venv
source venv/bin/activate

Windows:

python -m venv venv
venv\Scripts\activate

### 4. Установите зависимости

pip install --upgrade pip
pip install -r requirements.txt

---

## Структура проекта

poem_generator/
├── app.py
├── reward.py
├── requirements.txt
├── length_profiles.json
├── length_profiles_MLE.json
├── templates/
│   └── index.html
├── epoch-3/
├── best_model_optuna_2/
└── README.md

---

## Конфигурация

- length_profiles.json - оптимальные параметры семплинга для PPO-модели
- length_profiles_MLE.json - оптимальные параметры семплинга для MLE-модели
- epoch-3/ - базовая модель MLE
- best_model_optuna_2/ - дообученная модель PPO + Optuna

Важно: директории моделей должны находиться в корне проекта рядом с app.py.

---

## Запуск

uvicorn app:app --reload --host 0.0.0.0 --port 8000

После запуска откройте в браузере:

- Веб-интерфейс: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Веб-интерфейс

1. Откройте http://localhost:8000
2. Введите исходный текст
3. Укажите желаемую длину продолжения
4. Нажмите "Сгенерировать"

Результаты отобразятся в виде карточек с вариантами от MLE и PPO моделей, оценкой качества и временем выполнения.

---

## API

Эндпоинты:

- POST /generate - генерация поэтического продолжения
- GET /health - проверка работоспособности

Документация: http://localhost:8000/docs

---

## Стек технологий

- FastAPI
- Uvicorn
- Transformers (HuggingFace)
- PyTorch
- Bootstrap
- Jinja2

---

## Примечания

- Сервис использует две модели: MLE и PPO
- Параметры семплинга подбираются автоматически под длину продолжения
- Reward Score рассчитывается на основе метра, ритма, рифмы и лексики

