import os
import re
import json
import time
import argparse
import requests
import pandas as pd
import numpy as np

API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
MODEL = "yandexgpt-lite"         
MODEL_VERSION = "latest"
BATCH_SIZE = 10
SLEEP_BETWEEN_BATCHES = 0.3  
MAX_RETRIES = 5

CATEGORIES = [
    "Доставка/сроки",
    "Качество товара",
    "Сервис/персонал",
    "Цена/деньги/возврат",
    "Несоответствие описанию",
]
FURNITURE_TYPES = ["Мягкая", "Корпусная", "Кровать/матрас", "Не определено", "Несколько типов"]
SENTIMENTS = ["негативный", "нейтральный", "позитивный"]

RESPONSE_SCHEMA = {
    "schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "description": "номер отзыва из запроса"},
                        "categories": {
                            "type": "array",
                            "items": {"type": "string", "enum": CATEGORIES},
                            "description": "0 или несколько категорий жалобы; пустой массив, если отзыв не содержит жалоб",
                        },
                        "furniture_type": {"type": "string", "enum": FURNITURE_TYPES},
                        "sentiment": {"type": "string", "enum": SENTIMENTS},
                    },
                    "required": ["id", "categories", "furniture_type", "sentiment"],
                },
            }
        },
        "required": ["results"],
    }
}

SYSTEM_PROMPT = f"""Ты аналитик отзывов мебельной компании. Тебе дают пронумерованный список отзывов клиентов.

Для каждого отзыва определи:

1. categories — список категорий ЖАЛОБЫ клиента из набора: {CATEGORIES}.

   КРИТИЧЕСКИ ВАЖНО: категория ставится ТОЛЬКО если клиент выражает недовольство,
   проблему или критику по этой теме. Простое упоминание темы в нейтральном или
   положительном ключе — это НЕ жалоба, категория в этом случае НЕ ставится.

   Примеры НЕПРАВИЛЬНОЙ разметки (так делать не надо):
   - "Доставили быстро, всё понравилось" -> это НЕ жалоба на доставку (это похвала)
   - "Хорошее качество по приемлемой цене, без задержек" -> НЕ жалоба ни на что,
     это полностью позитивный отзыв, categories должен быть []
   - "Есть рассрочка! Удобно!" -> НЕ жалоба на цену, это позитивный комментарий
   - "Сделали всё без нареканий, сборщики чемпионы" -> НЕ жалоба на сервис, это похвала
   - "Норм" -> нет никакой конкретики, categories = [] (не придумывай проблему)

   Примеры ПРАВИЛЬНОЙ разметки:
   - "Ждали доставку 2 месяца, никто не предупредил о задержке" -> ["Доставка/сроки"]
   - "Обещанную скидку 5% сделать забыли" -> ["Цена/деньги/возврат"]
   - "Продавец нахамил, ничего толком не объяснил" -> ["Сервис/персонал"]
   - "Привезли не тот цвет, что заказывали" -> ["Несоответствие описанию"]
   - "Диван за месяц весь скрипит и разваливается" -> ["Качество товара"]

   Если отзыв полностью позитивный (хвалит сервис/доставку/качество/цену без единой
   претензии) — categories должен быть пустым списком [], даже если тема упомянута.
   Если отзыв короткий и без конкретики ("Норм", "Хорошо", "5 звёзд") - тоже [].
   Один отзыв может содержать несколько РАЗНЫХ жалоб одновременно - тогда перечисли все.

2. furniture_type — тип мебели, о которой идёт речь: {FURNITURE_TYPES}.
   "Мягкая" - диваны, кресла. "Корпусная" - шкафы, кухни, столы, комоды. "Не определено", если тип
   мебели не упоминается явно.

3. sentiment — общая тональность текста отзыва: {SENTIMENTS}.

Верни результат строго в формате JSON согласно схеме, для каждого отзыва из запроса, по порядку id."""


def get_headers():
    api_key = os.environ.get("YANDEX_API_KEY")
    if not api_key:
        raise RuntimeError("Переменная окружения YANDEX_API_KEY не задана")
    return {"Content-Type": "application/json", "Authorization": f"Api-Key {api_key}"}


def get_model_uri():
    folder_id = os.environ.get("YANDEX_FOLDER_ID")
    if not folder_id:
        raise RuntimeError("Переменная окружения YANDEX_FOLDER_ID не задана")
    return f"gpt://{folder_id}/{MODEL}/{MODEL_VERSION}"


def call_yandex_gpt(batch):
    """batch: список (id, text). Возвращает dict id -> {categories, furniture_type, sentiment}."""
    user_text = "\n\n".join(f"### Отзыв {i}\n{text[:1500]}" for i, text in batch)
    payload = {
        "modelUri": get_model_uri(),
        "completionOptions": {"stream": False, "temperature": 0.1, "maxTokens": "2000"},
        "messages": [
            {"role": "system", "text": SYSTEM_PROMPT},
            {"role": "user", "text": user_text},
        ],
        "json_schema": RESPONSE_SCHEMA,
    }

    for attempt in range(MAX_RETRIES):
        resp = requests.post(API_URL, headers=get_headers(), json=payload, timeout=60)
        if resp.status_code == 200:
            break
        if resp.status_code == 429:
            wait = 2 ** attempt
            print(f"  429 Too Many Requests, жду {wait}с...")
            time.sleep(wait)
            continue
        raise RuntimeError(f"YandexGPT API error {resp.status_code}: {resp.text[:500]}")
    else:
        raise RuntimeError("Превышено число попыток после 429")

    data = resp.json()
    raw_text = data["result"]["alternatives"][0]["message"]["text"]
    parsed = json.loads(raw_text)

    out = {}
    for item in parsed.get("results", []):
        out[int(item["id"])] = {
            "categories": item.get("categories", []),
            "furniture_type": item.get("furniture_type", "Не определено"),
            "sentiment": item.get("sentiment", "нейтральный"),
        }
    return out


def extract_price(text):
    m = re.search(r'(\d[\d\s]{2,7})\s*(тыс|т\.р|тр\b|руб)', text.lower())
    if m:
        num = re.sub(r'\s', '', m.group(1))
        try:
            val = int(num)
            if 'тыс' in m.group(2) or 'т' in m.group(2):
                val *= 1000
            if val < 1000 or val > 3_000_000:
                return np.nan
            return val
        except ValueError:
            return np.nan
    return np.nan


def emotionality_score(text):
    if not text:
        return np.nan
    excl = text.count('!')
    caps_words = len(re.findall(r'\b[А-ЯЁ]{3,}\b', str(text)))
    emo_words = len(re.findall(r'ужас|кошмар|отвратительн|восторг|прекрасн|обожа|бесит|разочарован|в шоке|никогда больше', text))
    return excl + caps_words + emo_words


def load_checkpoint(path):
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done[rec["row_id"]] = rec
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../output/reviews_clean.xlsx")
    ap.add_argument("--output", default="../output/reviews_categorized_gpt.xlsx")
    ap.add_argument("--checkpoint", default="../output/gpt_checkpoint.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="обработать только первые N отзывов (для быстрого теста)")
    ap.add_argument("--sample", type=int, default=None,
                     help="случайная выборка из N отзывов, стратифицированная по рейтингу "
                          "(сохраняет исходные пропорции 1-2★/3★/4-5★) - для контроля бюджета API")
    ap.add_argument("--seed", type=int, default=42, help="seed для воспроизводимости выборки")
    ap.add_argument("--resume", action="store_true", help="продолжить с checkpoint-файла")
    args = ap.parse_args()

    df = pd.read_excel(args.input)
    df = df.reset_index().rename(columns={"index": "row_id"})

    texted = df[df["Есть_текст"] == True].copy()

    if args.sample:
        bucket = pd.cut(texted["Оценка автора"], bins=[0, 2, 3, 5], labels=["neg", "neutral", "pos"])
        frac = args.sample / len(texted)
        texted = (
            texted.groupby(bucket, group_keys=False, observed=True)
            .apply(lambda g: g.sample(frac=frac, random_state=args.seed))
        )
        print(f"Стратифицированная выборка: {len(texted)} отзывов "
              f"(целились в {args.sample}, распределение по рейтингу сохранено)")
        print(texted["Оценка автора"].apply(lambda x: "1-2★" if x <= 2 else ("3★" if x == 3 else "4-5★")).value_counts())
    elif args.limit:
        texted = texted.head(args.limit)

    done = load_checkpoint(args.checkpoint) if args.resume else {}
    todo = texted[~texted["row_id"].isin(done.keys())]
    print(f"Всего отзывов с текстом: {len(texted)}, уже обработано: {len(done)}, осталось: {len(todo)}")

    checkpoint_f = open(args.checkpoint, "a", encoding="utf-8")
    rows = list(zip(todo["row_id"], todo["Текст"]))

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        # локальные id 0..N-1 внутри батча, чтобы не пугать модель большими числами
        local_batch = [(i, text) for i, (_, text) in enumerate(batch)]
        try:
            result = call_yandex_gpt(local_batch)
        except Exception as e:
            print(f"Ошибка на батче {start}-{start+len(batch)}: {e}")
            print("Прогресс сохранён в checkpoint, перезапустите с --resume")
            break

        for local_id, (row_id, text) in enumerate(batch):
            rec = result.get(local_id, {"categories": [], "furniture_type": "Не определено", "sentiment": "нейтральный"})
            rec["row_id"] = int(row_id)
            checkpoint_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # самопроверка: позитивный отзыв с жалобами - подозрительно, стоит перепроверить промпт/пример
            if rec["sentiment"] == "позитивный" and rec["categories"]:
                snippet = text[:80].replace("\n", " ")
                print(f"  ⚠ подозрительно: row_id={row_id} sentiment=позитивный, "
                      f"но categories={rec['categories']} | текст: {snippet}...")
        checkpoint_f.flush()

        done_n = start + len(batch)
        print(f"Обработано {done_n}/{len(rows)}")
        time.sleep(SLEEP_BETWEEN_BATCHES)

    checkpoint_f.close()

    # ---------- сборка финального файла (схема совпадает с 03_categorize.py) ----------
    done = load_checkpoint(args.checkpoint)
    for cat in CATEGORIES:
        df[f"Жалоба_{cat}"] = False
    df["Есть_жалоба"] = False
    df["Тип_мебели"] = "Не определено"
    df["Тональность_gpt"] = pd.Series([None] * len(df), dtype=object)
    df["Категории_жалоб"] = [[] for _ in range(len(df))]

    for row_id, rec in done.items():
        if row_id not in df["row_id"].values:
            continue
        idx = df.index[df["row_id"] == row_id][0]
        cats = rec.get("categories", [])
        for c in cats:
            if c in CATEGORIES:
                df.at[idx, f"Жалоба_{c}"] = True
        df.at[idx, "Есть_жалоба"] = len(cats) > 0
        df.at[idx, "Категории_жалоб"] = cats
        df.at[idx, "Тип_мебели"] = rec.get("furniture_type", "Не определено")
        df.at[idx, "Тональность_gpt"] = rec.get("sentiment", np.nan)

    df["N_категорий"] = df["Категории_жалоб"].apply(len)

    # те же доп. признаки, что и в rule-based версии (не связаны с категоризацией) -
    # нужны, чтобы 04_hypotheses.py и 05_prepare_dashboard_data.py работали без правок
    df["Упомянутая_цена"] = df["Текст"].fillna('').apply(extract_price)
    df["Эмоциональность"] = df["Текст"].fillna('').apply(emotionality_score)
    df.loc[~df["Есть_текст"], "Эмоциональность"] = np.nan

    df.drop(columns=["row_id", "Категории_жалоб"]).to_excel(args.output, index=False)
    print(f"\nГотово. Обработано записей из checkpoint: {len(done)}")
    print(f"Сохранено: {args.output}")


if __name__ == "__main__":
    main()
