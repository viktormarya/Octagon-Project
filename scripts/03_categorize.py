import re
import pandas as pd
import numpy as np

df = pd.read_excel('../output/reviews_clean.xlsx')
df['Дата'] = pd.to_datetime(df['Дата'])
df['Текст_lower'] = df['Текст'].fillna('').str.lower()

# ============================================================
# Словари категорий жалоб / тем (rule-based, по корням слов)
# ============================================================
CATEGORIES = {
    'Доставка/сроки': [
        r'доставк', r'привезл', r'привоз', r'опозда', r'задерж', r'просрочен',
        r'курьер', r'долго ждал', r'ждали \d', r'месяц\w* ждал', r'не привезли',
        r'срок\w* доставк', r'перенос\w* доставк',
    ],
    'Качество товара': [
        r'брак', r'сломал', r'треснул', r'скрип', r'дефект', r'царапин', r'испорчен',
        r'некачествен', r'разваливается', r'разошел', r'выцвел', r'вонь', r'запах хим',
        r'плохого качества', r'низкое качество',
    ],
    'Сервис/персонал': [
        r'груб', r'хамств', r'хамк', r'нев[еэ]жлив', r'не перезвонил', r'не отвечал',
        r'игнорир', r'обслуживан', r'менеджер\w* (плохо|ужас|груб|не)', r'равнодушн',
        r'наплевательск', r'отношени\w* (плохо|ужас|наплевательск)',
    ],
    'Цена/деньги/возврат': [
        r'обман', r'развод', r'вернуть деньги', r'возврат денег', r'не вернул',
        r'переплат', r'дорого', r'навязыва', r'скрытые платеж', r'доплат',
        r'деньги забрали', r'украли деньги',
    ],
    'Несоответствие описанию': [
        r'не соответствует', r'не тот цвет', r'другой цвет', r'не то что заказыва',
        r'не такой как на фото', r'не совпада', r'подмен', r'привезли не то',
        r'другая модель', r'не тот размер', r'ошиблись с',
    ],
}

def categorize(text):
    found = []
    for cat, patterns in CATEGORIES.items():
        if any(re.search(p, text) for p in patterns):
            found.append(cat)
    return found

df['Категории_жалоб'] = df['Текст_lower'].apply(categorize)
df['N_категорий'] = df['Категории_жалоб'].apply(len)
df['Есть_жалоба'] = df['N_категорий'] > 0

for cat in CATEGORIES:
    df[f'Жалоба_{cat}'] = df['Категории_жалоб'].apply(lambda lst: cat in lst)

# ============================================================
# Тип мебели (для гипотезы 4)
# ============================================================
SOFT_FURNITURE = [r'диван', r'кресл', r'пуф', r'оттоман']
CORPUS_FURNITURE = [r'шкаф', r'кухн', r'стол\b', r'комод', r'стеллаж', r'гардероб', r'тумб']
BED = [r'кроват', r'матрас']

def furniture_type(text):
    is_soft = any(re.search(p, text) for p in SOFT_FURNITURE)
    is_corpus = any(re.search(p, text) for p in CORPUS_FURNITURE)
    is_bed = any(re.search(p, text) for p in BED)
    tags = []
    if is_soft: tags.append('Мягкая')
    if is_corpus: tags.append('Корпусная')
    if is_bed: tags.append('Кровать/матрас')
    if len(tags) == 1:
        return tags[0]
    elif len(tags) == 0:
        return 'Не определено'
    else:
        return 'Несколько типов'

df['Тип_мебели'] = df['Текст_lower'].apply(furniture_type)

# ============================================================
# Упоминание цены (для гипотезы 5) - грубая эвристика
# ============================================================
def extract_price(text):
    # ищем "XXX тыс", "XXXXX руб", "XXX 000"
    m = re.search(r'(\d[\d\s]{2,7})\s*(тыс|т\.р|тр\b|руб)', text)
    if m:
        num = re.sub(r'\s', '', m.group(1))
        try:
            val = int(num)
            if 'тыс' in m.group(2) or 'т' in m.group(2):
                val *= 1000
            # отсекаем нереалистичные значения (вероятно, шум regex - даты, телефоны и т.п.)
            if val < 1000 or val > 3_000_000:
                return np.nan
            return val
        except ValueError:
            return np.nan
    return np.nan

df['Упомянутая_цена'] = df['Текст_lower'].apply(extract_price)

# ============================================================
# Эмоциональность (для гипотезы 4) - простая эвристика
# ============================================================
def emotionality_score(text):
    if not text:
        return np.nan
    excl = text.count('!')
    caps_words = len(re.findall(r'\b[А-ЯЁ]{3,}\b', str(text)))
    emo_words = len(re.findall(r'ужас|кошмар|отвратительн|восторг|прекрасн|обожа|бесит|разочарован|в шоке|никогда больше', text))
    return excl + caps_words + emo_words

df['Эмоциональность'] = df['Текст'].fillna('').apply(emotionality_score)
df.loc[~df['Есть_текст'], 'Эмоциональность'] = np.nan

out_path = '../output/reviews_categorized.xlsx'
df.drop(columns=['Текст_lower']).to_excel(out_path, index=False)

print("Сохранено:", out_path)
print(f"\nВсего отзывов с текстом: {df['Есть_текст'].sum()}")
print(f"Отзывов, где найдена хотя бы одна категория жалобы: {df['Есть_жалоба'].sum()} ({df['Есть_жалоба'].mean()*100:.1f}%)")

print("\n=== Частота категорий жалоб (по всем отзывам с текстом) ===")
for cat in CATEGORIES:
    n = df[f'Жалоба_{cat}'].sum()
    print(f"  {cat}: {n} ({n/df['Есть_текст'].sum()*100:.1f}%)")

print("\n=== Частота категорий жалоб СРЕДИ НЕГАТИВНЫХ отзывов (1-2 звезды) ===")
neg = df[df['Оценка автора'] <= 2]
for cat in CATEGORIES:
    n = neg[f'Жалоба_{cat}'].sum()
    print(f"  {cat}: {n} ({n/len(neg)*100:.1f}% от {len(neg)} негативных)")

print("\n=== Тип мебели: распределение ===")
print(df['Тип_мебели'].value_counts())

print("\n=== Упомянутая цена: найдено значений ===")
print(df['Упомянутая_цена'].notna().sum())
print(df['Упомянутая_цена'].describe())
