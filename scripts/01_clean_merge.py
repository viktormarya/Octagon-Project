import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

pd.set_option('display.max_colwidth', 100)

RU_MONTHS = {
    'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4, 'мая': 5, 'июня': 6,
    'июля': 7, 'августа': 8, 'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12
}

SCRAPE_DATE = {
    'Google': datetime(2024, 6, 1),
    'Yandex': datetime(2023, 6, 1),
    '2GIS':   datetime(2024, 4, 15), 
}


def parse_date(raw, source):
    if pd.isna(raw):
        return pd.NaT
    s = str(raw).strip().lower()
    ref = SCRAPE_DATE[source]

    if s.startswith('сегодня'):
        return ref
    if s.startswith('вчера'):
        return ref - timedelta(days=1)
    m = re.match(r'(\d+)?\s*(час|часа|часов|день|дня|дней|неделю|недели|недель|месяц|месяца|месяцев|год|года|лет)\s*назад', s)
    if m:
        n = int(m.group(1)) if m.group(1) else 1
        unit = m.group(2)
        if unit.startswith('час'):
            return ref - timedelta(hours=n)
        if unit.startswith('д'):
            return ref - timedelta(days=n)
        if unit.startswith('недел'):
            return ref - timedelta(weeks=n)
        if unit.startswith('месяц'):
            return ref - pd.DateOffset(months=n)
        if unit.startswith('год') or unit.startswith('лет'):
            return ref - pd.DateOffset(years=n)

    s_clean = s.split(',')[0].strip()
    m2 = re.match(r'(\d{1,2})\s+([а-я]+)\s*(\d{4})?', s_clean)
    if m2:
        day = int(m2.group(1))
        month = RU_MONTHS.get(m2.group(2))
        year = int(m2.group(3)) if m2.group(3) else ref.year
        if month:
            try:
                return datetime(year, month, day)
            except ValueError:
                return pd.NaT
    return pd.NaT


def load_source(path, source_name):
    df = pd.read_excel(path)
    df['Источник'] = source_name
    return df


def main():
    files = {
        'Google': '../data/reviews_google.xlsx',
        'Yandex': '../data/reviews_yandex__1_.xlsx',
        '2GIS':   '../data/reviews_2gis.xlsx',
    }

    dfs = [load_source(p, name) for name, p in files.items()]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Загружено строк всего: {len(df)}")

    df['Оценка'] = (
        df['Оценка'].astype(str).str.replace(',', '.', regex=False).str.strip()
    )
    df['Оценка'] = pd.to_numeric(df['Оценка'], errors='coerce')

    df['Оценка автора'] = (
        df['Оценка автора'].astype(str).str.replace(',', '.', regex=False)
        .str.extract(r'(\d+(?:\.\d+)?)')[0]
    )
    df['Оценка автора'] = pd.to_numeric(df['Оценка автора'], errors='coerce')
    df['Наименование'] = (
        df['Наименование'].astype(str).str.strip()
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip('"\' ')
    )

    # --- парсинг дат ---
    df['Дата_raw'] = df['Дата']
    df['Дата'] = df.apply(lambda r: parse_date(r['Дата_raw'], r['Источник']), axis=1)

    # --- флаг наличия текста отзыва ---
    df['Текст'] = df['Текст'].astype(str).replace('nan', np.nan)
    df['Есть_текст'] = df['Текст'].notna() & (df['Текст'].str.strip() != '')

    # --- количество лайков/дизлайков -> 0 если пусто (это не "неизвестно", а просто 0 реакций) ---
    df['Like'] = pd.to_numeric(df['Like'], errors='coerce').fillna(0).astype(int)
    df['Dislike'] = pd.to_numeric(df['Dislike'], errors='coerce').fillna(0).astype(int)

    # --- длина отзыва (в словах) - пригодится для гипотез 4 и 5 ---
    df['Длина_отзыва_слов'] = df['Текст'].fillna('').apply(lambda t: len(t.split()))
    df.loc[~df['Есть_текст'], 'Длина_отзыва_слов'] = np.nan

    # --- удаление явных дублей (один и тот же автор+дата+текст+компания) ---
    before = len(df)
    df = df.drop_duplicates(subset=['Наименование', 'Автор', 'Дата_raw', 'Текст'], keep='first')
    print(f"Удалено дублей: {before - len(df)}")

    # --- сортировка колонок ---
    cols = ['Источник', 'Наименование', 'Адрес', 'Координаты объекта', 'Оценка',
            'Количество оценок', 'Количество отзывов', 'Автор', 'Статус автора',
            'Оценка автора', 'Дата', 'Дата_raw', 'Текст', 'Есть_текст',
            'Длина_отзыва_слов', 'Like', 'Dislike']
    df = df[cols]

    print(f"\nИтоговое число строк: {len(df)}")
    print(f"Уникальных компаний: {df['Наименование'].nunique()}")
    print(f"Строк с текстом отзыва: {df['Есть_текст'].sum()} ({df['Есть_текст'].mean()*100:.1f}%)")
    print(f"Диапазон дат: {df['Дата'].min()} — {df['Дата'].max()}")
    print(f"\nПропуски по колонкам:\n{df.isna().sum()}")

    out_path = '../output/reviews_clean.xlsx'
    df.to_excel(out_path, index=False)
    print(f"\nСохранено: {out_path}")

    # быстрая сводка по источникам
    print("\n--- Сводка по источникам ---")
    print(df.groupby('Источник').agg(
        строк=('Наименование', 'size'),
        компаний=('Наименование', 'nunique'),
        средняя_оценка_автора=('Оценка автора', 'mean'),
        доля_с_текстом=('Есть_текст', 'mean'),
    ).round(2))


if __name__ == '__main__':
    main()
