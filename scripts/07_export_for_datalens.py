"""
Подготовка данных для Yandex DataLens: экспорт в CSV с "плоской" и
предагрегированной структурой, чтобы дашборд собирался просто drag-and-drop,
без сложных LOD-вычислений внутри DataLens.

Запуск:
    python3 07_export_for_datalens.py --input ../output/reviews_categorized.xlsx
    python3 07_export_for_datalens.py --input ../output/reviews_categorized_gpt.xlsx --suffix _gpt

Результат (в output/datalens/):
    reviews_flat.csv        - построчные данные, для фильтров/срезов/произвольных чартов
    complaints_long.csv     - категории жалоб в长 формате, для bar chart без LOD
    yearly_trend.csv        - число отзывов и средняя оценка по годам
    hypotheses_summary.csv  - сводка по 5 гипотезам с вердиктами и p-value
    companies_top.csv       - топ/худшие компании по рейтингу
"""
import re
import argparse
import pandas as pd
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument('--input', default='../output/reviews_categorized.xlsx')
ap.add_argument('--outdir', default='../output/datalens')
ap.add_argument('--suffix', default='', help="суффикс к именам файлов, напр. '_gpt'")
args = ap.parse_args()

import os
os.makedirs(args.outdir, exist_ok=True)

df = pd.read_excel(args.input)
df['Дата'] = pd.to_datetime(df['Дата'])
df['Год'] = df['Дата'].dt.year
df['Текст_lower'] = df['Текст'].fillna('').str.lower()

CATS = ['Доставка/сроки', 'Качество товара', 'Сервис/персонал', 'Цена/деньги/возврат', 'Несоответствие описанию']

# ==================================================================
# 1. reviews_flat.csv - построчные данные
# ==================================================================
flat = pd.DataFrame({
    'Источник': df['Источник'],
    'Компания': df['Наименование'],
    'Оценка': df['Оценка автора'],
    'Тональность': df['Оценка автора'].apply(lambda x: 'Негатив (1-2★)' if x <= 2 else ('Нейтрально (3★)' if x == 3 else 'Позитив (4-5★)')),
    'Дата': df['Дата'].dt.strftime('%Y-%m-%d'),
    'Год': df['Год'],
    'Длина_слов': df['Длина_отзыва_слов'],
    'Есть_текст': df['Есть_текст'].astype(int),
    'Тип_мебели': df['Тип_мебели'] if 'Тип_мебели' in df.columns else 'Не определено',
    'Эмоциональность': df['Эмоциональность'] if 'Эмоциональность' in df.columns else np.nan,
    'Цена_упомянута_руб': df['Упомянутая_цена'] if 'Упомянутая_цена' in df.columns else np.nan,
})
for cat in CATS:
    col = f'Жалоба_{cat}'
    flat[f'Жалоба: {cat}'] = df[col].astype(int) if col in df.columns else 0

flat_path = f'{args.outdir}/reviews_flat{args.suffix}.csv'
flat.to_csv(flat_path, index=False, encoding='utf-8-sig')  # utf-8-sig - чтобы Excel/DataLens не путали кодировку
print(f"Сохранено: {flat_path} ({len(flat)} строк)")

# ==================================================================
# 2. complaints_long.csv - категории жалоб в длинном формате
# ==================================================================
# Важно: если это GPT-файл с частичной выборкой (--sample), размечены не все
# строки - база для % должна считаться по реально размеченному подмножеству,
# а не по всему датасету, иначе доли будут занижены в десятки раз.
if 'Тональность_gpt' in df.columns:
    labeled_mask = df['Тональность_gpt'].notna()
    total_texted = int(labeled_mask.sum())
    neg = df[(df['Оценка автора'] <= 2) & labeled_mask]
    print(f"Обнаружена GPT-разметка: размечено {total_texted} из {df['Есть_текст'].sum()} отзывов с текстом. "
          f"База для % считается по размеченному подмножеству.")
else:
    total_texted = int(df['Есть_текст'].sum())
    neg = df[df['Оценка автора'] <= 2]

rows = []
for cat in CATS:
    col = f'Жалоба_{cat}'
    if col not in df.columns:
        continue
    n_overall = int(df[col].sum())
    n_neg = int(neg[col].sum())
    rows.append({'Категория': cat, 'Группа': 'Все отзывы', 'Количество': n_overall,
                 'База': int(total_texted), 'Доля_%': round(n_overall / total_texted * 100, 1)})
    rows.append({'Категория': cat, 'Группа': 'Только негативные (1-2★)', 'Количество': n_neg,
                 'База': int(len(neg)), 'Доля_%': round(n_neg / len(neg) * 100, 1)})
complaints_long = pd.DataFrame(rows)
complaints_path = f'{args.outdir}/complaints_long{args.suffix}.csv'
complaints_long.to_csv(complaints_path, index=False, encoding='utf-8-sig')
print(f"Сохранено: {complaints_path}")

# ==================================================================
# 3. yearly_trend.csv
# ==================================================================
yearly = df[(df['Год'] >= 2015) & (df['Год'] <= 2024)].groupby('Год').agg(
    Число_отзывов=('Наименование', 'size'), Средняя_оценка=('Оценка автора', 'mean')).round(2).reset_index()
yearly_path = f'{args.outdir}/yearly_trend{args.suffix}.csv'
yearly.to_csv(yearly_path, index=False, encoding='utf-8-sig')
print(f"Сохранено: {yearly_path}")

# ==================================================================
# 4. hypotheses_summary.csv
# ==================================================================
SERVICE_COLS = ['Жалоба_Доставка/сроки', 'Жалоба_Сервис/персонал', 'Жалоба_Цена/деньги/возврат']
PRODUCT_COLS = ['Жалоба_Качество товара', 'Жалоба_Несоответствие описанию']
neg2 = neg.copy()
neg2['service'] = neg2[SERVICE_COLS].any(axis=1)
neg2['product'] = neg2[PRODUCT_COLS].any(axis=1)
sub = neg2[neg2['service'] | neg2['product']]
cs, cp = int(sub['service'].sum()), int(sub['product'].sum())
p1 = stats.binomtest(cs, cs + cp, p=0.5).pvalue

FAST = re.compile(r'быстро.{0,40}(доставили|привезли|доставк)|доставили.{0,10}быстро|привезли.{0,15}(быстро|раньше срока|раньше)')
SLOW = re.compile(r'долго.{0,40}(доставк|привезл|жда)|доставк\w*.{0,20}(задерж|перенос|опозда)|опозда|задерж\w*.{0,15}(доставк|привоз)|не привезли вовремя')
df['fast'] = df['Текст_lower'].apply(lambda t: bool(FAST.search(t)))
df['slow'] = df['Текст_lower'].apply(lambda t: bool(SLOW.search(t)))
fast_r = df.loc[df['fast'] & ~df['slow'], 'Оценка автора'].dropna()
slow_r = df.loc[df['slow'] & ~df['fast'], 'Оценка автора'].dropna()
_, p2 = stats.mannwhitneyu(fast_r, slow_r, alternative='greater')

overall_counts = {c: int(df[f'Жалоба_{c}'].sum()) for c in CATS if f'Жалоба_{c}' in df.columns}
sorted_cats = sorted(overall_counts.items(), key=lambda x: -x[1])

soft = df[(df['Тип_мебели'] == 'Мягкая') & df['Есть_текст']] if 'Тип_мебели' in df.columns else df.iloc[0:0]
corpus = df[(df['Тип_мебели'] == 'Корпусная') & df['Есть_текст']] if 'Тип_мебели' in df.columns else df.iloc[0:0]
if len(soft) > 5 and len(corpus) > 5:
    _, p4 = stats.mannwhitneyu(soft['Длина_отзыва_слов'].dropna(), corpus['Длина_отзыва_слов'].dropna())
else:
    p4 = np.nan

priced = df[df['Упомянутая_цена'].notna() & df['Есть_текст']] if 'Упомянутая_цена' in df.columns else df.iloc[0:0]
if len(priced) > 10:
    rho, p5 = stats.spearmanr(priced['Упомянутая_цена'], priced['Длина_отзыва_слов'])
else:
    rho, p5 = np.nan, np.nan

hyp_rows = [
    {'ID': 1, 'Гипотеза': 'Негатив из-за сервиса, а не качества мебели',
     'Вердикт': 'Подтверждена', 'Метрика_1': 'Сервис, % от размеченных негативных', 'Значение_1': round(cs/len(sub)*100, 1) if len(sub) else np.nan,
     'Метрика_2': 'Товар, % от размеченных негативных', 'Значение_2': round(cp/len(sub)*100, 1) if len(sub) else np.nan,
     'p_value': p1},
    {'ID': 2, 'Гипотеза': 'Быстрая доставка повышает оценку',
     'Вердикт': 'Подтверждена', 'Метрика_1': 'Средняя оценка, быстрая доставка', 'Значение_1': round(fast_r.mean(), 2) if len(fast_r) else np.nan,
     'Метрика_2': 'Средняя оценка, медленная доставка', 'Значение_2': round(slow_r.mean(), 2) if len(slow_r) else np.nan,
     'p_value': p2},
    {'ID': 3, 'Гипотеза': 'Несоответствие описанию - главная причина претензий',
     'Вердикт': 'Не подтверждена', 'Метрика_1': f'Упоминаний топ-категории ({sorted_cats[0][0] if sorted_cats else "?"})', 'Значение_1': sorted_cats[0][1] if sorted_cats else np.nan,
     'Метрика_2': 'Несоответствие описанию - место в рейтинге (из 5)', 'Значение_2': [c for c,_ in sorted_cats].index('Несоответствие описанию')+1 if sorted_cats else np.nan,
     'p_value': np.nan},
    {'ID': 4, 'Гипотеза': 'Мягкая мебель -> длиннее/эмоциональнее отзывы',
     'Вердикт': 'Не подтверждена (эффект обратный)', 'Метрика_1': 'Длина отзыва, мягкая мебель', 'Значение_1': round(soft['Длина_отзыва_слов'].mean(),1) if len(soft) else np.nan,
     'Метрика_2': 'Длина отзыва, корпусная мебель', 'Значение_2': round(corpus['Длина_отзыва_слов'].mean(),1) if len(corpus) else np.nan,
     'p_value': p4},
    {'ID': 5, 'Гипотеза': 'Выше цена -> подробнее отзыв',
     'Вердикт': 'Недостаточно данных', 'Метрика_1': 'Корреляция Спирмена (rho)', 'Значение_1': round(rho, 3) if not np.isnan(rho) else np.nan,
     'Метрика_2': '% отзывов с явной ценой', 'Значение_2': round(len(priced)/total_texted*100, 2) if total_texted else np.nan,
     'p_value': p5},
]
hyp_df = pd.DataFrame(hyp_rows)
hyp_path = f'{args.outdir}/hypotheses_summary{args.suffix}.csv'
hyp_df.to_csv(hyp_path, index=False, encoding='utf-8-sig')
print(f"Сохранено: {hyp_path}")

# ==================================================================
# 5. companies_top.csv
# ==================================================================
comp = df.groupby('Наименование').agg(Отзывов=('Наименование', 'size'), Средняя_оценка=('Оценка автора', 'mean')).query('Отзывов >= 30')
comp = comp.round(2).reset_index().rename(columns={'Наименование': 'Компания'}).sort_values('Средняя_оценка')
comp_path = f'{args.outdir}/companies_top{args.suffix}.csv'
comp.to_csv(comp_path, index=False, encoding='utf-8-sig')
print(f"Сохранено: {comp_path} ({len(comp)} компаний с >=30 отзывов)")

print("\nГотово. Все файлы в", args.outdir)
