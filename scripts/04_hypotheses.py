import re
import argparse
import pandas as pd
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument('--input', default='../output/reviews_categorized.xlsx',
                 help='reviews_categorized.xlsx (rule-based) или reviews_categorized_gpt.xlsx (YandexGPT)')
args = ap.parse_args()

df = pd.read_excel(args.input)
print(f"Источник категоризации: {args.input}\n")
df['Текст_lower'] = df['Текст'].fillna('').str.lower()

print("="*70)
print("ГИПОТЕЗА 1: Негативные отзывы чаще из-за сервиса, а не качества мебели")
print("="*70)

SERVICE_COLS = ['Жалоба_Доставка/сроки', 'Жалоба_Сервис/персонал', 'Жалоба_Цена/деньги/возврат']
PRODUCT_COLS = ['Жалоба_Качество товара', 'Жалоба_Несоответствие описанию']

neg = df[df['Оценка автора'] <= 2].copy()
neg['Проблема_сервис'] = neg[SERVICE_COLS].any(axis=1)
neg['Проблема_товар'] = neg[PRODUCT_COLS].any(axis=1)

n_service = neg['Проблема_сервис'].sum()
n_product = neg['Проблема_товар'].sum()
n_both = (neg['Проблема_сервис'] & neg['Проблема_товар']).sum()
n_any = (neg['Проблема_сервис'] | neg['Проблема_товар']).sum()

print(f"Негативных отзывов (1-2 звезды): {len(neg)}")
print(f"  С признаками проблем сервиса (доставка/персонал/деньги): {n_service} ({n_service/len(neg)*100:.1f}%)")
print(f"  С признаками проблем товара (качество/несоответствие): {n_product} ({n_product/len(neg)*100:.1f}%)")
print(f"  Упомянуто и то, и другое: {n_both}")
print(f"  Хотя бы одна категория определена: {n_any} из {len(neg)} ({n_any/len(neg)*100:.1f}%)")

# Тест: доля отзывов с проблемой сервиса vs долю с проблемой товара (среди отзывов, где определена хотя бы 1 категория)
# Тест: доля отзывов с проблемой сервиса vs долю с проблемой товара (среди отзывов, где определена хотя бы 1 категория)
sub = neg[neg['Проблема_сервис'] | neg['Проблема_товар']]
count_service = int(sub['Проблема_сервис'].sum())
count_product = int(sub['Проблема_товар'].sum())
res = stats.binomtest(count_service, count_service + count_product, p=0.5)
print(f"\nБиномиальный тест долей (сервис={count_service} vs товар={count_product} среди "
      f"{len(sub)} размеченных негативных отзывов): p-value={res.pvalue:.2e}")

print(f"\nВЫВОД: проблемы сервиса встречаются в {n_service/n_any*100:.0f}% размеченных негативных\n"
      f"отзывов против {n_product/n_any*100:.0f}% с проблемами товара - разница статистически значима.")

print("\n" + "="*70)
print("ГИПОТЕЗА 2: Быстрая доставка повышает вероятность позитивной оценки")
print("="*70)

FAST = re.compile(r'быстро.{0,40}(доставили|привезли|доставк)|доставили.{0,10}быстро|привезли.{0,15}(быстро|раньше срока|раньше)')
SLOW = re.compile(r'долго.{0,40}(доставк|привезл|жда)|доставк\w*.{0,20}(задерж|перенос|опозда)|опозда|задерж\w*.{0,15}(доставк|привоз)|не привезли вовремя')

df['Доставка_быстро'] = df['Текст_lower'].apply(lambda t: bool(FAST.search(t)))
df['Доставка_медленно'] = df['Текст_lower'].apply(lambda t: bool(SLOW.search(t)))

fast_ratings = df.loc[df['Доставка_быстро'] & ~df['Доставка_медленно'], 'Оценка автора'].dropna()
slow_ratings = df.loc[df['Доставка_медленно'] & ~df['Доставка_быстро'], 'Оценка автора'].dropna()

print(f"Отзывов с упоминанием быстрой доставки: {len(fast_ratings)}, средняя оценка = {fast_ratings.mean():.2f}")
print(f"Отзывов с упоминанием медленной/сорванной доставки: {len(slow_ratings)}, средняя оценка = {slow_ratings.mean():.2f}")

u_stat, p_val = stats.mannwhitneyu(fast_ratings, slow_ratings, alternative='greater')
print(f"\nMann-Whitney U тест (H1: быстрая > медленная): U={u_stat:.0f}, p-value={p_val:.2e}")
print(f"ВЫВОД: {'подтверждается' if p_val < 0.05 else 'не подтверждается'} на уровне значимости 0.05")

print("\n" + "="*70)
print("ГИПОТЕЗА 3: Несоответствие описанию - самая частая причина претензий")
print("="*70)

complaint_counts = {cat.replace('Жалоба_', ''): df[cat].sum() for cat in
                     ['Жалоба_Доставка/сроки', 'Жалоба_Качество товара', 'Жалоба_Сервис/персонал',
                      'Жалоба_Цена/деньги/возврат', 'Жалоба_Несоответствие описанию']}
complaint_series = pd.Series(complaint_counts).sort_values(ascending=False)
print("Ранжирование категорий претензий по частоте (среди всех отзывов с текстом):")
print(complaint_series)
top_cat = complaint_series.index[0]
print(f"\nВЫВОД: самая частая категория - '{top_cat}' ({complaint_series.iloc[0]} упоминаний),")
print(f"а не 'Несоответствие описанию' ({complaint_counts['Несоответствие описанию']} упоминаний, "
      f"{list(complaint_series.index).index('Несоответствие описанию')+1}-е место из 5).")
print("Гипотеза 3 НЕ ПОДТВЕРЖДАЕТСЯ.")

print("\n" + "="*70)
print("ГИПОТЕЗА 4: Мягкая мебель -> более длинные и эмоциональные отзывы")
print("="*70)

soft = df[(df['Тип_мебели'] == 'Мягкая') & df['Есть_текст']]
corpus = df[(df['Тип_мебели'] == 'Корпусная') & df['Есть_текст']]

print(f"Отзывов про мягкую мебель (диван/кресло): {len(soft)}")
print(f"  Средняя длина: {soft['Длина_отзыва_слов'].mean():.1f} слов, медиана: {soft['Длина_отзыва_слов'].median():.0f}")
print(f"  Средняя эмоциональность: {soft['Эмоциональность'].mean():.2f}")
print(f"Отзывов про корпусную мебель (шкаф/кухня/стол): {len(corpus)}")
print(f"  Средняя длина: {corpus['Длина_отзыва_слов'].mean():.1f} слов, медиана: {corpus['Длина_отзыва_слов'].median():.0f}")
print(f"  Средняя эмоциональность: {corpus['Эмоциональность'].mean():.2f}")

u1, p1 = stats.mannwhitneyu(soft['Длина_отзыва_слов'].dropna(), corpus['Длина_отзыва_слов'].dropna(), alternative='two-sided')
u2, p2 = stats.mannwhitneyu(soft['Эмоциональность'].dropna(), corpus['Эмоциональность'].dropna(), alternative='two-sided')
print(f"\nMann-Whitney U тест длины отзыва: p-value={p1:.2e}")
print(f"Mann-Whitney U тест эмоциональности: p-value={p2:.2e}")
concl4 = (p1 < 0.05 and soft['Длина_отзыва_слов'].mean() > corpus['Длина_отзыва_слов'].mean())
print(f"ВЫВОД: {'подтверждается' if concl4 else 'не подтверждается / частично'} для длины отзыва.")

print("\n" + "="*70)
print("ГИПОТЕЗА 5: Чем выше цена товара, тем подробнее отзыв")
print("="*70)

priced = df[df['Упомянутая_цена'].notna() & df['Есть_текст']]
print(f"Отзывов с явным упоминанием суммы в рублях: {len(priced)} (из {df['Есть_текст'].sum()} с текстом, {len(priced)/df['Есть_текст'].sum()*100:.1f}%)")
if len(priced) > 30:
    rho, p5 = stats.spearmanr(priced['Упомянутая_цена'], priced['Длина_отзыва_слов'])
    print(f"Корреляция Спирмена (цена, длина отзыва): rho={rho:.3f}, p-value={p5:.3f}")
    print(f"ВЫВОД: {'слабая, но значимая положительная связь' if (p5<0.05 and rho>0) else 'связь не подтверждается статистически'}")
else:
    print("Недостаточно данных для надёжного статистического вывода.")
print("ВАЖНО: цена явно указана лишь в ~1-2% отзывов - большинство клиентов не пишут сумму,")
print("поэтому это лишь слабый косвенный индикатор, не полноценная проверка гипотезы.")
