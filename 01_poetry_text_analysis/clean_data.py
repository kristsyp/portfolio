import pandas as pd
import re
from bs4 import BeautifulSoup
import os
def clean_text(text):
"""
Очищает текст от HTML-тегов, лишних символов и пробелов.
Args:
 text (str): Входной текст.
Returns:
 str: Очищенный текст (пустая строка для нестроковых входов).
"""
if not isinstance(text, str):
 return ""
text = BeautifulSoup(text, "html.parser").get_text()
# Удаление управляющих символов (кроме \n)
text = re.sub(r'[\r\t]+', ' ', text)
# Нормализация пробелов (сохраняем переносы строк)
text = re.sub(r'[ ]{2,}', ' ', text) # Заменяем множественные пробелы
text = text.strip()
return text
def preprocess_data(df, min_length=None, max_length=None, text_quantile=0.05):
"""
Предобрабатывает данные: очистка текста, удаление дубликатов и пропущенных значений.
Args:
 df (pd.DataFrame): Исходный датасет. Обязательная колонка: 'Text'.
 min_length (int, optional): Минимальная длина текста. Автовычисление, если None.
 max_length (int, optional): Максимальная длина текста. Автовычисление, если None.
 text_quantile (float): Квантиль для автовычисления границ длины (по умолчанию 0.05).
Returns:
 pd.DataFrame: Предобработанный датасет.
 aises:
 ValueError: Если отсутствует колонка 'Text'.
"""
# Проверка наличия обязательных колонок
if 'Text' not in df.columns:
 raise ValueError("DataFrame должен содержать колонку 'Text'.")
print("Начало предобработки данных...")
# Очистка текста
print("Очистка текста...")
df['Text'] = df['Text'].apply(clean_text)
# Удаление пустых текстов
print("Удаление пустых текстов...")
initial_rows = len(df)
df = df[df['Text'] != ""]
removed_empty = initial_rows - len(df)
print(f"Удалено пустых текстов: {removed_empty}")
# Удаление пропущенных значений
print("Удаление строк с пропущенными значениями...")
initial_rows = len(df)
df = df.dropna(subset=['Text'])
removed_na = initial_rows - len(df)
print(f"Удалено строк с NaN: {removed_na}")
# Удаление дубликатов
print("Удаление дубликатов...")
initial_rows = len(df)
df = df.drop_duplicates(subset=['Text'])
removed_dup = initial_rows - len(df)
print(f"Удалено дубликатов: {removed_dup}")
# Проверка наличия данных после очистки
if df.empty:
 raise ValueError("После предобработки DataFrame пуст.")
# Добавление длины текста
print("Добавление длины текста...")
df['Text_length'] = df['Text'].str.len()
# Автоматическое определение границ длины
if min_length is None or max_length is None:
 q_low = text_quantile
 q_high = 1 - text_quantile
 min_length = int(df['Text_length'].quantile(q_low)) if min_length is None else min_length
 max_length = int(df['Text_length'].quantile(q_high)) if max_length is None else max_length
 print(f"Автовычисление границ: min_length={min_length}, max_length={max_length}")
# Фильтрация по длине
print(f"Фильтрация текстов ({min_length}-{max_length} символов)...")
initial_rows = len(df)
df = df[(df['Text_length'] >= min_length) & (df['Text_length'] <= max_length)]
removed_length = initial_rows - len(df)
print(f"Удалено текстов: {removed_length}")
# Удаление временной колонки (опционально)
# df = df.drop(columns=['Text_length'])
print(f"Предобработка завершена. Осталось строк: {len(df)}")
return df
def save_processed_data(df, output_path):
"""
Сохраняет предобработанные данные в CSV.
Args:
 df (pd.DataFrame): Данные для сохранения.
 output_path (str): Путь к файлу.
Raises:
 ValueError: Если данные пусты.
"""
if df.empty:
 raise ValueError("Нельзя сохранить пустой DataFrame.")
 os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False, encoding='utf-8')
print(f"Данные сохранены в {output_path}")
def main():
"""
Основная функция: загрузка, обработка и сохранение данных.
"""
try:
 # Для Google Colab
 from google.colab import drive
 drive.mount('/content/drive')
 base_path = '/content/drive/My Drive/vkr/'
except ImportError:
 base_path = './'
input_path = os.path.join(base_path, 'poems.csv')
output_path = os.path.join(base_path, 'poems_processed.csv')
try:
 print(f"Загрузка данных из {input_path}")
 df = pd.read_csv(input_path)
 print(f"Загружено строк: {len(df)}")
 print("Пример данных до обработки:\n", df.head(2))
except Exception as e:
 print(f"Ошибка загрузки: {e}")
 return
try:
 df_processed = preprocess_data(df)
 print("\nПример данных после обработки:\n", df_processed.head(2))
except Exception as e:
  rint(f"Ошибка обработки: {e}")
 return
try:
 save_processed_data(df_processed, output_path)
except Exception as e:
 print(f"Ошибка сохранения: {e}")
if __name__ == "__main__":
main()
