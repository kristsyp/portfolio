from rhymetagger import RhymeTagger
import nltk
import pymorphy3
import re
from ruaccent import RUAccent
import numpy as np
import rapidfuzz
from rapidfuzz import fuzz
from collections import Counter
import Levenshtein
import ast
import torch
import torch.nn as nn
import json
import pandas as pd

nltk.download('punkt_tab')
morph = pymorphy3.MorphAnalyzer()
accentizer = RUAccent()
accentizer.load(
    omograph_model_size='turbo3.1',
    use_dictionary=True,
    tiny_mode=False,
)
rt = RhymeTagger()
rt.load_model(model='ru')  # ru значит русский язык
print(rt)
rhyme_scheme_mapping = {
    # специальные формы
    "Монорифма (AAAA...)": 0,
    "Онегинская строфа (AbAbCCddEffEGG)": 1,
    "Шекспировский сонет (ABAB CDCD EFEF GG)": 2,
    "Итальянский сонет (ABBA ABBA CDC DCD или CDE CDE)": 3,
    "Французский сонет (ABBA ABBA CCD EED или CCD EDE)": 4,
    "Лимерик (AABBA)": 5,
    "Рубаи (AABA)": 6,
    "Терцины (ABA BCB CDC...)": 7,
    "Триолет (ABaa abAB)": 8,
    "Квадратная рифмовка (ABCD ABCD)": 9,
    "Треугольная рифмовка (ABC ABC)": 10,

    # классические паттерны
    "Парная (AABB)": 11,
    "Перекрестная (ABAB)": 12,
    "Охватная (ABBA)": 13,
    "Парное двухстишие (AA)": 14,
    "Смежное двухстишие (AB)": 15
}


def has_rhyme(rhyme_tags):
    """
    Проверяет наличие рифмы в списке тегов.
    """

    if not rhyme_tags:
        return False

    # Игнорируем None при подсчете рифм
    filtered_tags = [tag for tag in rhyme_tags if tag is not None]

    if len(filtered_tags) < 2:
        return False

    # Теперь продолжаем обработку
    unique_tags = set(filtered_tags)

    for tag in unique_tags:
        if filtered_tags.count(tag) > 1:
            return True

    return False


def check_rhyme_scheme(rhyme_tags):
    n = len(rhyme_tags)
    schemes = []

    def check_aabb(tags):
        """Проверяет парную рифмовку AABB, где A != B"""
        return (len(tags) >= 4
                and tags[0] == tags[1]
                and tags[2] == tags[3]
                and tags[0] != tags[2])

    def check_abab(tags):
        """Проверяет перекрестную рифмовку ABAB, где A != B"""
        return (len(tags) >= 4
                and tags[0] == tags[2]
                and tags[1] == tags[3]
                and tags[0] != tags[1])

    def check_abba(tags):
        """Проверяет охватную рифмовку ABBA, где A != B"""
        return (len(tags) >= 4
                and tags[0] == tags[3]
                and tags[1] == tags[2]
                and tags[0] != tags[1])

    def check_sonnet_uniqueness(tags, octave_end):
        """Проверяет, что рифмы октавы и секстета не пересекаются"""
        octave_rhymes = set(tags[:octave_end])
        sextet_rhymes = set(tags[octave_end:])
        return octave_rhymes.isdisjoint(sextet_rhymes)

    def check_onegin_stanza(tags):  # ABAB CCDD EFFE GG - 14 строк
        if not check_sonnet_uniqueness(tags, 8):
            return False
        return len(tags) == 14 and (
                check_abab(tags[0:4]) and
                check_aabb(tags[4:8]) and
                check_abba(tags[8:12]) and
                tags[12] == tags[13]
        )

    def check_shakespearean_sonnet(tags):  # АВAB СDCD EFEF GG - 14 строк
        if not check_sonnet_uniqueness(tags, 8):
            return False
        return len(tags) == 14 and (
                check_abab(tags[0:4]) and
                check_abab(tags[4:8]) and
                check_abab(tags[8:12]) and
                tags[12] == tags[13]
        )

    # Проверка на монорифму (все рифмы одинаковые)
    def check_monorhyme(tags):
        """Проверяет монорифму: все рифмы одинаковы и не None"""
        if not tags:  # Пустой список
            return False
        if None in tags:  # Есть хотя бы один None
            return False
        return len(set(tags)) == 1  # Все элементы одинаковы

    def check_italian_sonnet(tags):  # АВAB АВAB СDC DCD или CDE CDE
        if len(tags) != 14:
            return False

        # Проверяем первые две строфы (ABBA ABBA)
        if not (check_abab(tags[0:4]) and check_abab(tags[4:8])):
            return False

        if not check_sonnet_uniqueness(tags, 8):
            return False

        # Проверяем возможные варианты для последних 6 строк
        # Вариант 1: CDC DCD
        variant1 = (tags[8] == tags[10] and  # C
                    tags[9] == tags[11] == tags[13] and  # D
                    tags[12] == tags[8] and  # C
                    tags[8] != tags[9])

        # Вариант 2: CDE CDE
        variant2 = (tags[8] == tags[11] and  # C
                    tags[9] == tags[12] and  # D
                    tags[10] == tags[13] and  # E
                    tags[8] != tags[9] and  # C != D
                    tags[9] != tags[10] and  # D != E
                    tags[8] != tags[10])

        return variant1 or variant2

    def check_french_sonnet(tags):
        if len(tags) != 14:
            return False

        # Проверяем первые 8 строк (ABBA ABBA)
        if not (check_abba(tags[0:4]) and check_abba(tags[4:8])):
            return False

        if not check_sonnet_uniqueness(tags, 8):
            return False

        # Вариант 1: CCD EED
        variant1 = (
                tags[8] == tags[9] and  # CC
                tags[11] == tags[12] and  # EE
                tags[10] == tags[13] and  # DD
                tags[8] != tags[11] and  # C != D
                tags[11] != tags[10] and  # D != E
                tags[8] != tags[13])

        # Вариант 2: CCD EDE
        variant2 = (
                tags[8] == tags[9] and  # CC
                tags[10] == tags[12] and  # E E (10 и 12 строки)
                tags[11] == tags[13] and  # D D (11 и 13 строки)
                tags[8] != tags[11] and  # C != D
                tags[13] != tags[12] and  # D != E
                tags[8] != tags[10])

        return variant1 or variant2

    def check_six_lines(tags):
        if len(tags) != 6:
            return False

        # Создаем список рифм для каждой строки
        rhymes = [[] for _ in range(6)]
        for i in range(6):
            for j in range(6):
                if i != j and tags[i] == tags[j]:
                    rhymes[i].append(j)

        # Проверяем, что у каждой строки есть хотя бы одна рифма
        for r in rhymes:
            if not r:  # Если список рифм для строки пуст
                return False
        return True

    # Новая функция для проверки двухстиший
    def check_couplet(tags):  # AA или AB
        if len(tags) < 2:
            return False
        return tags[0] == tags[1]  # AA (парное двухстишие)

    def check_alternate_couplet(tags):  # AB (смежное двухстишие)
        if len(tags) < 2:
            return False
        return tags[0] != tags[1]  # AB

    def check_limerick(tags):
        if len(tags) != 5:
            return False

        # Проверяем схему AABBA
        return (tags[0] == tags[1] and  # AA
                tags[2] == tags[3] and  # BB
                tags[0] == tags[4] and  # A (возвращение к первой рифме)
                tags[0] != tags[2])

    def check_rubai(tags):
        if len(tags) != 4:
            return False

        # Проверяем схему AABA
        return (tags[0] == tags[1] and  # AA
                tags[1] != tags[2] and  # B (новая рифма)
                tags[0] == tags[3])  # A (возврат к основной рифме)

    def is_aba(tags, start_pos):
        """Проверяет, является ли последовательность ABA, начиная с start_pos"""
        if start_pos + 2 >= len(tags):
            return False, None

        a = tags[start_pos]
        b = tags[start_pos + 1]
        a2 = tags[start_pos + 2]

        return (a == a2 and a != b), b

    def check_tercina(tags):
        """Проверяет цепочку терцин: ABA → BCB → CDC → DCD..."""
        if len(tags) < 6 or len(tags) % 3 != 0:  # Минимум 2 терцины и длина кратна 3
            return False

        # Проверяем, что ПЕРВАЯ терцина начинается сразу (ABA)
        is_first, prev_b = is_aba(tags, 0)
        if not is_first:
            return False

        # Проверяем последующие терцины строго по порядку
        current_pos = 3  # Начинаем с 4-го элемента (после первой терцины)
        while current_pos < len(tags):
            # Следующая терцина должна начинаться с prev_b
            if tags[current_pos] != prev_b:
                return False

            # Проверяем следующую терцину (BCB)
            is_valid, new_b = is_aba(tags, current_pos)
            if not is_valid:
                return False

            prev_b = new_b
            current_pos += 3  # Переходим к следующей терцине

        return True

    def check_triolet(tags):
        """Проверяет схему триолета ABAA ABAB"""
        if len(tags) != 8:
            return False

        # Проверяем точное соответствие схеме ABAA ABAB
        return (tags[0] == tags[2] == tags[3] == tags[4] == tags[6] and  # Все A
                tags[1] == tags[5] == tags[7] and  # Все B
                tags[0] != tags[1])

    def check_abcd_abcd(tags):
        """Проверяет рифмовку 4+4 (ABCD ABCD)"""
        if len(tags) < 8 or len(tags) % 8 != 0:
            return False

        for i in range(0, len(tags), 8):
            # Берем две строфы по 4 строки
            stanza1 = tags[i:i + 4]
            stanza2 = tags[i + 4:i + 8]

            # Проверяем, что строфы идентичны по рифмовке
            if stanza1 != stanza2:
                return False

            # Проверяем, что в каждой строфе все рифмы разные
            if len(set(stanza1)) != 4:
                return False

        return True

    def check_abc_abc(tags):
        """Проверяет рифмовку 3+3 (ABC ABC)"""
        if len(tags) < 6 or len(tags) % 6 != 0:
            return False

        for i in range(0, len(tags), 6):
            # Берем два трёхстишия
            triplet1 = tags[i:i + 3]
            triplet2 = tags[i + 3:i + 6]

            # Проверяем, что трёхстишия идентичны по рифмовке
            if triplet1 != triplet2:
                return False

            # Проверяем, что в каждом трёхстишии все рифмы разные
            if len(set(triplet1)) != 3:
                return False

        return True

    if check_abcd_abcd(rhyme_tags):
        return ["Квадратная рифмовка (ABCD ABCD)"]
    elif check_abc_abc(rhyme_tags):  # Новая проверка 3+3
        return ["Треугольная рифмовка (ABC ABC)"]
    # Проверка специальных форм
    elif check_monorhyme(rhyme_tags):
        return ["Монорифма (AAAA...)"]
    elif check_onegin_stanza(rhyme_tags):
        return ["Онегинская строфа (AbAbCCddEffEGG)"]
    elif check_shakespearean_sonnet(rhyme_tags):
        return ["Шекспировский сонет (ABAB CDCD EFEF GG)"]
    elif check_italian_sonnet(rhyme_tags):
        return ["Итальянский сонет (ABBA ABBA CDC DCD или ABBA ABBA CDE CDE)"]
    elif check_french_sonnet(rhyme_tags):
        return ["Французский сонет (ABBA ABBA CCD EED или CCD EDE)"]
    elif check_limerick(rhyme_tags):
        return ["Лимерик (AABBA)"]
    elif check_rubai(rhyme_tags):
        return ["Рубаи (AABA)"]
    elif check_tercina(rhyme_tags):
        return ["Терцины (ABA BCB CDC...)"]
    elif check_triolet(rhyme_tags):
        return ["Триолет (ABaa abAB)"]
    elif check_six_lines(rhyme_tags):
        return ["Шестистишие (произвольная схема)"]

    # Проверка четверостиший, двухстиший и других фрагментов
    i = 0
    while i < n:
        remaining = n - i

        if remaining >= 4:
            # Проверка четверостиший
            quartet = rhyme_tags[i:i + 4]
            if check_monorhyme(quartet):  # Монорифма в четверостишии
                schemes.append("Монорифма (AAAA)")
            elif check_aabb(quartet):
                schemes.append("Парная (AABB)")
            elif check_abab(quartet):
                schemes.append("Перекрестная (ABAB)")
            elif check_abba(quartet):
                schemes.append("Охватная (ABBA)")
            else:
                schemes.append("Неопределенная")
            i += 4
        elif remaining >= 2:
            # Проверка двухстиший
            couplet = rhyme_tags[i:i + 2]
            if check_couplet(couplet):
                schemes.append("Парное двухстишие (AA)")
            elif check_alternate_couplet(couplet):
                schemes.append("Смежное двухстишие (AB)")
            else:
                schemes.append("Неопределенная")
            i += 2
        else:
            # Оставшиеся строки
            schemes.append("Неопределенная")
            i += remaining

    return schemes


def split_into_syllables(word):
    """
    Разбивает слово на слоги с учетом правил русского языка.
    Учитывает односложные слова (например, "в"), междометия (например, "хм", "тсс", "бррр")
    и звукоподражательные слова (например, "мрр", "хррр").
    """
    # Определяем гласные буквы русского алфавита
    vowels = set('аеёиоуыэюяАЕЁИОУЫЭЮЯ')

    # Удаляем символ ударения, если он есть
    word = word.replace('+', '')

    # Проверяем, является ли слово односложным (например, "в") или не содержит гласных
    if len(word) == 1 or not any(char.lower() in vowels for char in word):
        return [word]  # Возвращаем слово как единственный слог

    # Разбиваем слово на части, где каждая часть начинается с гласной
    syllables = []
    current_syllable = ''
    for char in word:
        current_syllable += char
        if char.lower() in vowels:
            syllables.append(current_syllable)
            current_syllable = ''
    if current_syllable:  # Если остались символы, добавляем их к последнему слогу
        syllables[-1] += current_syllable

    return syllables


def split_into_syllables(word):
    """
    Разбивает слово на слоги. Учитывает односложные и междометия.
    """
    vowels = set('аеёиоуыэюяАЕЁИОУЫЭЮЯ')
    word = word.replace('+', '')
    word = re.sub(r'[^а-яА-ЯёЁ]', '', word)

    if not word:
        return []

    if len(word) == 1 or not any(char.lower() in vowels for char in word):
        return [word]

    syllables = []
    current_syllable = ''
    for char in word:
        current_syllable += char
        if char.lower() in vowels:
            syllables.append(current_syllable)
            current_syllable = ''
    if current_syllable:
        if syllables:
            syllables[-1] += current_syllable
        else:
            syllables.append(current_syllable)

    return syllables


def get_word_stress(word):
    """
    Определяет, какой слог ударный. Возвращает индекс ударного слога.
    """
    syllables = split_into_syllables(word)

    # Если нет гласных — ударения нет
    if not syllables:
        return None

    # Исключаем служебные части речи
    parsed = morph.parse(word)[0]
    if any(tag in parsed.tag for tag in {'PREP', 'CONJ', 'PRCL', 'INTJ', 'NPRO'}):
        return None

    try:
        stressed = accentizer.process_all(word)
        if '+' in stressed:
            stress_pos = stressed.index('+')
            pos = 0
            for i, syl in enumerate(syllables):
                pos += len(syl)
                if stress_pos < pos:
                    return i
        else:
            # Если слово односложное и ударение не указано, считаем его ударным
            if len(syllables) == 1:
                return 0
    except Exception as e:
        # В случае ошибки — считаем односложное слово ударным
        if len(syllables) == 1:
            return 0

    return None


def build_stress_scheme(line):
    """
    Строит схему ударений строки: 'U' — ударный, 'u' — безударный слог.
    """
    words = re.findall(r'\b[а-яёА-ЯЁ]+\b', line)
    scheme = []

    for word in words:
        syllables = split_into_syllables(word)
        if not syllables:
            continue  # Пропускаем слова без слогов

        stress_pos = get_word_stress(word)

        for i in range(len(syllables)):
            scheme.append('U' if i == stress_pos else 'u')

    return ''.join(scheme)


def analyze_poem(text):
    """
    Анализирует текст стихотворения и возвращает частоту схем ударений.
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    schemes = [build_stress_scheme(line) for line in lines]
    return Counter(filter(None, schemes))


def len_poem(text):
    """
    Анализирует текст стихотворения и возвращает частоту схем ударений.
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return len(lines)


def get_most_common_syllable_count(pattern_counter):
    """
    Определяет самое частое количество слогов в строках стихотворения.
    Возвращает int (например, 8) или None, если данные пусты.
    """
    if not pattern_counter:
        return None

    # Считаем длину каждого паттерна (1 символ = 1 слог)
    syllable_counts = []
    for pattern, count in pattern_counter.items():
        length = len(pattern)  # Кол-во слогов в строке
        syllable_counts.extend([length] * count)  # Учитываем частоту паттерна

    if not syllable_counts:
        return None

    # Находим самое частое количество слогов
    most_common = Counter(syllable_counts).most_common(1)[0][0]
    return most_common


def normalize_schemes(counter, threshold=0.85, min_count=2):
    """
    Объединяет редкие метрические схемы с похожими более частыми.

    Параметры:
        counter (Counter): Счётчик схем
        threshold (float): Порог схожести (0-1)
        min_count (int): Порог частоты — ниже этого считаются редкими

    Возвращает:
        Counter: Нормализованный счётчик
        str: Эталонная схема (самая частая после нормализации)
    """
    if not counter:
        return Counter(), ""

    # 1. Отсортировать схемы по убыванию частоты
    sorted_schemes = sorted(counter.items(), key=lambda x: -x[1])

    # 2. Список уже обработанных схем
    used = set()
    new_counter = Counter()

    for i, (scheme, count) in enumerate(sorted_schemes):
        if scheme in used:
            continue

        # Если схема достаточно частая — оставляем как есть
        if count >= min_count:
            new_counter[scheme] += count
            used.add(scheme)

            # Ищем и объединяем похожие редкие схемы
            for other_scheme, other_count in sorted_schemes[i + 1:]:
                if other_scheme in used:
                    continue
                similarity = fuzz.ratio(scheme, other_scheme) / 100
                if similarity >= threshold:
                    new_counter[scheme] += counter[other_scheme]
                    used.add(other_scheme)
        else:
            # Если схема редкая и похожа на уже добавленную — объединяем
            best_match = None
            best_score = 0
            for target in new_counter:
                similarity = fuzz.ratio(scheme, target) / 100
                if similarity > best_score and similarity >= threshold:
                    best_score = similarity
                    best_match = target
            if best_match:
                new_counter[best_match] += count
                used.add(scheme)
            else:
                # Если нет похожей — всё равно добавим отдельно
                new_counter[scheme] += count
                used.add(scheme)

    # 3. Эталонная схема — самая частая после нормализации
    most_common_scheme = new_counter.most_common(1)[0][0] if new_counter else ""

    return new_counter, most_common_scheme


def levenshtein_distance(s1, s2):
    """Вычисляет расстояние Левенштейна между строками"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def process_dataframe(df, input_col="Meter Frequency", output_col="Нормализованные_схемы"):
    """Обрабатывает весь датафрейм без прогресс-бара"""

    # Создаем временные колонки
    temp_results = []
    temp_refs = []

    for counter in df[input_col]:
        if isinstance(counter, str):
            try:
                counter = eval(counter)
            except:
                counter = Counter()

        norm_counter, ref_scheme = normalize_schemes(counter)
        temp_results.append(norm_counter)
        temp_refs.append(ref_scheme)

    # Добавляем новые колонки, сохраняя исходные данные
    df = df.copy()
    df[output_col] = temp_results
    df["Эталонная_схема"] = temp_refs

    return df


def detect_rhythm(counter):
    """
    Проверяет наличие рифмы по частотному распределению схем.
    Возвращает True, если менее 20% схем имеют частоту 1 или 2.
    """
    if isinstance(counter, str):
        try:
            counter = eval(counter)
        except:
            print("Ошибка преобразования строки в Counter")
            return None

    if not isinstance(counter, dict) and not isinstance(counter, Counter):
        print("Неверный тип данных")
        return None

    total_schemes = sum(counter.values())

    if total_schemes == 0:
        return None

    rare_count = sum(count for count in counter.values() if count <= 1)
    rare_ratio = rare_count / total_schemes
    # print("Доля редких схем:", rare_ratio)

    result = rare_ratio < 0.4
    return result


def count_stress_counter(counter_obj):
    if isinstance(counter_obj, str):
        try:
            counter_obj = eval(counter_obj)
        except:
            return 0, 0
    if not isinstance(counter_obj, dict):
        return 0, 0
    total_U, total_non_U = 0, 0
    for scheme, freq in counter_obj.items():
        u = scheme.count('U') * freq
        total = len(scheme) * freq
        total_U += u
        total_non_U += (total - u)
    return total_U, total_non_U


def generate_etalon_scheme(meter, length):
    """
    Возвращает эталонную ритмическую схему для заданного метра и длины строки.
    """
    meter = meter.lower()
    schemes = {
        "ямб": lambda l: ("uU" * ((l + 1) // 2))[:l],
        "хорей": lambda l: ("Uu" * ((l + 1) // 2))[:l],
        "дактиль": lambda l: ("Uuu" * ((l + 2) // 3))[:l],
        "анапест": lambda l: ("uuU" * ((l + 2) // 3))[:l],
        "амфибрахий": lambda l: ("uUu" * ((l + 2) // 3))[:l],
    }
    return schemes.get(meter, lambda l: "")(length)


# Список всех возможных размеров
all_meters = ["ямб", "хорей", "дактиль", "анапест", "амфибрахий"]


def levenshtein_similarity(s1, s2):
    """Вычисление сходства на основе расстояния Левенштейна"""
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1 - Levenshtein.distance(s1, s2) / max_len


def parse_schemes(scheme_data):
    """Преобразуем данные схем в Counter объект"""
    if isinstance(scheme_data, Counter):
        return scheme_data
    elif isinstance(scheme_data, str):
        try:
            # Если строка начинается с "Counter", извлекаем внутренний словарь
            if scheme_data.startswith("Counter("):
                # Удаляем "Counter(" в начале и ")" в конце
                dict_str = scheme_data[8:-1]
                # Преобразуем строку словаря в настоящий словарь
                scheme_dict = ast.literal_eval(dict_str)
                return Counter(scheme_dict)
            else:
                # Если это просто строка схемы (не Counter)
                return Counter([scheme_data])
        except Exception as e:
            print(f"Ошибка при разборе строки: {scheme_data}\nОшибка: {e}")
            return Counter()
    elif isinstance(scheme_data, dict):
        return Counter(scheme_data)
    else:
        return Counter()


def calculate_all_meter_similarities(row):
    """
    Вычисление сходств частотных схем с эталонами всех 5 размеров.
    Возвращает 15 колонок:
        - Сходство с топ-1 схемой: по 5 метрам
        - Сходство с топ-2 схемой: по 5 метрам
        - Среднее сходство: по 5 метрам
    """
    try:
        counter = parse_schemes(row['Нормализованные_схемы'])
        if not counter:
            return pd.Series([np.nan] * 15, index=[
                f'{sim_type}_{meter}' for sim_type in [
                    'Сходство_с_топ1', 'Сходство_с_топ2', 'Среднее_сходство'
                ] for meter in ['ямб', 'хорей', 'дактиль', 'анапест', 'амфибрахий']
            ])

        most_common = counter.most_common(2)
        freq1 = most_common[0][0] if len(most_common) > 0 else ''
        freq2 = most_common[1][0] if len(most_common) > 1 else ''

        all_meters = ['ямб', 'хорей', 'дактиль', 'анапест', 'амфибрахий']
        results = []

        for meter in all_meters:
            etalon = row.get(f'Эталонная_схема_{meter}', '')
            if not etalon or pd.isna(etalon):
                results.extend([np.nan, np.nan, np.nan])
                continue

            # Топ-1, Топ-2
            sim_freq1 = levenshtein_similarity(freq1, etalon)
            sim_freq2 = levenshtein_similarity(freq2, etalon)

            # Среднее по всем
            total = sum(counter.values())
            sim_avg = 0.0
            for scheme, count in counter.items():
                sim_avg += (count / total) * levenshtein_similarity(scheme, etalon)

            results.extend([sim_freq1, sim_freq2, sim_avg])

        columns = [
            f'{sim_type}_{meter}'
            for sim_type in ['Сходство_с_топ1', 'Сходство_с_топ2', 'Среднее_сходство']
            for meter in all_meters
        ]

        return pd.Series(results, index=columns)

    except Exception as e:
        print(f"Ошибка при обработке строки: {e}")
        return pd.Series([np.nan] * 15, index=[
            f'{sim_type}_{meter}' for sim_type in [
                'Сходство_с_топ1', 'Сходство_с_топ2', 'Среднее_сходство'
            ] for meter in ['ямб', 'хорей', 'дактиль', 'анапест', 'амфибрахий']
        ])


top_20_features = [
    'Сходство_с_топ1_дактиль',
    'Среднее_сходство_амфибрахий',
    'Сходство_с_топ2_амфибрахий',
    'Среднее_сходство_дактиль',
    'Сходство_с_топ2_анапест',
    'Среднее_сходство_хорей',
    'Сходство_с_топ1_ямб',
    'Сходство_с_топ2_хорей',
    'Сходство_с_топ2_ямб',
    'Среднее_сходство_анапест',
    'Сходство_с_топ1_анапест',
    'Сходство_с_топ1_хорей',
    'Сходство_с_топ1_амфибрахий',
    'Преобладающая_длина_строки_dict',
    'Среднее_сходство_ямб',
    'Доля_ударных_dict',
    'Число_безударных_dict',
    'Число_ударных_dict',
    'Сходство_с_топ2_дактиль',
    'Number of Lines'
]
class_to_meter = {
    0: "амфибрахий",
    1: "анапест",
    2: "дактиль",
    3: "хорей",
    4: "ямб"
}

# Загрузка метаданных
with open("saved_model/meta.json", "r") as f:
    metadata = json.load(f)

input_size = metadata["input_size"]
num_classes = metadata["num_classes"]


class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Первый скрытый слой
        self.fc2 = nn.Linear(64, 32)  # Второй скрытый слой
        self.fc3 = nn.Linear(32, num_classes)  # Выходной слой (для многоклассовой классификации)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Для многоклассовой классификации

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x  # Возвращаем не софтмакс, потому что CrossEntropyLoss сам применяет softmax


# Инициализация и загрузка модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN(input_size, num_classes).to(device)
model.load_state_dict(torch.load("saved_model/model.pth", map_location=device))
model.eval()


def detect_rhythm_from_text(text):
    df = pd.DataFrame({
        'text': [text],
        'Meter Frequency': [analyze_poem(text)],
        'Number of Lines': [len_poem(text)],
        'Преобладающая_длина_строки_dict': [get_most_common_syllable_count(analyze_poem(text))],
    })
    df = process_dataframe(df)
    df["Есть_ритм_по_схеме"] = df["Нормализованные_схемы"].apply(detect_rhythm)
    return bool(df["Есть_ритм_по_схеме"].iloc[0])


def get_meter_similarity_score(text):
    df = pd.DataFrame({
        'text': [text],
        'Meter Frequency': [analyze_poem(text)],
        'Number of Lines': [len_poem(text)],
        'Преобладающая_длина_строки_dict': [get_most_common_syllable_count(analyze_poem(text))],
    })
    df = process_dataframe(df)
    df["Есть_ритм_по_схеме"] = df["Нормализованные_схемы"].apply(detect_rhythm)
    if not df["Есть_ритм_по_схеме"].iloc[0]:
        return 0.0

    df[['Число_ударных_dict', 'Число_безударных_dict']] = df['Нормализованные_схемы'].map(count_stress_counter).apply(
        pd.Series)
    df['Доля_ударных_dict'] = df['Число_ударных_dict'] / (df['Число_ударных_dict'] + df['Число_безударных_dict'])
    for meter in all_meters:
        df[f"Эталонная_схема_{meter}"] = df["Преобладающая_длина_строки_dict"].apply(
            lambda l: generate_etalon_scheme(meter, l))
    df_main_sim = df.apply(calculate_all_meter_similarities, axis=1)
    df = pd.concat([df, df_main_sim], axis=1)

    X_input = df[top_20_features]
    X_tensor = torch.tensor(X_input.values, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(X_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()

    max_prob = float(probabilities[0].max())
    return round(max_prob, 4)


def lexical_diversity_score(text):
    tokens = text.split()
    if len(tokens) == 0:
        return 0.0
    return len(set(tokens)) / len(tokens)


from collections import Counter
import re


def compute_reward(text: str) -> float:
    # Логируем полученный текст
    print(f"\n[compute_reward] Input text:\n{text}\n")

    # 1. Ритм (True / False)
    rhythm = detect_rhythm_from_text(text)
    rhythm_score = 1.0 if rhythm else 0.0
    print(f"  rhythm: {rhythm} → rhythm_score = {rhythm_score:.3f}")

    # 2. Метр — вероятность самого вероятного размера
    meter_score = get_meter_similarity_score(text)
    print(f"  meter_score = {meter_score:.3f}")

    # 3. Рифма
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    tags = rt.tag([lines], output_format=3)
    print(tags)
    rhyme = has_rhyme(tags)
    rhyme_score = 1.0 if rhyme else 0.0
    print(f"  rhyme: {rhyme} → rhyme_score = {rhyme_score:.3f}")

    # 4. Шаблон рифмы
    schemes = check_rhyme_scheme(tags)
    scheme_score = 1.0 if any(s in rhyme_scheme_mapping for s in schemes) else 0.0
    print(f"  schemes found = {schemes} → scheme_score = {scheme_score:.3f}")

    # 5. Лексическая оценка (разнообразие + штраф за повтор)
    words = [w.lower() for w in text.split()]
    word_counts = Counter(words)
    if words:
        diversity = len(set(words)) / len(words)
        repetition_penalty = word_counts.most_common(1)[0][1] / len(words)
        lex_score = diversity * (1.0 - repetition_penalty)
    else:
        diversity = 0.0
        repetition_penalty = 0.0
        lex_score = 0.0
    print(f"  diversity = {diversity:.3f}, repetition_penalty = {repetition_penalty:.3f} → lex_score = {lex_score:.3f}")

    # 6. Базовый подсчёт награды
    weights = {
        "rhythm": 1.2,
        "meter": 1.0,
        "rhyme": 1.9,
        "scheme": 0.5,
        "lex": 0.7,
    }
    weighted_sum = (
            weights["rhythm"] * rhythm_score +
            weights["meter"] * meter_score +
            weights["rhyme"] * rhyme_score +
            weights["scheme"] * scheme_score +
            weights["lex"] * lex_score
    )
    total_weight = sum(weights.values())
    base_score = weighted_sum / total_weight

    return base_score
