import scrapy
import logging
import json
import csv
import re
from scrapy.http import HtmlResponse
from poetry_parser.spiders.base_spider import BasePoetrySpider
from poetry_parser.utils.text_cleaner import clean_text, clean_poem_line # Импортируем функцию очистки
текста
logger = logging.getLogger(__name__)
class RustihSpider(BasePoetrySpider):
 name = 'rustih'
 def __init__(self, *args, **kwargs):
 super().__init__(site_name='rustih.ru', *args, **kwargs)
 self.start_urls = ['https://rustih.ru/spisok-poetov/']
 self.result_data = [] # Здесь будем хранить итоговые данные
def parse(self, response):
 logger.debug(f"Response URL: {response.url}")
 logger.debug(f"Response body (first 500 chars): {response.body[:500]}") # Для отладки, первые
500 символов
 # Извлекаем ссылки на страницы поэтов
 poet_links = response.css(self.site_config.selectors.poets_list).getall()
 logger.debug(f"Found {len(poet_links)} poet links")
 # count = 0 # Добавляем счетчик для ограничения количества поэтов
 for href in poet_links:
 # if count >= 5: # Ограничиваем обработку до 5 поэтов
 # break
 url = response.urljoin(href)
 logger.debug(f"Processing poet URL: {url}")
 yield scrapy.Request(url, callback=self.parse_author)
 # count += 1
 def parse_author(self, response):
 logger.debug(f"Poet page URL: {response.url}")
 # Извлекаем имя поэта
121
 poet_name = response.css(self.site_config.selectors.poet_name).get()
 if poet_name:
 # Очищаем имя поэта от лишнего текста
 poet_name = poet_name.split(':')[0].strip()
 logger.debug(f"Poet name: {poet_name}")
 else:
 logger.warning(f"Poet name not found on page: {response.url}")
 return
 # Извлекаем ссылки на стихотворения
 poem_links = response.css(self.site_config.selectors.poem_links).getall()
 if not poem_links:
 logger.warning(f"No poem links found for poet: {poet_name} on page: {response.url}")
 return
 logger.debug(f"Found {len(poem_links)} poem links for poet {poet_name}")
 # count = 0 # Добавляем счетчик для ограничения количества стихотворений
 for href in poem_links:
 # if count >= 5: # Ограничиваем обработку до 5 стихотворений
 # break
 url = response.urljoin(href)
 logger.debug(f"Processing poem URL: {url}")
 yield scrapy.Request(url, callback=self.parse_poem, cb_kwargs={"poet_name": poet_name})
 # count += 1
 def parse_poem(self, response, poet_name):
 logger.debug(f"Страница стихотворения: {response.url}")
 # Извлекаем название стихотворения
 title = response.css(self.site_config.selectors.poem_title).get()
 if title:
 title = title.strip()
 logger.debug(f"Название стихотворения: {title}")
 else:
 logger.warning(f"Название стихотворения не найдено на странице: {response.url}")
 return
 # Извлекаем весь текст
 full_text = response.body.decode('utf-8') # Явно указываем кодировку
 # Находим индекс заголовка анализа
 analysis_index = full_text.find('<h2>Анализ стихотворения')
 if analysis_index != -1:
 # Обрезаем текст до начала анализа
 poem_text = full_text[:analysis_index]
 else:
 poem_text = full_text
122
 # Создаем временный response для парсинга обрезанного текста
 limited_response = HtmlResponse(
 url=response.url,
 body=poem_text.encode('utf-8'),
 encoding='utf-8'
 )
 # Извлекаем строфы из обрезанного текста
 stanzas = limited_response.css(self.site_config.selectors.poem_stanzas).getall()
 if not stanzas:
 logger.warning(f"Текст стихотворения не найден на странице: {response.url}")
 return
 grouped_lines = []
 for stanza in stanzas:
 # Разделяем строфу на строки по тегу <br>
 lines = re.split(r'<br\s*/?>', stanza)
 cleaned_lines = []
 for line in lines:
 cleaned_line = clean_poem_line(line)
 if cleaned_line and self.is_poetry_line(cleaned_line): # Проверяем, что строка похожа
на поэзию
 cleaned_lines.append(cleaned_line)
 if cleaned_lines:
 grouped_lines.append(cleaned_lines)
 # Удаляем прозу после стихотворения
 grouped_lines = self.remove_prose_after_poem(grouped_lines)
 # Сохраняем данные о стихотворении
 self.result_data.append({
 "author": poet_name,
 "title": title,
 "text": grouped_lines, # Список строф
 "url": response.url
 })
def closed(self, reason):
 # Сохраняем данные в CSV при завершении работы паука
 with open('poems.csv', 'w', encoding='utf-8', newline='') as f:
 writer = csv.writer(f)
 writer.writerow(['Author', 'Title', 'Text', 'URL']) # Заголовки столбцов
 for poem in self.result_data:
 formatted_text = '\n\n'.join(['\n'.join(stanza) for stanza in poem["text"]])
 writer.writerow([poem['author'], poem['title'], formatted_text, poem['url']])
 logger.info(f"Saved {len(self.result_data)} poems to poems.csv")
123
 def is_poetry_line(self, line: str) -> bool:
 line = line.strip()
 if not line:
 return False
 if len(line) > 90: # Длинные строки — скорее всего, проза
 return False
 if len(line.split()) <= 1: # Слишком короткие строки игнорируем
 return False
 return True
 def remove_prose_after_poem(self, grouped_lines: list) -> list:
 cleaned_grouped_lines = []
 for stanza in grouped_lines:
 if all(len(line) > 150 for line in stanza): # Если строфа выглядит как проза
 break
 cleaned_grouped_lines.append(stanza)
 return cleaned_grouped_lines
