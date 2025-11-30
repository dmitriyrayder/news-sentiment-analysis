from fastapi import FastAPI
from transformers import pipeline
import psycopg2
import os
import re
from collections import Counter
import json

app = FastAPI(title="Advanced Sentiment Analysis API")

# Загрузка моделей
print("Loading models...")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
)

# Модель для определения эмоций (английский)
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

# Zero-shot классификация для категорий
zero_shot = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)
print("Models loaded!")

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'sentiment_db'),
    'user': os.getenv('DB_USER', 'sentiment_user'),
    'password': os.getenv('DB_PASSWORD', 'password')
}

# Категории новостей
CATEGORIES = [
    "war/conflict",
    "politics", 
    "economy",
    "technology",
    "sports",
    "health",
    "environment",
    "crime",
    "entertainment"
]

# Ключевые слова для детекции
FEAR_KEYWORDS = {
    'en': ['war', 'attack', 'threat', 'crisis', 'danger', 'terror', 'death', 'disaster', 'pandemic', 'crash'],
    'ru': ['война', 'атака', 'угроза', 'кризис', 'опасность', 'террор', 'смерть', 'катастрофа', 'пандемия'],
    'uk': ['війна', 'атака', 'загроза', 'криза', 'небезпека', 'терор', 'смерть', 'катастрофа', 'пандемія']
}

CLICKBAIT_PATTERNS = [
    r'вы не поверите',
    r'шокирующ',
    r'сенсация',
    r'срочно',
    r'breaking',
    r'shocking',
    r'you won\'t believe'
]

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def extract_keywords(text, lang, top_n=5):
    """Извлечение ключевых слов"""
    # Простая версия - частотный анализ
    words = re.findall(r'\b\w{4,}\b', text.lower())
    
    # Стоп-слова (упрощенно)
    stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'said', 
                  'они', 'был', 'это', 'как', 'для', 'все', 'был', 'она'}
    
    words = [w for w in words if w not in stop_words]
    common = Counter(words).most_common(top_n)
    return [word for word, count in common]

def extract_entities(text):
    """Извлечение сущностей (упрощенная версия)"""
    # Страны (паттерны заглавных букв)
    countries = re.findall(r'\b[A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ][a-zа-яё]+)?\b', text)
    
    # Фильтр: только известные страны/города
    known_entities = ['Ukraine', 'Russia', 'USA', 'China', 'Europe', 'Україна', 'Росія', 'США']
    entities = [e for e in countries if e in known_entities or len(e) > 6]
    
    return {
        'countries': list(set(entities[:5])),
        'people': [],  # Требует NER модели
        'companies': []
    }

def calculate_importance(text, sentiment, category):
    """Расчет важности новости (1-10)"""
    score = 5  # Базовый
    
    # Критические категории
    if category in ['war/conflict', 'politics', 'economy']:
        score += 2
    
    # Сильная тональность
    if sentiment in ['positive', 'negative']:
        score += 1
    
    # Длина текста
    if len(text) > 300:
        score += 1
    
    # Ключевые слова
    text_lower = text.lower()
    if any(kw in text_lower for kw in ['president', 'war', 'crisis', 'президент', 'война']):
        score += 1
    
    return min(10, score)

def detect_fake_probability(text, source):
    """Вероятность фейка (0-1)"""
    prob = 0.0
    
    # Кликбейт паттерны
    for pattern in CLICKBAIT_PATTERNS:
        if re.search(pattern, text.lower()):
            prob += 0.2
    
    # Слишком эмоциональный текст
    exclamations = text.count('!')
    if exclamations > 3:
        prob += 0.15
    
    # Ненадежные источники (можно расширить)
    unreliable_sources = ['unknown', 'blog']
    if any(src in source.lower() for src in unreliable_sources):
        prob += 0.3
    
    return min(1.0, prob)

def calculate_fear_index(text, lang, emotions):
    """Индекс страха (0-1)"""
    fear_score = 0.0
    
    # Из эмоций
    if emotions and 'fear' in emotions:
        fear_score += emotions['fear'] * 0.5
    
    # Ключевые слова страха
    text_lower = text.lower()
    keywords = FEAR_KEYWORDS.get(lang, FEAR_KEYWORDS['en'])
    fear_words = sum(1 for kw in keywords if kw in text_lower)
    fear_score += min(0.5, fear_words * 0.1)
    
    return min(1.0, fear_score)

@app.get("/")
def root():
    return {"status": "Advanced Sentiment API running"}

@app.post("/analyze")
def analyze_pending():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT n.id, n.title, n.description, n.source, n.language
        FROM news n
        LEFT JOIN sentiment_results sr ON n.id = sr.news_id
        WHERE sr.id IS NULL
        LIMIT 20
    """)
    
    news = cursor.fetchall()
    analyzed = 0
    
    for news_id, title, description, source, lang in news:
        full_text = f"{title}. {description or ''}"
        text_for_model = full_text[:512]
        
        try:
            # 1. Базовая тональность
            sentiment_result = sentiment_model(text_for_model)[0]
            sentiment = sentiment_result['label'].lower()
            score = sentiment_result['score']
            
            # 2. Эмоции (только для английского)
            emotions = {}
            if lang == 'en':
                emotion_result = emotion_model(text_for_model[:256])[0]
                emotions = {e['label']: round(e['score'], 3) for e in emotion_result}
            
            # 3. Категория
            category_result = zero_shot(text_for_model, CATEGORIES)
            category = category_result['labels'][0]
            category_conf = category_result['scores'][0]
            
            # 4. Ключевые слова
            keywords = extract_keywords(full_text, lang)
            
            # 5. Сущности
            entities = extract_entities(full_text)
            
            # 6. Важность
            importance = calculate_importance(full_text, sentiment, category)
            
            # 7. Детекция фейка
            fake_prob = detect_fake_probability(full_text, source)
            
            # 8. Кликбейт
            is_clickbait = any(re.search(p, full_text.lower()) for p in CLICKBAIT_PATTERNS)
            
            # 9. Индекс страха
            fear_idx = calculate_fear_index(full_text, lang, emotions)
            
            # Сохранение
            cursor.execute("""
                INSERT INTO sentiment_results (
                    news_id, sentiment, score, emotions, category, category_confidence,
                    importance_score, keywords, entities, is_fake_probability,
                    is_clickbait, fear_index
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                news_id, sentiment, score, json.dumps(emotions), category, category_conf,
                importance, keywords, json.dumps(entities), fake_prob, is_clickbait, fear_idx
            ))
            
            analyzed += 1
            
        except Exception as e:
            print(f"Error analyzing news {news_id}: {e}")
    
    conn.commit()
    cursor.close()
    conn.close()
    
    return {"analyzed": analyzed}

@app.get("/stats")
def get_stats():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Общая статистика
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            AVG(fear_index) as avg_fear,
            AVG(is_fake_probability) as avg_fake,
            COUNT(*) FILTER (WHERE is_clickbait) as clickbait_count
        FROM sentiment_results
    """)
    stats = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    return {
        "total_analyzed": stats[0],
        "average_fear_index": round(stats[1], 3) if stats[1] else 0,
        "average_fake_probability": round(stats[2], 3) if stats[2] else 0,
        "clickbait_count": stats[3]
    }