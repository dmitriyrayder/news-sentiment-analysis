from fastapi import FastAPI
from transformers import pipeline
import psycopg2
import os
import re
from collections import Counter
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
from statistics import mean, stdev

app = FastAPI(title="Advanced Sentiment Analysis & Decision Support API")

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

# ============================================================================
# НОВЫЕ ФУНКЦИИ ДЛЯ ПРИНЯТИЯ РЕШЕНИЙ
# ============================================================================

def calculate_composite_risk_score(fear_index, fake_prob, sentiment, importance):
    """
    Композитный индекс риска (0-100)
    Комплексная оценка для принятия решений
    """
    risk = 0.0

    # Индекс страха (вес 30%)
    risk += fear_index * 30

    # Вероятность фейка (вес 25%)
    risk += fake_prob * 25

    # Негативная тональность (вес 20%)
    if sentiment == 'negative':
        risk += 20
    elif sentiment == 'neutral':
        risk += 10

    # Высокая важность = больше риск (вес 25%)
    risk += (importance / 10) * 25

    return min(100.0, risk)

def calculate_trust_score(source, fake_prob, is_clickbait):
    """
    Индекс доверия к источнику (0-100)
    100 = максимальное доверие
    """
    trust = 100.0

    # Вероятность фейка снижает доверие
    trust -= fake_prob * 50

    # Кликбейт снижает доверие
    if is_clickbait:
        trust -= 20

    # Известные надежные источники (можно расширить)
    trusted_sources = ['bbc', 'reuters', 'ap', 'dw', 'france24']
    if any(src in source.lower() for src in trusted_sources):
        trust = min(100, trust + 15)

    return max(0.0, trust)

def detect_sentiment_momentum(news_id, conn):
    """
    Определение momentum (скорости изменения) тональности
    Возвращает: 'accelerating_negative', 'stable', 'accelerating_positive'
    """
    cursor = conn.cursor()

    # Получаем последние 10 новостей по той же категории
    cursor.execute("""
        SELECT sr.sentiment, sr.score, n.published_date
        FROM sentiment_results sr
        JOIN news n ON sr.news_id = n.id
        WHERE sr.category = (
            SELECT category FROM sentiment_results WHERE news_id = %s
        )
        ORDER BY n.published_date DESC
        LIMIT 10
    """, (news_id,))

    results = cursor.fetchall()

    if len(results) < 5:
        return 'insufficient_data'

    # Конвертируем sentiment в числа
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    scores = [sentiment_map.get(r[0], 0) * r[1] for r in results]

    # Анализ тренда (простая линейная регрессия)
    if len(scores) >= 5:
        recent_avg = mean(scores[:3])
        older_avg = mean(scores[-3:])

        if recent_avg < older_avg - 0.2:
            return 'accelerating_negative'
        elif recent_avg > older_avg + 0.2:
            return 'accelerating_positive'

    return 'stable'

def detect_anomaly(fear_index, category, conn):
    """
    Детекция аномалий - необычно высокий страх для данной категории
    """
    cursor = conn.cursor()

    # Получаем статистику по категории за последние 7 дней
    cursor.execute("""
        SELECT AVG(fear_index) as avg_fear, STDDEV(fear_index) as std_fear
        FROM sentiment_results sr
        JOIN news n ON sr.news_id = n.id
        WHERE sr.category = %s
        AND n.published_date >= NOW() - INTERVAL '7 days'
        AND fear_index IS NOT NULL
    """, (category,))

    result = cursor.fetchone()

    if result and result[0] is not None and result[1] is not None:
        avg_fear, std_fear = result[0], result[1]

        # Аномалия если > 2 стандартных отклонений
        if std_fear > 0 and fear_index > avg_fear + 2 * std_fear:
            return True

    return False

def get_geographic_risks(entities):
    """
    Анализ географических рисков на основе упоминаемых стран
    """
    if not entities or 'countries' not in entities:
        return []

    # Список стран с высоким геополитическим риском (можно расширять)
    high_risk_countries = {
        'Russia': 'high',
        'Ukraine': 'high',
        'China': 'medium',
        'Iran': 'high',
        'North Korea': 'high',
        'Syria': 'high',
        'Росія': 'high',
        'Україна': 'high'
    }

    risks = []
    for country in entities.get('countries', []):
        risk_level = high_risk_countries.get(country, 'low')
        if risk_level in ['high', 'medium']:
            risks.append({
                'country': country,
                'risk_level': risk_level
            })

    return risks

def generate_decision_recommendations(risk_score, trust_score, momentum, is_anomaly, geo_risks):
    """
    Генерация конкретных рекомендаций для принятия решений
    """
    recommendations = []

    # Высокий риск
    if risk_score > 70:
        recommendations.append({
            'level': 'critical',
            'action': 'immediate_attention',
            'message': 'Критический уровень риска. Требуется немедленное внимание и верификация информации.'
        })
    elif risk_score > 50:
        recommendations.append({
            'level': 'warning',
            'action': 'monitor_closely',
            'message': 'Повышенный риск. Рекомендуется усиленный мониторинг ситуации.'
        })

    # Низкое доверие
    if trust_score < 40:
        recommendations.append({
            'level': 'warning',
            'action': 'verify_source',
            'message': 'Низкий уровень доверия к источнику. Требуется проверка через альтернативные источники.'
        })

    # Негативный momentum
    if momentum == 'accelerating_negative':
        recommendations.append({
            'level': 'alert',
            'action': 'trend_analysis',
            'message': 'Обнаружен ускоряющийся негативный тренд. Рекомендуется анализ причин.'
        })

    # Аномалия
    if is_anomaly:
        recommendations.append({
            'level': 'alert',
            'action': 'anomaly_investigation',
            'message': 'Обнаружена статистическая аномалия. Возможно начало кризисной ситуации.'
        })

    # Географические риски
    if geo_risks:
        high_risk_count = sum(1 for r in geo_risks if r['risk_level'] == 'high')
        if high_risk_count > 0:
            recommendations.append({
                'level': 'warning',
                'action': 'geopolitical_assessment',
                'message': f'Упоминаются {high_risk_count} страны с высоким геополитическим риском.'
            })

    # Если рисков нет
    if not recommendations:
        recommendations.append({
            'level': 'info',
            'action': 'routine_monitoring',
            'message': 'Ситуация в норме. Продолжайте рутинный мониторинг.'
        })

    return recommendations

# ============================================================================
# ВРЕМЕННЫЕ ПАТТЕРНЫ И ТРЕНДЫ
# ============================================================================

def analyze_emerging_topics(conn, days=7):
    """
    Определение восходящих тем (emerging topics)
    Сравнивает активность ключевых слов за разные периоды
    """
    cursor = conn.cursor()

    # Ключевые слова за последние 2 дня
    cursor.execute("""
        SELECT unnest(keywords) as keyword, COUNT(*) as count
        FROM sentiment_results sr
        JOIN news n ON sr.news_id = n.id
        WHERE n.published_date >= NOW() - INTERVAL '2 days'
        AND keywords IS NOT NULL
        GROUP BY keyword
        ORDER BY count DESC
        LIMIT 50
    """)
    recent_keywords = {row[0]: row[1] for row in cursor.fetchall()}

    # Ключевые слова за предыдущий период (3-7 дней назад)
    cursor.execute("""
        SELECT unnest(keywords) as keyword, COUNT(*) as count
        FROM sentiment_results sr
        JOIN news n ON sr.news_id = n.id
        WHERE n.published_date >= NOW() - INTERVAL '7 days'
        AND n.published_date < NOW() - INTERVAL '2 days'
        AND keywords IS NOT NULL
        GROUP BY keyword
        ORDER BY count DESC
        LIMIT 50
    """)
    older_keywords = {row[0]: row[1] for row in cursor.fetchall()}

    emerging = []
    declining = []

    for keyword, recent_count in recent_keywords.items():
        older_count = older_keywords.get(keyword, 0)

        # Рост более чем в 2 раза
        if older_count > 0:
            growth_rate = (recent_count - older_count) / older_count
            if growth_rate > 1.0:  # 100% рост
                emerging.append({
                    'keyword': keyword,
                    'recent_mentions': recent_count,
                    'previous_mentions': older_count,
                    'growth_rate': round(growth_rate, 2)
                })
        elif recent_count >= 5:  # Новая тема
            emerging.append({
                'keyword': keyword,
                'recent_mentions': recent_count,
                'previous_mentions': 0,
                'growth_rate': 'new'
            })

    # Угасающие темы
    for keyword, older_count in older_keywords.items():
        recent_count = recent_keywords.get(keyword, 0)
        if older_count > 5 and recent_count < older_count / 2:
            decline_rate = (older_count - recent_count) / older_count
            declining.append({
                'keyword': keyword,
                'recent_mentions': recent_count,
                'previous_mentions': older_count,
                'decline_rate': round(decline_rate, 2)
            })

    return {
        'emerging': sorted(emerging, key=lambda x: x['recent_mentions'], reverse=True)[:10],
        'declining': sorted(declining, key=lambda x: x['previous_mentions'], reverse=True)[:10]
    }

def calculate_sentiment_volatility(conn, category, days=7):
    """
    Волатильность настроений - насколько резко меняется sentiment
    Высокая волатильность = нестабильная ситуация
    """
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            DATE(n.published_date) as date,
            AVG(CASE
                WHEN sr.sentiment = 'positive' THEN 1
                WHEN sr.sentiment = 'negative' THEN -1
                ELSE 0
            END) as avg_sentiment
        FROM sentiment_results sr
        JOIN news n ON sr.news_id = n.id
        WHERE sr.category = %s
        AND n.published_date >= NOW() - INTERVAL '%s days'
        GROUP BY DATE(n.published_date)
        ORDER BY date
    """, (category, days))

    data = cursor.fetchall()

    if len(data) < 3:
        return 0.0

    sentiments = [row[1] for row in data if row[1] is not None]

    if len(sentiments) < 2:
        return 0.0

    # Стандартное отклонение как мера волатильности
    volatility = stdev(sentiments)

    return round(volatility, 3)

# ============================================================================
# ДЕТЕКЦИЯ ПРОПАГАНДЫ И МАНИПУЛЯЦИЙ
# ============================================================================

def detect_propaganda_patterns(text, source, category):
    """
    Детекция признаков пропаганды и манипуляций
    """
    propaganda_score = 0.0
    flags = []

    text_lower = text.lower()

    # 1. Эмоциональная манипуляция
    emotional_words = ['ужас', 'шок', 'катастрофа', 'скандал', 'сенсация',
                       'horror', 'shock', 'disaster', 'scandal', 'sensation']
    emotional_count = sum(1 for word in emotional_words if word in text_lower)
    if emotional_count >= 3:
        propaganda_score += 0.3
        flags.append('emotional_manipulation')

    # 2. Абсолютные утверждения
    absolute_patterns = ['все знают', 'никто не', 'всегда', 'никогда', 'каждый',
                         'everyone knows', 'nobody', 'always', 'never', 'everybody']
    if any(pattern in text_lower for pattern in absolute_patterns):
        propaganda_score += 0.2
        flags.append('absolute_statements')

    # 3. Нас vs Них (дихотомия)
    us_vs_them = ['наши враги', 'они против нас', 'our enemies', 'they against us']
    if any(pattern in text_lower for pattern in us_vs_them):
        propaganda_score += 0.25
        flags.append('us_vs_them')

    # 4. Призыв к действию без аргументов
    call_to_action = ['мы должны', 'необходимо', 'пора', 'we must', 'it is time']
    if any(pattern in text_lower for pattern in call_to_action):
        propaganda_score += 0.15
        flags.append('call_to_action')

    # 5. Односторонность (только негатив или только позитив по политике/войне)
    if category in ['politics', 'war/conflict']:
        if text_lower.count('!') > 5:
            propaganda_score += 0.1
            flags.append('excessive_exclamation')

    return {
        'propaganda_score': min(1.0, propaganda_score),
        'flags': flags,
        'risk_level': 'high' if propaganda_score > 0.6 else 'medium' if propaganda_score > 0.3 else 'low'
    }

def calculate_source_consistency_score(source, conn):
    """
    Индекс последовательности источника
    Проверяет, насколько стабилен источник в качестве
    """
    cursor = conn.cursor()

    # Статистика за последние 30 дней
    cursor.execute("""
        SELECT
            AVG(is_fake_probability) as avg_fake,
            STDDEV(is_fake_probability) as std_fake,
            AVG(CASE WHEN is_clickbait THEN 1 ELSE 0 END) as clickbait_rate,
            COUNT(*) as total
        FROM sentiment_results sr
        JOIN news n ON sr.news_id = n.id
        WHERE n.source = %s
        AND n.published_date >= NOW() - INTERVAL '30 days'
    """, (source,))

    result = cursor.fetchone()

    if not result or result[3] < 10:
        return {
            'consistency_score': 50,
            'status': 'insufficient_data',
            'total_articles': result[3] if result else 0
        }

    avg_fake, std_fake, clickbait_rate, total = result

    # Чем меньше разброс - тем лучше
    consistency = 100

    if avg_fake:
        consistency -= avg_fake * 30  # Средняя фейковость

    if std_fake and std_fake > 0.2:  # Высокая нестабильность
        consistency -= 20

    if clickbait_rate > 0.3:
        consistency -= 15

    consistency = max(0, min(100, consistency))

    return {
        'consistency_score': round(consistency, 2),
        'status': 'reliable' if consistency > 70 else 'questionable' if consistency > 40 else 'unreliable',
        'total_articles': total,
        'avg_fake_probability': round(avg_fake, 3) if avg_fake else 0,
        'stability': 'stable' if (std_fake or 0) < 0.15 else 'unstable'
    }

# ============================================================================
# СОЦИАЛЬНЫЙ АНАЛИЗ
# ============================================================================

def analyze_society_fears(conn, region=None):
    """
    Анализ страхов общества по ключевым словам и категориям
    """
    cursor = conn.cursor()

    fear_categories = {
        'war': ['war', 'война', 'conflict', 'конфликт', 'attack', 'атака'],
        'economy': ['crisis', 'кризис', 'inflation', 'инфляция', 'poverty', 'бедность'],
        'health': ['pandemic', 'пандемия', 'disease', 'болезнь', 'virus', 'вирус'],
        'climate': ['climate', 'климат', 'disaster', 'катастрофа', 'flood', 'наводнение'],
        'crime': ['crime', 'преступ', 'terror', 'террор', 'violence', 'насилие']
    }

    fears = {}

    for fear_type, keywords in fear_categories.items():
        # Подсчет новостей с высоким индексом страха по каждой категории
        keyword_conditions = ' OR '.join([f"LOWER(n.title || ' ' || n.description) LIKE %s" for _ in keywords])

        cursor.execute(f"""
            SELECT
                COUNT(*) as count,
                AVG(sr.fear_index) as avg_fear,
                AVG(sr.importance_score) as avg_importance
            FROM sentiment_results sr
            JOIN news n ON sr.news_id = n.id
            WHERE ({keyword_conditions})
            AND n.published_date >= NOW() - INTERVAL '7 days'
            AND sr.fear_index > 0.5
        """, tuple(f'%{kw}%' for kw in keywords))

        result = cursor.fetchone()

        if result and result[0] > 0:
            fears[fear_type] = {
                'mention_count': result[0],
                'avg_fear_index': round(result[1], 3) if result[1] else 0,
                'avg_importance': round(result[2], 2) if result[2] else 0,
                'intensity': 'high' if result[1] and result[1] > 0.7 else 'medium' if result[1] and result[1] > 0.5 else 'low'
            }

    # Сортируем по интенсивности
    sorted_fears = dict(sorted(fears.items(), key=lambda x: x[1]['mention_count'], reverse=True))

    return sorted_fears

def calculate_optimism_index(conn, days=7):
    """
    Индекс оптимизма/пессимизма общества (0-100)
    100 = максимальный оптимизм
    """
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) FILTER (WHERE sentiment = 'positive') as positive_count,
            COUNT(*) FILTER (WHERE sentiment = 'negative') as negative_count,
            COUNT(*) FILTER (WHERE sentiment = 'neutral') as neutral_count,
            AVG(fear_index) as avg_fear,
            AVG(CASE WHEN is_clickbait THEN 1 ELSE 0 END) as clickbait_rate
        FROM sentiment_results sr
        JOIN news n ON sr.news_id = n.id
        WHERE n.published_date >= NOW() - INTERVAL '%s days'
    """ % days)

    result = cursor.fetchone()

    if not result:
        return {'optimism_index': 50, 'status': 'neutral'}

    pos, neg, neu, avg_fear, clickbait = result
    total = pos + neg + neu

    if total == 0:
        return {'optimism_index': 50, 'status': 'insufficient_data'}

    # Расчет индекса
    positive_ratio = pos / total
    negative_ratio = neg / total

    optimism = 50  # Базовая точка
    optimism += (positive_ratio - negative_ratio) * 50  # От -50 до +50

    # Корректировка на страх
    if avg_fear:
        optimism -= avg_fear * 20

    # Корректировка на кликбейт (признак паники)
    if clickbait:
        optimism -= clickbait * 10

    optimism = max(0, min(100, optimism))

    # Определение статуса
    if optimism > 70:
        status = 'optimistic'
    elif optimism > 50:
        status = 'cautiously_optimistic'
    elif optimism > 30:
        status = 'pessimistic'
    else:
        status = 'highly_pessimistic'

    return {
        'optimism_index': round(optimism, 2),
        'status': status,
        'sentiment_breakdown': {
            'positive': round(positive_ratio * 100, 1),
            'negative': round(negative_ratio * 100, 1),
            'neutral': round((neu / total) * 100, 1)
        }
    }

def detect_apathy_index(conn):
    """
    Детекция апатии общества
    Признаки: низкая важность новостей, рост нейтрального sentiment, снижение активности
    """
    cursor = conn.cursor()

    # Последние 3 дня vs предыдущие 7 дней
    cursor.execute("""
        WITH recent AS (
            SELECT
                AVG(importance_score) as avg_importance,
                COUNT(*) FILTER (WHERE sentiment = 'neutral') * 100.0 / COUNT(*) as neutral_rate
            FROM sentiment_results sr
            JOIN news n ON sr.news_id = n.id
            WHERE n.published_date >= NOW() - INTERVAL '3 days'
        ),
        older AS (
            SELECT
                AVG(importance_score) as avg_importance,
                COUNT(*) FILTER (WHERE sentiment = 'neutral') * 100.0 / COUNT(*) as neutral_rate
            FROM sentiment_results sr
            JOIN news n ON sr.news_id = n.id
            WHERE n.published_date >= NOW() - INTERVAL '10 days'
            AND n.published_date < NOW() - INTERVAL '3 days'
        )
        SELECT
            recent.avg_importance as recent_imp,
            older.avg_importance as older_imp,
            recent.neutral_rate as recent_neutral,
            older.neutral_rate as older_neutral
        FROM recent, older
    """)

    result = cursor.fetchone()

    if not result:
        return {'apathy_index': 0, 'status': 'insufficient_data'}

    recent_imp, older_imp, recent_neutral, older_neutral = result

    apathy = 0.0
    indicators = []

    # Снижение важности
    if recent_imp and older_imp and recent_imp < older_imp - 1:
        apathy += 0.3
        indicators.append('declining_importance')

    # Рост нейтральности
    if recent_neutral and older_neutral and recent_neutral > older_neutral + 10:
        apathy += 0.4
        indicators.append('increasing_neutrality')

    # Общий уровень нейтральности
    if recent_neutral and recent_neutral > 50:
        apathy += 0.3
        indicators.append('high_neutrality')

    apathy = min(1.0, apathy)

    return {
        'apathy_index': round(apathy, 2),
        'status': 'high_apathy' if apathy > 0.6 else 'moderate_apathy' if apathy > 0.3 else 'engaged',
        'indicators': indicators
    }

# ============================================================================
# БИЗНЕС-ИНДИКАТОРЫ
# ============================================================================

def calculate_investment_risk_index(conn, region=None, sector=None):
    """
    Индекс риска для инвестиций (0-100)
    100 = максимальный риск
    """
    cursor = conn.cursor()

    # Анализ новостей по экономике, политике, войнам
    high_risk_categories = ['war/conflict', 'politics', 'economy']

    conditions = ["sr.category = ANY(%s)"]
    params = [high_risk_categories]

    cursor.execute("""
        SELECT
            AVG(fear_index) as avg_fear,
            AVG(is_fake_probability) as avg_fake,
            COUNT(*) FILTER (WHERE sentiment = 'negative') * 100.0 / NULLIF(COUNT(*), 0) as negative_rate,
            AVG(importance_score) as avg_importance
        FROM sentiment_results sr
        JOIN news n ON sr.news_id = n.id
        WHERE sr.category = ANY(%s)
        AND n.published_date >= NOW() - INTERVAL '7 days'
    """, params)

    result = cursor.fetchone()

    if not result:
        return {'investment_risk': 50, 'status': 'moderate'}

    avg_fear, avg_fake, negative_rate, avg_importance = result

    risk = 0.0

    # Факторы риска
    if avg_fear:
        risk += avg_fear * 30  # 0-30 points

    if negative_rate:
        risk += (negative_rate / 100) * 30  # 0-30 points

    if avg_importance and avg_importance > 7:
        risk += 20  # Высокая важность = высокий риск

    if avg_fake and avg_fake > 0.4:
        risk += 20  # Много дезинформации = риск

    risk = min(100, risk)

    return {
        'investment_risk': round(risk, 2),
        'status': 'critical' if risk > 70 else 'high' if risk > 50 else 'moderate' if risk > 30 else 'low',
        'factors': {
            'fear_level': round(avg_fear, 3) if avg_fear else 0,
            'negative_sentiment': round(negative_rate, 1) if negative_rate else 0,
            'disinformation_level': round(avg_fake, 3) if avg_fake else 0
        }
    }

def analyze_reputation_risks(source_name, conn):
    """
    Анализ репутационных рисков для организации/бренда
    """
    cursor = conn.cursor()

    # Поиск упоминаний в новостях
    cursor.execute("""
        SELECT
            COUNT(*) as total_mentions,
            AVG(CASE WHEN sr.sentiment = 'negative' THEN 1 ELSE 0 END) as negative_rate,
            AVG(sr.fear_index) as avg_fear,
            COUNT(*) FILTER (WHERE sr.is_fake_probability > 0.6) as potential_fake_mentions
        FROM sentiment_results sr
        JOIN news n ON sr.news_id = n.id
        WHERE LOWER(n.title || ' ' || COALESCE(n.description, '')) LIKE %s
        AND n.published_date >= NOW() - INTERVAL '30 days'
    """, (f'%{source_name.lower()}%',))

    result = cursor.fetchone()

    if not result or result[0] == 0:
        return {
            'reputation_risk': 0,
            'status': 'no_mentions',
            'total_mentions': 0
        }

    total, neg_rate, avg_fear, fake_count = result

    risk = 0.0

    # Высокий негатив
    if neg_rate and neg_rate > 0.5:
        risk += 40

    # Страх в упоминаниях
    if avg_fear and avg_fear > 0.6:
        risk += 30

    # Фейковые упоминания
    if fake_count > 0:
        risk += 30

    risk = min(100, risk)

    return {
        'reputation_risk': round(risk, 2),
        'status': 'critical' if risk > 70 else 'concerning' if risk > 40 else 'stable',
        'total_mentions': total,
        'negative_rate': round(neg_rate * 100, 1) if neg_rate else 0,
        'fake_mentions': fake_count
    }

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

            # 10. Композитный индекс риска
            risk_score = calculate_composite_risk_score(fear_idx, fake_prob, sentiment, importance)

            # 11. Индекс доверия к источнику
            trust = calculate_trust_score(source, fake_prob, is_clickbait)

            # 12. Географические риски
            geo_risks = get_geographic_risks(entities)

            # Сохранение (расширенная версия)
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

            # После сохранения - анализ momentum и аномалий
            momentum = detect_sentiment_momentum(news_id, conn)
            is_anomaly = detect_anomaly(fear_idx, category, conn)

            analyzed += 1

            # Логирование важных событий
            if risk_score > 70 or is_anomaly:
                print(f"⚠️ HIGH RISK detected: news_id={news_id}, risk={risk_score:.1f}, anomaly={is_anomaly}")
            
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

# ============================================================================
# НОВЫЕ ENDPOINTS ДЛЯ ПРИНЯТИЯ РЕШЕНИЙ
# ============================================================================

@app.get("/trends")
def get_trends():
    """
    Анализ трендов: динамика настроений по категориям
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Тренды по категориям за последние 7 дней
    cursor.execute("""
        SELECT
            sr.category,
            DATE(n.published_date) as date,
            AVG(CASE
                WHEN sr.sentiment = 'positive' THEN 1
                WHEN sr.sentiment = 'negative' THEN -1
                ELSE 0
            END) as sentiment_score,
            AVG(sr.fear_index) as avg_fear,
            COUNT(*) as news_count
        FROM sentiment_results sr
        JOIN news n ON sr.news_id = n.id
        WHERE n.published_date >= NOW() - INTERVAL '7 days'
        GROUP BY sr.category, DATE(n.published_date)
        ORDER BY date DESC, sr.category
    """)

    trends = []
    for row in cursor.fetchall():
        trends.append({
            'category': row[0],
            'date': row[1].isoformat() if row[1] else None,
            'sentiment_score': round(row[2], 3) if row[2] else 0,
            'fear_index': round(row[3], 3) if row[3] else 0,
            'news_count': row[4]
        })

    cursor.close()
    conn.close()

    return {"trends": trends}

@app.get("/alerts")
def get_alerts():
    """
    Критические алерты и предупреждения
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    alerts = []

    # 1. Новости с высоким индексом страха (>0.7) за последние 24 часа
    cursor.execute("""
        SELECT
            n.id, n.title, n.source, sr.fear_index, sr.category,
            n.published_date
        FROM news n
        JOIN sentiment_results sr ON n.id = sr.news_id
        WHERE sr.fear_index > 0.7
        AND n.published_date >= NOW() - INTERVAL '24 hours'
        ORDER BY sr.fear_index DESC
        LIMIT 10
    """)

    for row in cursor.fetchall():
        alerts.append({
            'type': 'high_fear',
            'news_id': row[0],
            'title': row[1],
            'source': row[2],
            'fear_index': round(row[3], 3),
            'category': row[4],
            'published': row[5].isoformat() if row[5] else None,
            'severity': 'critical' if row[3] > 0.85 else 'warning'
        })

    # 2. Вероятные фейки с высокой важностью
    cursor.execute("""
        SELECT
            n.id, n.title, n.source, sr.is_fake_probability,
            sr.importance_score
        FROM news n
        JOIN sentiment_results sr ON n.id = sr.news_id
        WHERE sr.is_fake_probability > 0.6
        AND sr.importance_score >= 7
        AND n.published_date >= NOW() - INTERVAL '24 hours'
        ORDER BY sr.is_fake_probability DESC
        LIMIT 5
    """)

    for row in cursor.fetchall():
        alerts.append({
            'type': 'potential_fake',
            'news_id': row[0],
            'title': row[1],
            'source': row[2],
            'fake_probability': round(row[3], 3),
            'importance': row[4],
            'severity': 'warning'
        })

    # 3. Категории с резким ростом негатива
    cursor.execute("""
        WITH recent AS (
            SELECT category, AVG(CASE
                WHEN sentiment = 'negative' THEN 1 ELSE 0
            END) as neg_rate
            FROM sentiment_results sr
            JOIN news n ON sr.news_id = n.id
            WHERE n.published_date >= NOW() - INTERVAL '24 hours'
            GROUP BY category
        ),
        older AS (
            SELECT category, AVG(CASE
                WHEN sentiment = 'negative' THEN 1 ELSE 0
            END) as neg_rate
            FROM sentiment_results sr
            JOIN news n ON sr.news_id = n.id
            WHERE n.published_date >= NOW() - INTERVAL '7 days'
            AND n.published_date < NOW() - INTERVAL '24 hours'
            GROUP BY category
        )
        SELECT
            recent.category,
            recent.neg_rate as recent_negative,
            older.neg_rate as baseline_negative,
            (recent.neg_rate - older.neg_rate) as spike
        FROM recent
        LEFT JOIN older ON recent.category = older.category
        WHERE recent.neg_rate - COALESCE(older.neg_rate, 0) > 0.3
        ORDER BY spike DESC
    """)

    for row in cursor.fetchall():
        alerts.append({
            'type': 'sentiment_spike',
            'category': row[0],
            'recent_negative_rate': round(row[1], 3),
            'baseline_negative_rate': round(row[2], 3) if row[2] else 0,
            'spike': round(row[3], 3),
            'severity': 'alert'
        })

    cursor.close()
    conn.close()

    return {
        "alerts": alerts,
        "total_alerts": len(alerts),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/risk-assessment")
def get_risk_assessment():
    """
    Комплексная оценка рисков
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Топ новостей по риску за последние 24 часа
    cursor.execute("""
        SELECT
            n.id,
            n.title,
            n.source,
            n.url,
            sr.category,
            sr.sentiment,
            sr.fear_index,
            sr.is_fake_probability,
            sr.importance_score,
            sr.is_clickbait,
            sr.entities,
            n.published_date
        FROM news n
        JOIN sentiment_results sr ON n.id = sr.news_id
        WHERE n.published_date >= NOW() - INTERVAL '24 hours'
        ORDER BY n.published_date DESC
        LIMIT 20
    """)

    risk_items = []
    for row in cursor.fetchall():
        news_id, title, source, url, category = row[0], row[1], row[2], row[3], row[4]
        sentiment, fear_idx, fake_prob, importance = row[5], row[6], row[7], row[8]
        is_clickbait, entities = row[9], row[10]

        # Расчет метрик
        risk_score = calculate_composite_risk_score(fear_idx, fake_prob, sentiment, importance)
        trust_score = calculate_trust_score(source, fake_prob, is_clickbait)
        momentum = detect_sentiment_momentum(news_id, conn)
        is_anomaly = detect_anomaly(fear_idx, category, conn)

        # Парсинг entities если это строка
        if isinstance(entities, str):
            try:
                entities = json.loads(entities)
            except:
                entities = {}

        geo_risks = get_geographic_risks(entities)
        recommendations = generate_decision_recommendations(
            risk_score, trust_score, momentum, is_anomaly, geo_risks
        )

        risk_items.append({
            'news_id': news_id,
            'title': title,
            'source': source,
            'url': url,
            'category': category,
            'metrics': {
                'composite_risk_score': round(risk_score, 2),
                'trust_score': round(trust_score, 2),
                'fear_index': round(fear_idx, 3) if fear_idx else 0,
                'fake_probability': round(fake_prob, 3) if fake_prob else 0,
                'importance': importance,
                'sentiment': sentiment
            },
            'analysis': {
                'momentum': momentum,
                'is_anomaly': is_anomaly,
                'geographic_risks': geo_risks
            },
            'recommendations': recommendations,
            'published': row[11].isoformat() if row[11] else None
        })

    # Сортируем по риску
    risk_items.sort(key=lambda x: x['metrics']['composite_risk_score'], reverse=True)

    # Общая оценка рисков
    avg_risk = mean([item['metrics']['composite_risk_score'] for item in risk_items]) if risk_items else 0
    high_risk_count = sum(1 for item in risk_items if item['metrics']['composite_risk_score'] > 70)
    anomaly_count = sum(1 for item in risk_items if item['analysis']['is_anomaly'])

    cursor.close()
    conn.close()

    return {
        "summary": {
            "average_risk_score": round(avg_risk, 2),
            "high_risk_items": high_risk_count,
            "anomalies_detected": anomaly_count,
            "total_analyzed": len(risk_items),
            "overall_status": "critical" if avg_risk > 70 else "warning" if avg_risk > 50 else "normal"
        },
        "risk_items": risk_items[:10],  # Топ 10
        "timestamp": datetime.now().isoformat()
    }

@app.get("/predictions")
def get_predictions():
    """
    Прогнозирование: предсказание трендов на основе исторических данных
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    predictions = []

    # Анализ по категориям
    categories = ['war/conflict', 'politics', 'economy', 'technology']

    for category in categories:
        # Получаем данные за последние 7 дней
        cursor.execute("""
            SELECT
                DATE(n.published_date) as date,
                AVG(sr.fear_index) as avg_fear,
                AVG(CASE
                    WHEN sr.sentiment = 'negative' THEN 1 ELSE 0
                END) as neg_rate,
                COUNT(*) as count
            FROM sentiment_results sr
            JOIN news n ON sr.news_id = n.id
            WHERE sr.category = %s
            AND n.published_date >= NOW() - INTERVAL '7 days'
            GROUP BY DATE(n.published_date)
            ORDER BY date
        """, (category,))

        data = cursor.fetchall()

        if len(data) >= 3:
            # Простое прогнозирование на основе тренда
            fear_values = [row[1] for row in data if row[1] is not None]
            neg_rates = [row[2] for row in data if row[2] is not None]

            if fear_values and neg_rates:
                # Линейная экстраполяция
                recent_fear = mean(fear_values[-3:])
                older_fear = mean(fear_values[:3]) if len(fear_values) >= 3 else recent_fear
                fear_trend = recent_fear - older_fear

                recent_neg = mean(neg_rates[-3:])
                older_neg = mean(neg_rates[:3]) if len(neg_rates) >= 3 else recent_neg
                neg_trend = recent_neg - older_neg

                # Прогноз на следующий день
                predicted_fear = max(0, min(1, recent_fear + fear_trend))
                predicted_neg_rate = max(0, min(1, recent_neg + neg_trend))

                # Определение направления
                if fear_trend > 0.1 or neg_trend > 0.1:
                    direction = "worsening"
                    confidence = "medium"
                elif fear_trend < -0.1 or neg_trend < -0.1:
                    direction = "improving"
                    confidence = "medium"
                else:
                    direction = "stable"
                    confidence = "high"

                predictions.append({
                    'category': category,
                    'current_fear_index': round(recent_fear, 3),
                    'predicted_fear_index': round(predicted_fear, 3),
                    'current_negative_rate': round(recent_neg, 3),
                    'predicted_negative_rate': round(predicted_neg_rate, 3),
                    'direction': direction,
                    'confidence': confidence,
                    'recommendation': (
                        "Ситуация может ухудшиться. Усильте мониторинг." if direction == "worsening"
                        else "Ситуация улучшается. Продолжайте наблюдение." if direction == "improving"
                        else "Ситуация стабильна. Рутинный мониторинг."
                    )
                })

    cursor.close()
    conn.close()

    return {
        "predictions": predictions,
        "horizon": "24_hours",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/source-reliability")
def get_source_reliability():
    """
    Рейтинг надежности источников
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            n.source,
            COUNT(*) as total_news,
            AVG(sr.is_fake_probability) as avg_fake_prob,
            AVG(CASE WHEN sr.is_clickbait THEN 1 ELSE 0 END) as clickbait_rate,
            AVG(sr.importance_score) as avg_importance
        FROM news n
        JOIN sentiment_results sr ON n.id = sr.news_id
        WHERE n.published_date >= NOW() - INTERVAL '30 days'
        GROUP BY n.source
        HAVING COUNT(*) >= 5
        ORDER BY avg_fake_prob ASC, clickbait_rate ASC
    """)

    sources = []
    for row in cursor.fetchall():
        source, total, fake_prob, clickbait_rate, avg_importance = row

        # Расчет trust score
        trust = calculate_trust_score(source, fake_prob, clickbait_rate > 0.3)

        # Рейтинг
        if trust >= 80:
            rating = "excellent"
        elif trust >= 60:
            rating = "good"
        elif trust >= 40:
            rating = "moderate"
        else:
            rating = "low"

        sources.append({
            'source': source,
            'trust_score': round(trust, 2),
            'rating': rating,
            'statistics': {
                'total_articles': total,
                'avg_fake_probability': round(fake_prob, 3),
                'clickbait_rate': round(clickbait_rate, 3),
                'avg_importance': round(avg_importance, 2)
            }
        })

    cursor.close()
    conn.close()

    return {
        "sources": sources,
        "total_sources": len(sources),
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# РАСШИРЕННЫЕ ENDPOINTS ДЛЯ ГЛУБОКОЙ АНАЛИТИКИ
# ============================================================================

@app.get("/emerging-topics")
def get_emerging_topics():
    """
    Восходящие и угасающие темы в новостях
    Показывает "что волнует мир сейчас vs неделю назад"
    """
    conn = get_db_connection()

    result = analyze_emerging_topics(conn, days=7)

    conn.close()

    return {
        "emerging_topics": result['emerging'],
        "declining_topics": result['declining'],
        "analysis_period": "last_7_days",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/propaganda-detection")
def detect_propaganda():
    """
    Детекция пропаганды и манипуляций в новостях
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Анализ последних новостей
    cursor.execute("""
        SELECT
            n.id,
            n.title,
            n.description,
            n.source,
            sr.category
        FROM news n
        JOIN sentiment_results sr ON n.id = sr.news_id
        WHERE n.published_date >= NOW() - INTERVAL '24 hours'
        ORDER BY n.published_date DESC
        LIMIT 50
    """)

    propaganda_items = []

    for row in cursor.fetchall():
        news_id, title, desc, source, category = row
        full_text = f"{title}. {desc or ''}"

        propaganda_result = detect_propaganda_patterns(full_text, source, category)

        if propaganda_result['propaganda_score'] > 0.3:  # Только подозрительные
            propaganda_items.append({
                'news_id': news_id,
                'title': title,
                'source': source,
                'propaganda_score': propaganda_result['propaganda_score'],
                'risk_level': propaganda_result['risk_level'],
                'flags': propaganda_result['flags']
            })

    # Сортируем по propaganda_score
    propaganda_items.sort(key=lambda x: x['propaganda_score'], reverse=True)

    cursor.close()
    conn.close()

    return {
        "high_risk_items": [item for item in propaganda_items if item['risk_level'] == 'high'],
        "medium_risk_items": [item for item in propaganda_items if item['risk_level'] == 'medium'],
        "total_flagged": len(propaganda_items),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/source-consistency/{source_name}")
def get_source_consistency(source_name: str):
    """
    Индекс последовательности и надежности источника
    """
    conn = get_db_connection()

    consistency = calculate_source_consistency_score(source_name, conn)

    conn.close()

    return {
        "source": source_name,
        "consistency": consistency,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/society-analysis")
def get_society_analysis():
    """
    Комплексный анализ настроений общества
    Включает страхи, оптимизм, апатию
    """
    conn = get_db_connection()

    # Анализ страхов
    fears = analyze_society_fears(conn)

    # Индекс оптимизма
    optimism = calculate_optimism_index(conn, days=7)

    # Индекс апатии
    apathy = detect_apathy_index(conn)

    conn.close()

    return {
        "fears": fears,
        "optimism": optimism,
        "apathy": apathy,
        "summary": {
            "dominant_fear": list(fears.keys())[0] if fears else None,
            "mood": optimism['status'],
            "engagement": apathy['status']
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/business-intelligence")
def get_business_intelligence():
    """
    Бизнес-индикаторы для принятия инвестиционных решений
    """
    conn = get_db_connection()

    # Инвестиционный риск
    investment_risk = calculate_investment_risk_index(conn)

    # Волатильность по категориям
    volatility = {}
    for category in ['politics', 'economy', 'war/conflict', 'technology']:
        vol = calculate_sentiment_volatility(conn, category, days=7)
        if vol > 0:
            volatility[category] = {
                'volatility': vol,
                'stability': 'unstable' if vol > 0.5 else 'moderate' if vol > 0.3 else 'stable'
            }

    conn.close()

    return {
        "investment_risk": investment_risk,
        "market_volatility": volatility,
        "recommendation": (
            "HOLD - Высокий риск, избегайте новых инвестиций"
            if investment_risk['investment_risk'] > 70
            else "CAUTION - Умеренный риск, инвестируйте осторожно"
            if investment_risk['investment_risk'] > 50
            else "FAVORABLE - Низкий риск, благоприятные условия"
        ),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/reputation-monitor/{brand_name}")
def monitor_reputation(brand_name: str):
    """
    Мониторинг репутации бренда/организации
    """
    conn = get_db_connection()

    reputation = analyze_reputation_risks(brand_name, conn)

    conn.close()

    return {
        "brand": brand_name,
        "reputation_analysis": reputation,
        "action_required": reputation['status'] in ['critical', 'concerning'],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/volatility-index")
def get_volatility_index():
    """
    Индекс волатильности новостей по всем категориям
    """
    conn = get_db_connection()

    categories = ['war/conflict', 'politics', 'economy', 'technology',
                  'health', 'environment', 'sports', 'entertainment']

    volatility_data = []

    for category in categories:
        vol = calculate_sentiment_volatility(conn, category, days=7)
        if vol > 0:
            volatility_data.append({
                'category': category,
                'volatility': vol,
                'status': 'highly_volatile' if vol > 0.6 else 'volatile' if vol > 0.4 else 'stable'
            })

    # Сортируем по волатильности
    volatility_data.sort(key=lambda x: x['volatility'], reverse=True)

    # Общий индекс волатильности
    avg_volatility = mean([item['volatility'] for item in volatility_data]) if volatility_data else 0

    conn.close()

    return {
        "overall_volatility": round(avg_volatility, 3),
        "status": 'unstable' if avg_volatility > 0.5 else 'moderate' if avg_volatility > 0.3 else 'stable',
        "categories": volatility_data,
        "interpretation": (
            "Крайне нестабильная информационная среда. Высокий риск резких изменений."
            if avg_volatility > 0.6
            else "Умеренная волатильность. Ситуация может измениться."
            if avg_volatility > 0.3
            else "Стабильная информационная среда. Предсказуемая динамика."
        ),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/comprehensive-dashboard")
def get_comprehensive_dashboard():
    """
    Комплексный дашборд - все ключевые метрики для принятия решений
    """
    conn = get_db_connection()

    # Сбор всех метрик
    dashboard = {
        "alerts": [],
        "risk_assessment": {
            "investment_risk": calculate_investment_risk_index(conn),
            "composite_risk": None  # Будет заполнено из risk-assessment
        },
        "society": {
            "optimism": calculate_optimism_index(conn),
            "apathy": detect_apathy_index(conn),
            "top_fears": analyze_society_fears(conn)
        },
        "trends": {
            "emerging": analyze_emerging_topics(conn)
        },
        "recommendations": []
    }

    # Генерация главных рекомендаций
    inv_risk = dashboard['risk_assessment']['investment_risk']['investment_risk']
    optimism = dashboard['society']['optimism']['optimism_index']
    apathy = dashboard['society']['apathy']['apathy_index']

    if inv_risk > 70:
        dashboard['recommendations'].append({
            'priority': 'critical',
            'message': 'Инвестиционный риск критический. Рекомендуется защита активов.'
        })

    if optimism < 30:
        dashboard['recommendations'].append({
            'priority': 'warning',
            'message': 'Общественные настроения крайне пессимистичны. Ожидайте социальной напряженности.'
        })

    if apathy > 0.6:
        dashboard['recommendations'].append({
            'priority': 'info',
            'message': 'Высокий уровень апатии. Общество устало - возможен всплеск после накопления.'
        })

    conn.close()

    dashboard['timestamp'] = datetime.now().isoformat()
    dashboard['status'] = 'operational'

    return dashboard