CREATE TABLE IF NOT EXISTS news (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    url TEXT UNIQUE NOT NULL,
    source VARCHAR(100),
    language VARCHAR(10),
    published_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Расширенный анализ
CREATE TABLE IF NOT EXISTS sentiment_results (
    id SERIAL PRIMARY KEY,
    news_id INTEGER REFERENCES news(id) ON DELETE CASCADE,
    
    -- Базовая тональность
    sentiment VARCHAR(20),
    score FLOAT,
    
    -- Эмоции (JSON: {anger: 0.2, fear: 0.5, joy: 0.1, sadness: 0.2})
    emotions JSONB,
    
    -- Категория новости
    category VARCHAR(50),
    category_confidence FLOAT,
    
    -- Важность (1-10)
    importance_score INTEGER,
    
    -- Ключевые слова
    keywords TEXT[],
    
    -- Сущности (JSON: {countries: [], people: [], companies: []})
    entities JSONB,
    
    -- Индикаторы
    is_fake_probability FLOAT,  -- Вероятность фейка (0-1)
    is_clickbait BOOLEAN,       -- Кликбейт
    fear_index FLOAT,           -- Индекс страха (0-1)
    
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Индексы
CREATE INDEX idx_news_published ON news(published_date);
CREATE INDEX idx_news_language ON news(language);
CREATE INDEX idx_sentiment_news ON sentiment_results(news_id);
CREATE INDEX idx_category ON sentiment_results(category);
CREATE INDEX idx_importance ON sentiment_results(importance_score);

-- Агрегированная статистика (для быстрых запросов)
CREATE TABLE IF NOT EXISTS daily_stats (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE,
    total_news INTEGER,
    fear_index_avg FLOAT,
    fake_news_count INTEGER,
    top_keywords JSONB,
    top_entities JSONB,
    category_distribution JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);