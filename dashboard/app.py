import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import json

st.set_page_config(page_title="Advanced News Analytics", layout="wide", page_icon="📊")

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'sentiment_db'),
    'user': os.getenv('DB_USER', 'sentiment_user'),
    'password': os.getenv('DB_PASSWORD', 'password')
}

def load_data():
    """Загрузка данных с новым подключением каждый раз"""
    conn = psycopg2.connect(**DB_CONFIG)
    
    query = """
        SELECT 
            n.id,
            n.title,
            n.description,
            n.source,
            n.language,
            n.published_date,
            n.url,
            sr.sentiment,
            sr.score,
            sr.emotions,
            sr.category,
            sr.category_confidence,
            sr.importance_score,
            sr.keywords,
            sr.entities,
            sr.is_fake_probability,
            sr.is_clickbait,
            sr.fear_index,
            sr.analyzed_at
        FROM news n
        LEFT JOIN sentiment_results sr ON n.id = sr.news_id
        WHERE n.published_date >= NOW() - INTERVAL '14 days'
        ORDER BY n.published_date DESC
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    return df

# Заголовок
st.title("🧠 Advanced News Analytics Dashboard")
st.markdown("**Интеллектуальный анализ новостей с ML**")
st.markdown("---")

# Загрузка данных
try:
    df = load_data()
except Exception as e:
    st.error(f"Ошибка подключения к БД: {e}")
    st.info("Проверьте что PostgreSQL запущен: `docker-compose ps`")
    st.stop()

# Sidebar фильтры
with st.sidebar:
    st.header("⚙️ Фильтры")
    
    language_filter = st.multiselect(
        "🌍 Язык",
        options=df['language'].unique() if 'language' in df.columns else [],
        default=df['language'].unique() if 'language' in df.columns else []
    )
    
    source_filter = st.multiselect(
        "📰 Источник",
        options=df['source'].unique() if 'source' in df.columns else [],
        default=df['source'].unique() if 'source' in df.columns else []
    )
    
    category_filter = st.multiselect(
        "📁 Категория",
        options=df['category'].dropna().unique() if 'category' in df.columns else [],
        default=df['category'].dropna().unique() if 'category' in df.columns else []
    )
    
    days = st.slider("📅 Последние N дней", 1, 14, 7)
    
    importance_min = st.slider("⭐ Минимальная важность", 1, 10, 1)
    
    show_clickbait = st.checkbox("🎯 Показать кликбейт", value=True)

# Применение фильтров
df_filtered = df.copy()

if language_filter:
    df_filtered = df_filtered[df_filtered['language'].isin(language_filter)]

if source_filter:
    df_filtered = df_filtered[df_filtered['source'].isin(source_filter)]

if 'published_date' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['published_date'] >= datetime.now() - timedelta(days=days)]

if category_filter and 'category' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['category'].isin(category_filter)]

if not show_clickbait and 'is_clickbait' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['is_clickbait'] != True]

if 'importance_score' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['importance_score'] >= importance_min]

# === ГЛАВНЫЕ МЕТРИКИ ===
st.markdown("### 📊 Ключевые показатели")

col1, col2, col3, col4, col5 = st.columns(5)

total_news = len(df_filtered)
analyzed_news = df_filtered['sentiment'].notna().sum() if 'sentiment' in df_filtered.columns else 0

with col1:
    st.metric("📰 Всего новостей", total_news)
    
with col2:
    st.metric("✅ Проанализировано", analyzed_news)

with col3:
    if 'fear_index' in df_filtered.columns:
        avg_fear = df_filtered['fear_index'].mean()
        st.metric(
            "😰 Индекс страха", 
            f"{avg_fear:.2f}" if pd.notna(avg_fear) else "N/A",
            delta=f"{(avg_fear - 0.5):.2f}" if pd.notna(avg_fear) else None,
            delta_color="inverse"
        )
    else:
        st.metric("😰 Индекс страха", "N/A")

with col4:
    if 'is_fake_probability' in df_filtered.columns:
        avg_fake = df_filtered['is_fake_probability'].mean()
        st.metric(
            "🚨 Вероятность фейка", 
            f"{avg_fake*100:.1f}%" if pd.notna(avg_fake) else "N/A",
            delta_color="inverse"
        )
    else:
        st.metric("🚨 Вероятность фейка", "N/A")

with col5:
    if 'is_clickbait' in df_filtered.columns:
        clickbait_count = df_filtered['is_clickbait'].sum()
        st.metric("🎯 Кликбейт", clickbait_count)
    else:
        st.metric("🎯 Кликбейт", "N/A")

st.markdown("---")

# === ИНДЕКС СТРАХА ПО ВРЕМЕНИ ===
if 'fear_index' in df_filtered.columns and 'published_date' in df_filtered.columns:
    st.markdown("### 😰 Динамика индекса страха")
    
    df_fear = df_filtered[df_filtered['fear_index'].notna()].copy()
    if not df_fear.empty:
        df_fear['date'] = pd.to_datetime(df_fear['published_date']).dt.date
        fear_timeline = df_fear.groupby('date')['fear_index'].mean().reset_index()
        
        fig_fear = go.Figure()
        fig_fear.add_trace(go.Scatter(
            x=fear_timeline['date'],
            y=fear_timeline['fear_index'],
            mode='lines+markers',
            fill='tozeroy',
            line=dict(color='#EF553B', width=3),
            marker=dict(size=8)
        ))
        fig_fear.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Норма")
        fig_fear.update_layout(
            height=300,
            yaxis_title="Fear Index",
            xaxis_title="Дата",
            showlegend=False
        )
        st.plotly_chart(fig_fear, use_container_width=True)
    else:
        st.info("Нет данных для отображения индекса страха")

# === ЭМОЦИОНАЛЬНЫЙ АНАЛИЗ ===
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎭 Распределение эмоций")
    
    if 'emotions' in df_filtered.columns:
        emotions_data = []
        for idx, row in df_filtered[df_filtered['emotions'].notna()].iterrows():
            try:
                emotions = json.loads(row['emotions']) if isinstance(row['emotions'], str) else row['emotions']
                if emotions:
                    for emotion, score in emotions.items():
                        emotions_data.append({'emotion': emotion, 'score': score})
            except:
                pass
        
        if emotions_data:
            df_emotions = pd.DataFrame(emotions_data)
            emotion_agg = df_emotions.groupby('emotion')['score'].mean().reset_index()
            
            fig_emotions = px.bar(
                emotion_agg.sort_values('score', ascending=False),
                x='emotion',
                y='score',
                color='score',
                color_continuous_scale='RdYlGn_r'
            )
            fig_emotions.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_emotions, use_container_width=True)
        else:
            st.info("Нет данных об эмоциях (только для английских новостей)")
    else:
        st.info("Колонка эмоций отсутствует")

with col2:
    st.markdown("### 📁 Категории новостей")
    
    if 'category' in df_filtered.columns:
        category_counts = df_filtered['category'].value_counts().reset_index()
        category_counts.columns = ['category', 'count']
        
        if not category_counts.empty:
            fig_cat = px.pie(
                category_counts,
                names='category',
                values='count',
                hole=0.4
            )
            fig_cat.update_layout(height=350)
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("Нет данных о категориях")
    else:
        st.info("Колонка категорий отсутствует")

# === ВАЖНОСТЬ И ФЕЙКИ ===
col1, col2 = st.columns(2)

with col1:
    st.markdown("### ⭐ Распределение важности")
    
    if 'importance_score' in df_filtered.columns:
        importance_dist = df_filtered['importance_score'].value_counts().sort_index().reset_index()
        importance_dist.columns = ['importance', 'count']
        
        if not importance_dist.empty:
            fig_imp = px.bar(
                importance_dist,
                x='importance',
                y='count',
                color='importance',
                color_continuous_scale='Viridis'
            )
            fig_imp.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Колонка важности отсутствует")

with col2:
    st.markdown("### 🚨 Топ источников по вероятности фейка")
    
    if 'is_fake_probability' in df_filtered.columns and 'source' in df_filtered.columns:
        fake_by_source = df_filtered.groupby('source')['is_fake_probability'].mean().sort_values(ascending=False).reset_index()
        fake_by_source.columns = ['source', 'fake_prob']
        
        if not fake_by_source.empty:
            fig_fake = px.bar(
                fake_by_source,
                x='source',
                y='fake_prob',
                color='fake_prob',
                color_continuous_scale='Reds'
            )
            fig_fake.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_fake, use_container_width=True)
    else:
        st.info("Нет данных о фейках")

# === ТАБЛИЦА НОВОСТЕЙ ===
st.markdown("### 📋 Последние новости")

display_cols = ['title', 'source', 'language', 'sentiment', 'published_date']
display_cols = [col for col in display_cols if col in df_filtered.columns]

if display_cols:
    st.dataframe(
        df_filtered[display_cols].head(20),
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("Нет данных для отображения")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"Обновлено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col2:
    if st.button("🔄 Обновить"):
        st.rerun()