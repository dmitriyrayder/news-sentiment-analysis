import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

st.set_page_config(
    page_title="Intelligence News Analytics",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# API Configuration
SENTIMENT_API = os.getenv('SENTIMENT_API', 'http://localhost:8001')

# Custom CSS
st.markdown("""
<style>
    .big-metric { font-size: 2em; font-weight: bold; }
    .critical { color: #FF4B4B; }
    .warning { color: #FFA500; }
    .success { color: #00CC00; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# === ЗАГОЛОВОК ===
st.title("🧠 Intelligence News Analytics Platform")
st.markdown("**Комплексная система анализа новостей и поддержки принятия решений**")
st.markdown("---")

# === SIDEBAR ===
with st.sidebar:
    st.header("⚙️ Навигация")

    page = st.radio(
        "Выберите раздел:",
        ["📊 Главный дашборд",
         "🔥 Восходящие темы",
         "🚨 Детекция пропаганды",
         "👥 Социальный анализ",
         "💼 Бизнес-индикаторы",
         "📈 Волатильность рынка"]
    )

    st.markdown("---")
    st.caption(f"Обновлено: {datetime.now().strftime('%H:%M:%S')}")

    if st.button("🔄 Обновить данные"):
        st.rerun()

# === ФУНКЦИИ ДЛЯ API ===
@st.cache_data(ttl=60)
def get_comprehensive_dashboard():
    try:
        response = requests.get(f"{SENTIMENT_API}/comprehensive-dashboard", timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None

@st.cache_data(ttl=120)
def get_emerging_topics():
    try:
        response = requests.get(f"{SENTIMENT_API}/emerging-topics", timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None

@st.cache_data(ttl=120)
def get_propaganda_detection():
    try:
        response = requests.get(f"{SENTIMENT_API}/propaganda-detection", timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None

@st.cache_data(ttl=120)
def get_society_analysis():
    try:
        response = requests.get(f"{SENTIMENT_API}/society-analysis", timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None

@st.cache_data(ttl=120)
def get_business_intelligence():
    try:
        response = requests.get(f"{SENTIMENT_API}/business-intelligence", timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None

@st.cache_data(ttl=120)
def get_volatility_index():
    try:
        response = requests.get(f"{SENTIMENT_API}/volatility-index", timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None

@st.cache_data(ttl=60)
def get_risk_assessment():
    try:
        response = requests.get(f"{SENTIMENT_API}/risk-assessment", timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None

# ============================================================================
# СТРАНИЦА: ГЛАВНЫЙ ДАШБОРД
# ============================================================================
if page == "📊 Главный дашборд":
    st.header("📊 Комплексный дашборд")

    dashboard_data = get_comprehensive_dashboard()

    if not dashboard_data:
        st.error("❌ Не удалось загрузить данные. Проверьте API.")
        st.stop()

    # === КРИТИЧЕСКИЕ РЕКОМЕНДАЦИИ ===
    if dashboard_data.get('recommendations'):
        st.subheader("⚠️ Критические рекомендации")

        for rec in dashboard_data['recommendations']:
            if rec['priority'] == 'critical':
                st.error(f"🔴 **КРИТИЧНО**: {rec['message']}")
            elif rec['priority'] == 'warning':
                st.warning(f"🟡 **ВНИМАНИЕ**: {rec['message']}")
            else:
                st.info(f"ℹ️ {rec['message']}")

    st.markdown("---")

    # === ГЛАВНЫЕ МЕТРИКИ ===
    col1, col2, col3, col4 = st.columns(4)

    inv_risk = dashboard_data['risk_assessment']['investment_risk']
    optimism = dashboard_data['society']['optimism']
    apathy = dashboard_data['society']['apathy']

    with col1:
        risk_value = inv_risk['investment_risk']
        risk_color = "🔴" if risk_value > 70 else "🟡" if risk_value > 50 else "🟢"
        st.metric(
            "💼 Инвестиционный риск",
            f"{risk_color} {risk_value:.1f}",
            delta=inv_risk['status'].upper()
        )

    with col2:
        opt_value = optimism['optimism_index']
        opt_color = "🟢" if opt_value > 70 else "🟡" if opt_value > 50 else "🔴"
        st.metric(
            "😊 Индекс оптимизма",
            f"{opt_color} {opt_value:.1f}",
            delta=optimism['status']
        )

    with col3:
        apathy_value = apathy['apathy_index'] * 100
        apathy_color = "🔴" if apathy_value > 60 else "🟡" if apathy_value > 30 else "🟢"
        st.metric(
            "😴 Индекс апатии",
            f"{apathy_color} {apathy_value:.1f}%",
            delta=apathy['status']
        )

    with col4:
        fears = dashboard_data['society']['top_fears']
        dominant_fear = list(fears.keys())[0] if fears else "N/A"
        st.metric(
            "😰 Главный страх",
            dominant_fear.upper(),
            delta=f"{fears[dominant_fear]['mention_count']} упоминаний" if fears else None
        )

    st.markdown("---")

    # === ВОСХОДЯЩИЕ ТЕМЫ ===
    st.subheader("🔥 Восходящие темы")

    emerging = dashboard_data['trends']['emerging']

    if emerging['emerging']:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**⬆️ Растущие темы**")
            for topic in emerging['emerging'][:5]:
                growth = topic['growth_rate']
                if growth == 'new':
                    st.success(f"🆕 **{topic['keyword']}** - новая тема ({topic['recent_mentions']} упоминаний)")
                else:
                    st.info(f"📈 **{topic['keyword']}** - рост {growth*100:.0f}% ({topic['recent_mentions']} упоминаний)")

        with col2:
            st.markdown("**⬇️ Угасающие темы**")
            for topic in emerging['declining'][:5]:
                st.warning(f"📉 **{topic['keyword']}** - спад {topic['decline_rate']*100:.0f}%")
    else:
        st.info("Недостаточно данных для анализа трендов")

    st.markdown("---")

    # === СТРАХИ ОБЩЕСТВА ===
    st.subheader("😰 Топ страхов общества")

    if fears:
        fear_df = pd.DataFrame([
            {
                'fear_type': fear_type,
                'mentions': data['mention_count'],
                'fear_index': data['avg_fear_index'],
                'intensity': data['intensity']
            }
            for fear_type, data in fears.items()
        ])

        fig = px.bar(
            fear_df,
            x='fear_type',
            y='mentions',
            color='fear_index',
            color_continuous_scale='Reds',
            title="Распределение страхов по категориям"
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# СТРАНИЦА: ВОСХОДЯЩИЕ ТЕМЫ
# ============================================================================
elif page == "🔥 Восходящие темы":
    st.header("🔥 Восходящие и угасающие темы")
    st.markdown("**Что волнует мир сейчас vs неделю назад**")

    topics_data = get_emerging_topics()

    if not topics_data:
        st.error("❌ Не удалось загрузить данные")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⬆️ Растущие темы")

        emerging = topics_data['emerging_topics']

        if emerging:
            # Таблица
            df_emerging = pd.DataFrame(emerging)

            st.dataframe(
                df_emerging,
                use_container_width=True,
                hide_index=True
            )

            # График
            fig = go.Figure(data=[
                go.Bar(
                    x=[t['keyword'] for t in emerging[:10]],
                    y=[t['recent_mentions'] for t in emerging[:10]],
                    marker_color='lightgreen',
                    text=[f"+{t['growth_rate']*100:.0f}%" if isinstance(t['growth_rate'], float) else t['growth_rate']
                          for t in emerging[:10]],
                    textposition='outside'
                )
            ])
            fig.update_layout(
                title="Топ-10 растущих тем",
                xaxis_title="Тема",
                yaxis_title="Упоминания (последние 2 дня)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Нет данных о растущих темах")

    with col2:
        st.subheader("⬇️ Угасающие темы")

        declining = topics_data['declining_topics']

        if declining:
            # Таблица
            df_declining = pd.DataFrame(declining)

            st.dataframe(
                df_declining,
                use_container_width=True,
                hide_index=True
            )

            # График
            fig = go.Figure(data=[
                go.Bar(
                    x=[t['keyword'] for t in declining[:10]],
                    y=[t['previous_mentions'] for t in declining[:10]],
                    marker_color='lightcoral',
                    text=[f"-{t['decline_rate']*100:.0f}%" for t in declining[:10]],
                    textposition='outside'
                )
            ])
            fig.update_layout(
                title="Топ-10 угасающих тем",
                xaxis_title="Тема",
                yaxis_title="Упоминания (неделю назад)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Нет данных об угасающих темах")

# ============================================================================
# СТРАНИЦА: ДЕТЕКЦИЯ ПРОПАГАНДЫ
# ============================================================================
elif page == "🚨 Детекция пропаганды":
    st.header("🚨 Детекция пропаганды и манипуляций")

    prop_data = get_propaganda_detection()

    if not prop_data:
        st.error("❌ Не удалось загрузить данные")
        st.stop()

    # Метрики
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🔴 Высокий риск", len(prop_data['high_risk_items']))
    with col2:
        st.metric("🟡 Средний риск", len(prop_data['medium_risk_items']))
    with col3:
        st.metric("📊 Всего проверено", prop_data['total_flagged'])

    st.markdown("---")

    # Высокий риск
    if prop_data['high_risk_items']:
        st.subheader("🔴 Новости с высоким риском пропаганды")

        for item in prop_data['high_risk_items']:
            with st.expander(f"⚠️ {item['title'][:100]}..."):
                st.markdown(f"**Источник:** {item['source']}")
                st.markdown(f"**Propaganda Score:** {item['propaganda_score']:.2f}")
                st.markdown(f"**Флаги:**")
                for flag in item['flags']:
                    if flag == 'emotional_manipulation':
                        st.error("🎭 Эмоциональная манипуляция")
                    elif flag == 'absolute_statements':
                        st.warning("❗ Абсолютные утверждения")
                    elif flag == 'us_vs_them':
                        st.warning("👥 Дихотомия 'нас vs них'")
                    elif flag == 'call_to_action':
                        st.info("📢 Призыв к действию")
                    elif flag == 'excessive_exclamation':
                        st.info("‼️ Избыток восклицаний")
    else:
        st.success("✅ Новостей с высоким риском пропаганды не обнаружено")

    # Средний риск
    if prop_data['medium_risk_items']:
        st.subheader("🟡 Новости со средним риском")

        for item in prop_data['medium_risk_items'][:10]:
            st.warning(f"📰 {item['title'][:100]}... (Score: {item['propaganda_score']:.2f})")

# ============================================================================
# СТРАНИЦА: СОЦИАЛЬНЫЙ АНАЛИЗ
# ============================================================================
elif page == "👥 Социальный анализ":
    st.header("👥 Анализ настроений общества")

    society_data = get_society_analysis()

    if not society_data:
        st.error("❌ Не удалось загрузить данные")
        st.stop()

    # === ГЛАВНЫЕ МЕТРИКИ ===
    col1, col2, col3 = st.columns(3)

    optimism = society_data['optimism']
    apathy = society_data['apathy']

    with col1:
        opt_val = optimism['optimism_index']
        color = "success" if opt_val > 70 else "warning" if opt_val > 50 else "error"
        st.metric(
            "😊 Индекс оптимизма",
            f"{opt_val:.1f}",
            delta=optimism['status']
        )

        # Разбивка sentiment
        st.markdown("**Разбивка настроений:**")
        breakdown = optimism['sentiment_breakdown']
        st.progress(breakdown['positive']/100, text=f"Позитив: {breakdown['positive']}%")
        st.progress(breakdown['negative']/100, text=f"Негатив: {breakdown['negative']}%")
        st.progress(breakdown['neutral']/100, text=f"Нейтрально: {breakdown['neutral']}%")

    with col2:
        apathy_val = apathy['apathy_index'] * 100
        st.metric(
            "😴 Индекс апатии",
            f"{apathy_val:.1f}%",
            delta=apathy['status']
        )

        if apathy['indicators']:
            st.markdown("**Индикаторы:**")
            for indicator in apathy['indicators']:
                if indicator == 'declining_importance':
                    st.warning("📉 Снижение важности новостей")
                elif indicator == 'increasing_neutrality':
                    st.info("😐 Рост нейтральности")
                elif indicator == 'high_neutrality':
                    st.info("😶 Высокая нейтральность")

    with col3:
        fears = society_data['fears']
        if fears:
            dominant = list(fears.keys())[0]
            st.metric(
                "😰 Главный страх",
                dominant.upper(),
                delta=f"{fears[dominant]['mention_count']} упоминаний"
            )

    st.markdown("---")

    # === СТРАХИ ===
    st.subheader("😰 Анализ страхов общества")

    if fears:
        fear_df = pd.DataFrame([
            {
                'Категория': fear_type.upper(),
                'Упоминания': data['mention_count'],
                'Индекс страха': data['avg_fear_index'],
                'Важность': data['avg_importance'],
                'Интенсивность': data['intensity']
            }
            for fear_type, data in fears.items()
        ])

        # Таблица
        st.dataframe(fear_df, use_container_width=True, hide_index=True)

        # График
        fig = px.bar(
            fear_df,
            x='Категория',
            y='Упоминания',
            color='Индекс страха',
            color_continuous_scale='Reds',
            title="Распределение страхов по категориям"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Радар-чарт
        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=[data['avg_fear_index'] for data in fears.values()],
            theta=[fear_type.upper() for fear_type in fears.keys()],
            fill='toself',
            name='Fear Index'
        ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            title="Радар страхов"
        )

        st.plotly_chart(fig_radar, use_container_width=True)

# ============================================================================
# СТРАНИЦА: БИЗНЕС-ИНДИКАТОРЫ
# ============================================================================
elif page == "💼 Бизнес-индикаторы":
    st.header("💼 Бизнес-индикаторы для принятия решений")

    business_data = get_business_intelligence()

    if not business_data:
        st.error("❌ Не удалось загрузить данные")
        st.stop()

    # === ИНВЕСТИЦИОННЫЙ РИСК ===
    inv_risk = business_data['investment_risk']

    st.subheader("📊 Инвестиционный риск")

    col1, col2 = st.columns([2, 1])

    with col1:
        risk_val = inv_risk['investment_risk']

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_val,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Investment Risk Index"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if risk_val > 70 else "orange" if risk_val > 50 else "lightgreen"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 50], 'color': "gray"},
                    {'range': [50, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"### Статус: {inv_risk['status'].upper()}")

        st.markdown("**Рекомендация:**")
        st.markdown(f"*{business_data['recommendation']}*")

        st.markdown("---")

        st.markdown("**Факторы риска:**")
        factors = inv_risk['factors']
        st.metric("😰 Уровень страха", f"{factors['fear_level']:.3f}")
        st.metric("😞 Негативный sentiment", f"{factors['negative_sentiment']:.1f}%")
        st.metric("🎭 Дезинформация", f"{factors['disinformation_level']:.3f}")

    st.markdown("---")

    # === ВОЛАТИЛЬНОСТЬ ===
    st.subheader("📈 Волатильность по категориям")

    volatility = business_data['market_volatility']

    if volatility:
        vol_df = pd.DataFrame([
            {
                'Категория': cat,
                'Волатильность': data['volatility'],
                'Стабильность': data['stability']
            }
            for cat, data in volatility.items()
        ])

        fig = px.bar(
            vol_df,
            x='Категория',
            y='Волатильность',
            color='Волатильность',
            color_continuous_scale='RdYlGn_r',
            title="Индекс волатильности по категориям"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Таблица
        st.dataframe(vol_df, use_container_width=True, hide_index=True)

# ============================================================================
# СТРАНИЦА: ВОЛАТИЛЬНОСТЬ РЫНКА
# ============================================================================
elif page == "📈 Волатильность рынка":
    st.header("📈 Индекс волатильности информационной среды")

    vol_data = get_volatility_index()

    if not vol_data:
        st.error("❌ Не удалось загрузить данные")
        st.stop()

    # Общая волатильность
    overall = vol_data['overall_volatility']
    status = vol_data['status']

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(
            "🌊 Общая волатильность",
            f"{overall:.3f}",
            delta=status.upper()
        )

        st.markdown("---")

        st.markdown(f"**Интерпретация:**")
        st.info(vol_data['interpretation'])

    with col2:
        # График волатильности
        categories = vol_data['categories']

        if categories:
            vol_df = pd.DataFrame(categories)

            fig = px.bar(
                vol_df,
                x='category',
                y='volatility',
                color='volatility',
                color_continuous_scale='RdYlGn_r',
                title="Волатильность по категориям"
            )

            fig.add_hline(
                y=0.5,
                line_dash="dash",
                line_color="red",
                annotation_text="Критический уровень"
            )

            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Детальная таблица
    if categories:
        st.subheader("📊 Детальный анализ")

        vol_df = pd.DataFrame(categories)

        # Раскраска по статусу
        def color_status(val):
            if val == 'highly_volatile':
                return 'background-color: #ffcccc'
            elif val == 'volatile':
                return 'background-color: #ffffcc'
            else:
                return 'background-color: #ccffcc'

        st.dataframe(
            vol_df.style.applymap(color_status, subset=['status']),
            use_container_width=True,
            hide_index=True
        )

# === FOOTER ===
st.markdown("---")
st.caption(f"🧠 Intelligence News Analytics Platform | Обновлено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
