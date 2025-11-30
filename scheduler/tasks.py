import time
import requests
import psycopg2
import os
from datetime import datetime, timedelta

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'sentiment_db'),
    'user': os.getenv('DB_USER', 'sentiment_user'),
    'password': os.getenv('DB_PASSWORD', 'password')
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def cleanup_old_news():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        DELETE FROM news 
        WHERE published_date < NOW() - INTERVAL '14 days'
    """)
    
    deleted = cursor.rowcount
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"Cleaned up {deleted} old news entries")
    return deleted

def trigger_sentiment_analysis():
    try:
        response = requests.post("http://sentiment:8000/analyze", timeout=300)
        result = response.json()
        print(f"Sentiment analysis completed: {result}")
        return result
    except Exception as e:
        print(f"Error triggering sentiment analysis: {e}")
        return None

def main():
    print("Scheduler started...")
    
    time.sleep(30)
    
    last_cleanup = datetime.now()
    
    while True:
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running scheduled tasks...")
            
            trigger_sentiment_analysis()
            
            if datetime.now() - last_cleanup > timedelta(days=1):
                cleanup_old_news()
                last_cleanup = datetime.now()
            
            print("Tasks completed. Waiting 10 minutes...")
            time.sleep(600)
            
        except Exception as e:
            print(f"Error in scheduler: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()