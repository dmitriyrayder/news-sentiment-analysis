import feedparser
import psycopg2
import json
import os
import time
from datetime import datetime
from dateutil import parser as date_parser

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'sentiment_db'),
    'user': os.getenv('DB_USER', 'sentiment_user'),
    'password': os.getenv('DB_PASSWORD', 'password')
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def parse_sources():
    with open('sources.json', 'r') as f:
        sources = json.load(f)['sources']
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    for source in sources:
        print(f"Parsing {source['name']}...")
        try:
            feed = feedparser.parse(source['url'])
            
            for entry in feed.entries[:20]:
                title = entry.get('title', '')
                description = entry.get('summary', entry.get('description', ''))
                url = entry.get('link', '')
                
                published = entry.get('published', entry.get('updated'))
                if published:
                    try:
                        published_date = date_parser.parse(published)
                    except:
                        published_date = datetime.now()
                else:
                    published_date = datetime.now()
                
                try:
                    cursor.execute("""
                        INSERT INTO news (title, description, url, source, language, published_date)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (url) DO NOTHING
                    """, (title, description, url, source['name'], source['lang'], published_date))
                except Exception as e:
                    print(f"Error inserting: {e}")
            
            conn.commit()
            print(f"✓ {source['name']} parsed")
            
        except Exception as e:
            print(f"✗ Error parsing {source['name']}: {e}")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    print("Parser started...")
    while True:
        parse_sources()
        print(f"Waiting 2 hours...")
        time.sleep(7200)