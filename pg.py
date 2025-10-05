import os
import re
import time
import requests
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import hashlib

HEADERS = {"User-Agent": "Mozilla/5.0"}

# Neon connection string
DATABASE_URL = "postgresql://neondb_owner:npg_cZJvwbxs23YS@ep-flat-term-a1bhljd0-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

def fix_database_schema():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        c = conn.cursor()
        
        print("🔍 Checking existing tables...")
        
        # Check what tables exist
        c.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = c.fetchall()
        print(f"📋 Existing tables: {[t[0] for t in tables]}")
        
        # Check papers table structure
        if ('papers',) in tables:
            c.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'papers'
            """)
            columns = c.fetchall()
            print(f"📊 Papers table columns: {columns}")
            
            # Drop and recreate if structure is wrong
            print("🔄 Dropping existing papers table...")
            c.execute("DROP TABLE IF EXISTS paper_chunks CASCADE")
            c.execute("DROP TABLE IF EXISTS papers CASCADE")
            conn.commit()
        
        # Create papers table with correct structure
        print("✅ Creating papers table...")
        c.execute('''
        CREATE TABLE papers (
            pmc_id VARCHAR(50) PRIMARY KEY,
            title TEXT,
            text_chunks_count INTEGER DEFAULT 0,
            download_status VARCHAR(20) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create paper_chunks table
        print("✅ Creating paper_chunks table...")
        c.execute('''
        CREATE TABLE paper_chunks (
            id SERIAL PRIMARY KEY,
            pmc_id VARCHAR(50) REFERENCES papers(pmc_id) ON DELETE CASCADE,
            chunk_index INTEGER,
            chunk_text TEXT,
            chunk_hash VARCHAR(64),
            section_type VARCHAR(50),
            word_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(pmc_id, chunk_index)
        )
        ''')
        
        # Create indexes
        print("✅ Creating indexes...")
        c.execute('CREATE INDEX idx_paper_chunks_pmc_id ON paper_chunks(pmc_id)')
        c.execute('CREATE INDEX idx_paper_chunks_hash ON paper_chunks(chunk_hash)')
        
        conn.commit()
        conn.close()
        print("🎉 Database schema fixed successfully!")
        
    except Exception as e:
        print(f"❌ Error fixing database: {e}")

class PaperDownloaderNeon:
    def __init__(self, database_url, base_dir=r"C:\pdfparsing", chunk_size=1000):
        self.database_url = database_url
        self.base_dir = base_dir
        self.chunk_size = chunk_size  # Characters per chunk
        self.parsed_url = urlparse(database_url)
        self.setup_database()
    
    def get_connection(self):
        """Create a new database connection"""
        return psycopg2.connect(
            host=self.parsed_url.hostname,
            database=self.parsed_url.path[1:],
            user=self.parsed_url.username,
            password=self.parsed_url.password,
            port=self.parsed_url.port or 5432,
            sslmode='require'
        )
    
    def setup_database(self):
        """Initialize the database and create tables if they don't exist"""
        try:
            conn = self.get_connection()
            c = conn.cursor()
            
            # Papers table
            c.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                pmc_id VARCHAR(50) PRIMARY KEY,
                title TEXT,
                text_chunks_count INTEGER DEFAULT 0,
                download_status VARCHAR(20) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Text chunks table
            c.execute('''
            CREATE TABLE IF NOT EXISTS paper_chunks (
                id SERIAL PRIMARY KEY,
                pmc_id VARCHAR(50) REFERENCES papers(pmc_id) ON DELETE CASCADE,
                chunk_index INTEGER,
                chunk_text TEXT,
                chunk_hash VARCHAR(64),
                section_type VARCHAR(50),
                word_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(pmc_id, chunk_index)
            )
            ''')
            
            # Index for better search performance
            c.execute('''
            CREATE INDEX IF NOT EXISTS idx_paper_chunks_pmc_id ON paper_chunks(pmc_id);
            CREATE INDEX IF NOT EXISTS idx_paper_chunks_hash ON paper_chunks(chunk_hash);
            ''')
            
            conn.commit()
            conn.close()
            print("✅ Database initialized successfully")
        except psycopg2.Error as e:
            print(f"❌ Database setup error: {e}")
    
    def chunk_text(self, text, chunk_size=None):
        """Split text into chunks of specified size"""
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        # Simple chunking by character count with sentence boundaries
        chunks = []
        sentences = text.split('. ')
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 2 <= chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def get_text_hash(self, text):
        """Generate hash for chunk deduplication"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def store_text_chunks(self, pmcid, text, section_type="main"):
        """Store text chunks in database"""
        if not text or not text.strip():
            return 0
            
        chunks = self.chunk_text(text)
        stored_count = 0
        
        try:
            conn = self.get_connection()
            c = conn.cursor()
            
            # Ensure paper record exists before storing chunks
            c.execute('''
                INSERT INTO papers (pmc_id, download_status)
                VALUES (%s, %s)
                ON CONFLICT (pmc_id) DO NOTHING
            ''', (pmcid, 'processing'))
            
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                    
                chunk_hash = self.get_text_hash(chunk)
                word_count = len(chunk.split())
                
                c.execute('''
                    INSERT INTO paper_chunks 
                    (pmc_id, chunk_index, chunk_text, chunk_hash, section_type, word_count)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (pmc_id, chunk_index)
                    DO UPDATE SET
                        chunk_text = EXCLUDED.chunk_text,
                        chunk_hash = EXCLUDED.chunk_hash,
                        section_type = EXCLUDED.section_type,
                        word_count = EXCLUDED.word_count
                ''', (pmcid, i, chunk, chunk_hash, section_type, word_count))
                
                stored_count += 1
            
            conn.commit()
            conn.close()
            print(f"✅ Stored {stored_count} text chunks for {pmcid}")
            return stored_count
            
        except psycopg2.Error as e:
            print(f"❌ Error storing text chunks: {e}")
            return 0
    
    def extract_text_content(self, soup):
        """Extract structured text content from BeautifulSoup object"""
        sections = {}
        
        # Try to extract different sections
        # Abstract
        abstract = soup.find('div', {'class': 'abstract'}) or soup.find('section', {'class': 'abstract'})
        if abstract:
            sections['abstract'] = abstract.get_text(strip=True)
        
        # Introduction
        intro = soup.find('section', string=re.compile(r'Introduction', re.I)) or \
                soup.find('h2', string=re.compile(r'Introduction', re.I))
        if intro:
            # Get text from this section
            sections['introduction'] = self.get_section_text(intro)
        
        # Methods
        methods = soup.find('section', string=re.compile(r'Method|Material', re.I)) or \
                  soup.find('h2', string=re.compile(r'Method|Material', re.I))
        if methods:
            sections['methods'] = self.get_section_text(methods)
        
        # Results
        results = soup.find('section', string=re.compile(r'Result', re.I)) or \
                  soup.find('h2', string=re.compile(r'Result', re.I))
        if results:
            sections['results'] = self.get_section_text(results)
        
        # Discussion
        discussion = soup.find('section', string=re.compile(r'Discussion|Conclusion', re.I)) or \
                     soup.find('h2', string=re.compile(r'Discussion|Conclusion', re.I))
        if discussion:
            sections['discussion'] = self.get_section_text(discussion)
        
        # If no structured sections found, get main content
        if not sections:
            article_body = soup.find('div', {'class': 'article-body'}) or \
                          soup.find('div', {'class': 'content'}) or \
                          soup.find('article') or \
                          soup.find('main')
            
            if article_body:
                main_text = ""
                for element in article_body.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    text = element.get_text(strip=True)
                    if text:
                        main_text += text + "\n\n"
                sections['main'] = main_text
        
        return sections
    
    def get_section_text(self, element):
        """Extract text from a section element"""
        if not element:
            return ""
        
        # Find the parent section or get next siblings until next heading
        text = ""
        current = element.parent if element.parent else element
        
        for elem in current.find_all(['p', 'div']):
            text += elem.get_text(strip=True) + "\n\n"
        
        return text
    
    def check_if_exists(self, pmcid):
        """Check if PMC ID already exists in database"""
        try:
            conn = self.get_connection()
            c = conn.cursor()
            c.execute('SELECT pmc_id, download_status FROM papers WHERE pmc_id = %s', (pmcid,))
            result = c.fetchone()
            conn.close()
            if result:
                return True, result[1]
            return False, None
        except psycopg2.Error as e:
            print(f"❌ Database check error: {e}")
            return False, None
    
    def update_database(self, pmcid, title="", text_chunks_count=0, status='completed'):
        """Update database with paper information"""
        try:
            conn = self.get_connection()
            c = conn.cursor()
            c.execute('''
                INSERT INTO papers 
                (pmc_id, title, text_chunks_count, download_status)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (pmc_id) 
                DO UPDATE SET 
                    title = EXCLUDED.title,
                    text_chunks_count = EXCLUDED.text_chunks_count,
                    download_status = EXCLUDED.download_status,
                    updated_at = CURRENT_TIMESTAMP
            ''', (pmcid, title, text_chunks_count, status))
            conn.commit()
            conn.close()
        except psycopg2.Error as e:
            print(f"❌ Database update error: {e}")
    
    def mark_as_failed(self, pmcid, reason="download_failed"):
        """Mark a paper as failed in database"""
        try:
            conn = self.get_connection()
            c = conn.cursor()
            c.execute('''
                INSERT INTO papers (pmc_id, download_status)
                VALUES (%s, %s)
                ON CONFLICT (pmc_id)
                DO UPDATE SET 
                    download_status = EXCLUDED.download_status,
                    updated_at = CURRENT_TIMESTAMP
            ''', (pmcid, reason))
            conn.commit()
            conn.close()
        except psycopg2.Error as e:
            print(f"❌ Database error marking as failed: {e}")
    
    def extract_pmcid(self, url):
        """Extract PMC ID from URL"""
        match = re.search(r'(PMC\d+)', url)
        return match.group(1) if match else None
    
    def fetch_html(self, url):
        """Fetch HTML content from URL"""
        try:
            res = requests.get(url, headers=HEADERS, timeout=15)
            res.raise_for_status()
            return res.text
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to fetch HTML for {url}: {e}")
            return None
    
    def download_paper(self, pmcid, html_text):
        """Extract and store text for a paper"""
        soup = BeautifulSoup(html_text, "html.parser")

        # Extract title
        title_elem = soup.find('title') or soup.find('h1')
        title = title_elem.get_text(strip=True) if title_elem else ""

        # Extract and store text content in chunks
        text_sections = self.extract_text_content(soup)
        total_chunks = 0
        
        for section_type, text in text_sections.items():
            if text:
                chunk_count = self.store_text_chunks(pmcid, text, section_type)
                total_chunks += chunk_count

        self.update_database(pmcid, title, total_chunks)
        print(f"📊 Updated database for {pmcid} ({total_chunks} text chunks)")
        
        return total_chunks
    
    def search_text_chunks(self, query, limit=10):
        """Search text chunks by content"""
        try:
            conn = self.get_connection()
            c = conn.cursor(cursor_factory=RealDictCursor)
            c.execute('''
                SELECT p.pmc_id, p.title, pc.chunk_text, pc.section_type, pc.word_count
                FROM paper_chunks pc
                JOIN papers p ON pc.pmc_id = p.pmc_id
                WHERE pc.chunk_text ILIKE %s
                ORDER BY pc.word_count DESC
                LIMIT %s
            ''', (f'%{query}%', limit))
            results = c.fetchall()
            conn.close()
            return results
        except psycopg2.Error as e:
            print(f"❌ Search error: {e}")
            return []
    
    def get_paper_chunks(self, pmcid):
        """Get all chunks for a specific paper"""
        try:
            conn = self.get_connection()
            c = conn.cursor(cursor_factory=RealDictCursor)
            c.execute('''
                SELECT chunk_index, chunk_text, section_type, word_count
                FROM paper_chunks
                WHERE pmc_id = %s
                ORDER BY chunk_index
            ''', (pmcid,))
            results = c.fetchall()
            conn.close()
            return results
        except psycopg2.Error as e:
            print(f"❌ Error getting chunks: {e}")
            return []
    
    def process_csv(self, csv_file, skip_existing=True, skip_failed=True):
        """Process all papers in CSV file"""
        if not os.path.exists(csv_file):
            print(f"❌ CSV file not found: {csv_file}")
            return
        
        df = pd.read_csv(csv_file)
        processed = 0
        skipped = 0
        failed = 0
        
        print(f"🚀 Starting to process {len(df)} papers...")
        
        for idx, row in df.iterrows():
            link = row.get("Link", "")
            print(f"\n📄 Processing article {idx+1}/{len(df)}: {link}")
            
            pmcid = self.extract_pmcid(link)
            if not pmcid:
                print("⚠ No PMC ID found in link")
                continue
            
            # Check if already exists
            exists, status = self.check_if_exists(pmcid)
            if exists:
                if skip_existing and status == 'completed':
                    print(f"⏭ Skipping {pmcid} - already completed")
                    skipped += 1
                    continue
                elif skip_failed and status in ['download_failed', 'fetch_failed']:
                    print(f"⏭ Skipping {pmcid} - previously failed ({status})")
                    skipped += 1
                    continue

            # Fetch HTML
            html_text = self.fetch_html(link)
            if not html_text:
                self.mark_as_failed(pmcid, "fetch_failed")
                failed += 1
                continue
            
            # Download paper content
            try:
                self.download_paper(pmcid, html_text)
                processed += 1
            except Exception as e:
                print(f"❌ Error downloading {pmcid}: {e}")
                self.mark_as_failed(pmcid, "download_failed")
                failed += 1

            # Be respectful to the server
            time.sleep(1)
        
        print(f"\n🎉 Processing complete!")
        print(f"📊 Processed: {processed} papers")
        print(f"⏭ Skipped: {skipped} papers")
        print(f"❌ Failed: {failed} papers")
        
        # Show final database stats
        self.show_stats()
    
    def show_stats(self):
        """Display database statistics"""
        try:
            conn = self.get_connection()
            c = conn.cursor(cursor_factory=RealDictCursor)
            
            c.execute("SELECT COUNT(*) as total FROM papers")
            total = c.fetchone()['total']
            
            c.execute("SELECT download_status, COUNT(*) as count FROM papers GROUP BY download_status")
            status_counts = c.fetchall()
            
            c.execute('''
                SELECT 
                    SUM(text_chunks_count) as total_chunks
                FROM papers WHERE download_status = 'completed'
            ''')
            content_stats = c.fetchone()
            
            c.execute("SELECT COUNT(*) as total_chunks FROM paper_chunks")
            chunk_count = c.fetchone()['total_chunks']
            
            conn.close()
            
            print(f"\n📊 Database Statistics:")
            print(f"   Total papers: {total}")
            for status in status_counts:
                print(f"   {status['download_status']}: {status['count']}")
            
            if content_stats:
                print(f"\n📁 Content Statistics:")
                print(f"   Total text chunks: {chunk_count or 0}")
                
        except psycopg2.Error as e:
            print(f"❌ Error getting stats: {e}")

# Usage
if __name__ == "__main__":
    fix_database_schema() 
    downloader = PaperDownloaderNeon(DATABASE_URL, chunk_size=800)
    
    # Process CSV
    print("📋 Processing CSV file...")
    downloader.process_csv("SB_publication_PMC.csv")
    
    # Example: Search for specific content
    print("\n🔍 Example search:")
    results = downloader.search_text_chunks("microgravity", limit=5)
    for result in results:
        print(f"Paper: {result['pmc_id']} - {result['section_type']}")
        print(f"Text: {result['chunk_text'][:200]}...")
        print("---")