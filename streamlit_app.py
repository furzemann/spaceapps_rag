import streamlit as st
import os
import psycopg2
from psycopg2.extras import execute_values, Json
from psycopg2.extensions import register_adapter
import hashlib
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document
import numpy as np
from pathlib import Path
from collections import deque

# Register dict to JSON adapter globally
register_adapter(dict, Json)

# Set page config
st.set_page_config(
    page_title="Biology RAG Chatbot",
    page_icon="🧬",
    layout="wide"
)

# Database Configuration
NEON_CONNECTION_STRING = "YOUR_DATABASE_URL"
EXISTING_TABLE_NAME = "paper_chunks"
TEXT_COLUMN_NAME = "chunk_text"
ID_COLUMN_NAME = "id"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = False

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

class NeonVectorStore:
    """Persistent vector store using Neon PostgreSQL with pgvector"""

    def __init__(self, connection_string, embedding_model, source_table=None, text_column=None, id_column=None):
        """Initialize vector store with database connection"""
        self.connection_string = connection_string
        self.embedding_model = embedding_model
        self.source_table = source_table
        self.text_column = text_column
        self.id_column = id_column

        # Establish database connection
        self.conn = psycopg2.connect(connection_string)
        
        # Enable pgvector extension and create tables
        self._enable_pgvector()
        self._create_embeddings_table()

    def _enable_pgvector(self):
        """Enable pgvector extension in the database"""
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.conn.commit()

    def _create_embeddings_table(self):
        """Create embeddings table if it doesn't exist"""
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS document_embeddings (
            id SERIAL PRIMARY KEY,
            source_id TEXT,
            content_hash TEXT UNIQUE,
            page_content TEXT NOT NULL,
            metadata JSONB DEFAULT '{{}}'::jsonb,
            embedding vector({embedding_dim}),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

        with self.conn.cursor() as cur:
            cur.execute(create_table_sql)
            self.conn.commit()

    def _generate_content_hash(self, content):
        """Generate SHA256 hash for content deduplication"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def fetch_strings_from_existing_database(self, limit=None):
        """Fetch existing strings from your Neon PostgreSQL database"""
        if not self.source_table or not self.text_column:
            raise ValueError("source_table and text_column must be specified")

        with self.conn.cursor() as cur:
            query = f"""
                SELECT {self.id_column}, {self.text_column}
                FROM {self.source_table}
                WHERE {self.text_column} IS NOT NULL
                AND LENGTH(TRIM({self.text_column})) > 0
            """

            if limit:
                query += f" LIMIT {limit}"

            cur.execute(query)
            results = cur.fetchall()

        return results

    def check_if_exists(self, content_hash):
        """Check if document already has embeddings"""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM document_embeddings WHERE content_hash = %s)",
                (content_hash,)
            )
            return cur.fetchone()[0]

    def convert_strings_to_embeddings(self, batch_size=50):
        """Convert strings from database to embeddings"""
        string_data = self.fetch_strings_from_existing_database()

        if not string_data:
            return 0

        total_strings = len(string_data)
        new_embeddings = 0
        skipped = 0

        for batch_start in range(0, total_strings, batch_size):
            batch_end = min(batch_start + batch_size, total_strings)
            batch = string_data[batch_start:batch_end]

            batch_values = []

            for source_id, text_content in batch:
                content_hash = self._generate_content_hash(text_content)

                if self.check_if_exists(content_hash):
                    skipped += 1
                    continue

                embedding = self.embedding_model.encode(text_content).tolist()

                batch_values.append((
                    str(source_id),
                    content_hash,
                    text_content,
                    Json({}),
                    embedding
                ))

            if batch_values:
                with self.conn.cursor() as cur:
                    execute_values(
                        cur,
                        """
                        INSERT INTO document_embeddings
                        (source_id, content_hash, page_content, metadata, embedding)
                        VALUES %s
                        ON CONFLICT (content_hash) DO NOTHING
                        """,
                        batch_values,
                        template="(%s, %s, %s, %s, %s::vector)"
                    )
                    self.conn.commit()

                new_embeddings += len(batch_values)

        return new_embeddings

    def get_document_count(self):
        """Get total number of embeddings in database"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM document_embeddings")
            return cur.fetchone()[0]

    def create_index(self, index_type="hnsw", distance_metric="cosine"):
        """Create vector index for faster similarity search"""
        metric_ops = {
            "cosine": "vector_cosine_ops",
            "l2": "vector_l2_ops",
            "ip": "vector_ip_ops"
        }

        ops = metric_ops.get(distance_metric, "vector_cosine_ops")
        index_name = "document_embeddings_embedding_idx"

        with self.conn.cursor() as cur:
            cur.execute(f"DROP INDEX IF EXISTS {index_name}")
            self.conn.commit()

        if index_type == "hnsw":
            create_index_sql = f"""
            CREATE INDEX {index_name}
            ON document_embeddings
            USING hnsw (embedding {ops})
            WITH (m = 16, ef_construction = 64)
            """
        else:
            create_index_sql = f"""
            CREATE INDEX {index_name}
            ON document_embeddings
            USING ivfflat (embedding {ops})
            WITH (lists = 100)
            """

        with self.conn.cursor() as cur:
            cur.execute(create_index_sql)
            self.conn.commit()

    def similarity_search(self, query, k=4):
        """Perform similarity search using cosine distance"""
        query_embedding = self.embedding_model.encode(query).tolist()

        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, page_content, metadata,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM document_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, k))

            results = cur.fetchall()
            return [Document(page_content=row[1], metadata=row[2] or {}) for row in results]

    def mmr_search(self, query, k=4, lambda_mult=0.5, fetch_k=20):
        """Maximal Marginal Relevance search for diversity"""
        query_embedding = self.embedding_model.encode(query).tolist()

        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, page_content, metadata, embedding,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM document_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, fetch_k))

            candidates = cur.fetchall()

        if not candidates:
            return []

        selected_indices = []
        candidate_embeddings = np.array([c[3] for c in candidates])
        query_emb = np.array(query_embedding)

        selected_indices.append(0)

        while len(selected_indices) < k and len(selected_indices) < len(candidates):
            best_score = -float('inf')
            best_idx = -1

            for i in range(len(candidates)):
                if i in selected_indices:
                    continue

                relevance = 1 - np.linalg.norm(query_emb - candidate_embeddings[i])
                diversity = min([
                    1 - np.linalg.norm(candidate_embeddings[i] - candidate_embeddings[j])
                    for j in selected_indices
                ])

                mmr_score = lambda_mult * relevance + (1 - lambda_mult) * diversity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx != -1:
                selected_indices.append(best_idx)

        return [Document(page_content=candidates[i][1], metadata=candidates[i][2] or {})
                for i in selected_indices]

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    hf_token = st.text_input("Hugging Face Token", type="password", key="hf_token")
    
    style = st.selectbox("Response Style", ["simple", "detailed", "technical"], index=0)
    level = st.selectbox("User Level", ["General", "Researcher"], index=0)
    max_token = st.slider("Max Tokens", 50, 1000, 512)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1)
    search_k = st.slider("Search Results", 1, 10, 2)
    lambda_mult = st.slider("MMR Lambda (Diversity)", 0.0, 1.0, 0.5)
    
    if st.button("Initialize RAG System"):
        if hf_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
            
            with st.spinner("Loading embedding model..."):
                try:
                    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
                    
                    vector_store = NeonVectorStore(
                        connection_string=NEON_CONNECTION_STRING,
                        embedding_model=embedding_model,
                        source_table=EXISTING_TABLE_NAME,
                        text_column=TEXT_COLUMN_NAME,
                        id_column=ID_COLUMN_NAME
                    )
                    
                    existing_count = vector_store.get_document_count()
                    
                    if existing_count == 0:
                        st.info("Converting text chunks to embeddings...")
                        new_count = vector_store.convert_strings_to_embeddings(batch_size=50)
                        if new_count > 0:
                            vector_store.create_index(index_type="hnsw", distance_metric="cosine")
                            st.success(f"Created {new_count} embeddings and index!")
                    else:
                        st.success(f"Using existing {existing_count} embeddings")
                    
                    st.session_state.vector_store = vector_store
                    st.session_state.rag_initialized = True
                    
                except Exception as e:
                    st.error(f"Error initializing RAG system: {str(e)}")
        else:
            st.error("Please enter your Hugging Face token")

# Main content
st.title("🧬 Biology RAG Chatbot")
st.markdown("Ask questions about space biology experiments!")

def generate_response(question, style, level, max_token, temperature, search_k, lambda_mult):
    try:
        vector_store = st.session_state.vector_store
        
        # Get relevant context based on user level
        if level == 'General':
            result_docs = vector_store.similarity_search(question, k=search_k)
        else:  # Researcher
            result_docs = vector_store.mmr_search(question, k=search_k, lambda_mult=lambda_mult)
        
        content_only = [doc.page_content for doc in result_docs]
        required_text_for_query = "\n".join(content_only)
        
        # Initialize LLM
        llm_final = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.2-3B-Instruct",
            task="text-generation",
            temperature=temperature,
        )
        model = ChatHuggingFace(llm=llm_final)
        
        # Load chat history
        chat = []
        chat_file = Path('chat.txt')
        if chat_file.exists():
            with open(chat_file) as f:
                chat.extend(f.readlines())
        
        # Create chat template
        chat_template = ChatPromptTemplate([
            ('system', """You are a biology rag chatbot, you will be given a context and a question from that context and you have to answer that question with {style}, and considering him as {level}.
Only give answer the question when you are sure of it and it is present in Context somewhere.
Also consider the max token limit to be {max_token}, do not go over this token limit.

Context regarding query:
{required_text_for_query}

Explain the following query - """),
            MessagesPlaceholder(variable_name='chat'),
            ('human', "{query} with {style} and assuming user to be {level}")
        ])
        
        prompts = chat_template.invoke({
            'chat': chat,
            'required_text_for_query': required_text_for_query,
            'query': question,
            'style': style,
            'level': level,
            'max_token': max_token
        })
        
        result = model.invoke(prompts)
        
        # Save to chat history
        def append_message(result_content: str, query: str):
            MAX_LINES = 10
            DATA = Path("chat.txt")
            COUNT = Path("chat.count")
            
            line_query = f'HumanMessage(content={{{query}}})\n'
            line = f'AIMessage(content="{{{result_content}}}")\n'

            if DATA.exists():
                dq = deque(DATA.read_text(encoding="utf-8").splitlines(keepends=True), maxlen=MAX_LINES)
            else:
                dq = deque(maxlen=MAX_LINES)

            dq.append(line_query)
            dq.append(line)
            DATA.write_text("".join(dq), encoding="utf-8")

            total = int(COUNT.read_text()) if COUNT.exists() else 0
            total += 2
            COUNT.write_text(str(total))
        
        append_message(result.content, question)
        return result.content
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Chat interface
if st.session_state.rag_initialized:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    prompt = st.chat_input("Ask about space biology experiments...")
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base and generating response..."):
                response = generate_response(prompt, style, level, max_token, temperature, search_k, lambda_mult)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        if Path("chat.txt").exists():
            Path("chat.txt").unlink()
        if Path("chat.count").exists():
            Path("chat.count").unlink()
        st.rerun()

else:
    st.info("Please initialize the RAG system using the sidebar configuration.")
    st.markdown("""
    ### Getting Started:
    1. Enter your Hugging Face API token in the sidebar
    2. Configure your preferred settings (user level determines search strategy)
    3. Click "Initialize RAG System" 
    4. Start asking questions about space biology experiments!
    
    ### User Levels:
    - **General**: Uses similarity search for straightforward answers
    - **Researcher**: Uses MMR search for diverse, comprehensive results
    
    ### Example Questions:
    - "Give me Overview of the Bion-m 1 mission?"
    - "What animals were used in space research?"
    - "What are the advantages of using mice in space experiments?"
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, LangChain and pgvector 🚀")

