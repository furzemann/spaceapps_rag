

How to run:-
1. Install dependencies and use ```pip install -r requirements.txt```
Skip to 5 if you dont want to make your own database locally
2. Add the csv to your database
3. Make a pg database and change the db url in these files to the connection link
4. Run the pg.py file ```python pg.py```
5. replace "YOUR_DATABASE_URL" with connection url - postgresql://neondb_owner:npg_cZJvwbxs23YS@ep-flat-term-a1bhljd0-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require
6. Run the streamlit app with ```streamlit run streamlit_app.py```



## System Architecture

```
flowchart TB
    %% Document Processing Pipeline
    subgraph DocProcess["📄 Document Processing Pipeline"]
        direction TB
        A[Text Documents] --> B[Splitting & Chunking]
        B --> C[(String Database<br/>PostgreSQL)]
        C --> D[Vector Embedding Model<br/>all-mpnet-base-v2]
        D --> E[(Vector Database<br/>Neon + pgvector)]
    end
    
    %% Query Processing
    subgraph QueryProcess["🔍 Query Processing"]
        direction TB
        F[User Query] --> G[Vector Embedding Model<br/>all-mpnet-base-v2]
        G --> H[Query Vector]
    end
    
    %% Retrieval Strategy
    subgraph Retrieval["🎯 Retrieval Strategy"]
        direction TB
        I{User Level?}
        I -->|Student| J[MultiQueryRetriever<br/>T5-Small]
        I -->|Researcher| K[MMR Retriever<br/>Maximal Marginal Relevance]
        J --> L[Generate Multiple<br/>Query Variations]
        K --> M[Diverse Document<br/>Selection]
        L --> N[Retrieved Documents]
        M --> N
    end
    
    %% Response Generation
    subgraph ResponseGen["🤖 Response Generation"]
        direction TB
        O[Prompt Template] --> P[Context + Query<br/>Combination]
        P --> Q[LLM<br/>Llama 3.2 3B Instruct]
        Q --> R[Generated Response]
    end
    
    %% Main Flow
    E --> I
    H --> I
    N --> O
    
    %% Styling
    classDef processStyle fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
    classDef databaseStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef modelStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef decisionStyle fill:#fff9c4,stroke:#f9a825,stroke-width:3px
    
    class A,B,F,L,M,P processStyle
    class C,E databaseStyle
    class D,G,J,K,Q modelStyle
    class I decisionStyle
```

### Key Components

- **Document Pipeline**: Processes and stores document embeddings
- **Query Processing**: Converts user queries to vector format
- **Adaptive Retrieval**: Different strategies for students vs researchers
- **LLM Generation**: Combines context and query for final response
```

## Alternative: Simplified Version

If you want a cleaner, more compact version :[4][1]

````markdown
## RAG System Architecture

```
flowchart LR
    %% Input Sources
    A[📄 Text Documents] --> B[Text Splitting]
    B --> C[(String DB)]
    C --> D[Embeddings]
    D --> E[(Vector DB<br/>pgvector)]
    
    F[👤 User Query] --> G[Query Embedding]
    
    %% Retrieval Decision
    E --> H{User Level}
    G --> H
    
    H -->|Student| I[MultiQuery<br/>Retriever]
    H -->|Researcher| J[MMR<br/>Retriever]
    
    I --> K[Context Docs]
    J --> K
    
    %% Generation
    K --> L[Prompt Builder]
    G --> L
    L --> M[🤖 LLM<br/>Llama 3.2]
    M --> N[💬 Response]
    
    %% Styling
    style E fill:#90caf9
    style H fill:#fff59d
    style M fill:#ce93d8
```
```

## Detailed Version with Technical Specs

For a comprehensive technical diagram :[5][6]

````markdown
## Biology RAG System - Complete Architecture

```
flowchart TB
    %% Data Ingestion
    subgraph DataIngest["📥 Data Ingestion Layer"]
        direction LR
        DOC[Research Papers<br/>Biology Documents] 
        SPLIT[Chunking Engine<br/>Token-based splitting]
        STORE[(Neon PostgreSQL<br/>paper_chunks table)]
        DOC --> SPLIT
        SPLIT --> STORE
    end
    
    %% Embedding Generation
    subgraph EmbedGen["🧬 Embedding Generation"]
        direction TB
        EMBED1[SentenceTransformer<br/>all-mpnet-base-v2<br/>768 dimensions]
        HASH[SHA256 Hash<br/>Deduplication]
        VSTORE[(Vector Store<br/>document_embeddings<br/>pgvector + HNSW index)]
        STORE --> EMBED1
        EMBED1 --> HASH
        HASH --> VSTORE
    end
    
    %% Query Pipeline
    subgraph QueryPipe["🔎 Query Pipeline"]
        direction TB
        QUERY[User Question] --> QEMBED[Query Embedding<br/>all-mpnet-base-v2]
        QEMBED --> QVEC[Query Vector<br/>768-dim]
    end
    
    %% Intelligent Retrieval
    subgraph IntelRetrieval["🎯 Intelligent Retrieval System"]
        direction TB
        DECISION{User Profile}
        
        subgraph StudentPath["Student Path"]
            T5[T5-Small Model<br/>Query Generator]
            MQR[MultiQueryRetriever]
            MULTI[Multiple Query<br/>Perspectives]
            T5 --> MQR
            MQR --> MULTI
        end
        
        subgraph ResearcherPath["Researcher Path"]
            MMR[MMR Algorithm<br/>λ = 0.5]
            DIVERSE[Diverse Results<br/>Relevance + Diversity]
            MMR --> DIVERSE
        end
        
        DECISION -->|level='General'| StudentPath
        DECISION -->|level='Researcher'| ResearcherPath
        MULTI --> CONTEXT
        DIVERSE --> CONTEXT
        CONTEXT[Retrieved Context<br/>Top-k Documents]
    end
    
    %% LLM Generation
    subgraph LLMGen["🤖 Response Generation"]
        direction TB
        TEMPLATE[Chat Template<br/>System + User + Context]
        COMBINE[Prompt Combination]
        LLM[Llama 3.2 3B Instruct<br/>Temperature: 0.1<br/>Max Tokens: 512]
        HISTORY[(Chat History<br/>Last 10 exchanges)]
        
        TEMPLATE --> COMBINE
        HISTORY -.-> COMBINE
        COMBINE --> LLM
        LLM --> RESPONSE[Final Response]
        RESPONSE -.-> HISTORY
    end
    
    %% Main Connections
    VSTORE -.-> DECISION
    QVEC --> DECISION
    CONTEXT --> TEMPLATE
    QVEC --> COMBINE
    
    %% Styling
    classDef inputStyle fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef processStyle fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef dbStyle fill:#fff3e0,stroke:#e65100,stroke-width:3px
    classDef modelStyle fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef decisionStyle fill:#fff9c4,stroke:#f57f17,stroke-width:3px,stroke-dasharray: 5 5
    classDef outputStyle fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class DOC,QUERY inputStyle
    class SPLIT,HASH,QEMBED,COMBINE processStyle
    class STORE,VSTORE,HISTORY dbStyle
    class EMBED1,T5,MQR,MMR,LLM modelStyle
    class DECISION decisionStyle
    class RESPONSE outputStyle
```

### Architecture Highlights

#### Document Processing
- **Input**: Research papers and biology documents
- **Chunking**: Token-based text splitting for optimal context
- **Storage**: PostgreSQL with pgvector extension
- **Embeddings**: 768-dimensional vectors using all-mpnet-base-v2
- **Index**: HNSW for fast similarity search

#### Query Processing
- **Embedding**: Same model as documents for consistency
- **Vectorization**: Real-time query to vector conversion

#### Intelligent Retrieval
- **Student Mode**: MultiQueryRetriever generates 3-5 query variations using T5-Small
- **Researcher Mode**: MMR algorithm balances relevance and diversity
- **Adaptive**: Automatically selects strategy based on user level

#### Response Generation
- **Model**: Llama 3.2 3B Instruct via HuggingFace
- **Context**: Retrieved documents + user query
- **History**: Maintains last 10 conversation exchanges
- **Style**: Adapts response complexity to user level
```

## Copy-Paste Ready Code

Here's the most balanced version for your README :[7][8]

````markdown
# Biology RAG Chatbot

## System Architecture

```
flowchart TB
    subgraph Input["📥 Input Layer"]
        A[Text Documents] 
        B[User Query]
    end
    
    subgraph Processing["⚙️ Processing Layer"]
        C[Text Chunking] --> D[(String Database)]
        D --> E[Vector Embeddings<br/>all-mpnet-base-v2]
        E --> F[(Vector Database<br/>Neon + pgvector)]
        
        B --> G[Query Embedding]
    end
    
    subgraph Retrieval["🎯 Retrieval Layer"]
        H{User Level?}
        H -->|Student| I[MultiQueryRetriever<br/>Multiple query variants]
        H -->|Researcher| J[MMR Retriever<br/>Diverse results]
    end
    
    subgraph Generation["🤖 Generation Layer"]
        K[Prompt Template]
        L[Context + Query]
        M[LLM: Llama 3.2 3B]
        N[Response]
        
        K --> L --> M --> N
    end
    
    A --> C
    F --> H
    G --> H
    I --> K
    J --> K
    G --> L
    
    classDef blue fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef orange fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef purple fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef green fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    
    class A,B green
    class C,D,E,F,G blue
    class H,I,J orange
    class K,L,M,N purple
```

### How It Works

1. **Document Processing**: Text documents are split into chunks, embedded, and stored in a vector database
2. **Query Processing**: User queries are converted to vectors using the same embedding model
3. **Smart Retrieval**: 
   - **Students**: Get comprehensive results via MultiQueryRetriever
   - **Researchers**: Get diverse results via MMR algorithm
4. **Response Generation**: Retrieved context + query → LLM → Tailored response
```

## Preview Your Diagram

Test and customize at [mermaid.live](https://mermaid.live) before adding to README.[6][9]

Choose the version that best fits your documentation needs! The detailed version is great for technical documentation, while the simplified version works better for quick overviews.[2][3]

