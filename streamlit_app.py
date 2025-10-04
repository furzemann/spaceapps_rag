import streamlit as st
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.prompts import PromptTemplate
import numpy as np

# Set page config
st.set_page_config(
    page_title="Biology RAG Chatbot",
    page_icon="🧬",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = False

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    hf_token = st.text_input("Hugging Face Token", type="password", key="hf_token")
    
    style = st.selectbox("Response Style", ["fun", "formal", "casual"], index=0)
    level = st.selectbox("User Level", ["student", "researcher", "expert"], index=0)
    max_token = st.slider("Max Tokens", 20, 200, 50)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1)
    
    if st.button("Initialize RAG System"):
        if hf_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
            st.session_state.rag_initialized = True
            st.success("RAG System Initialized!")
        else:
            st.error("Please enter your Hugging Face token")

# Main content
st.title("🧬 Biology RAG Chatbot")
st.markdown("Ask questions about space biology experiments!")

# Initialize RAG components
@st.cache_resource
def initialize_rag():
    # Text data
    text = [
        "After a 16-year hiatus, Russia resumed in 2013 its program of biomedical research in space, with the successful 30-day flight of the Bion-M 1 biosatellite (April 19–May 19, 2013), a specially designed automated spacecraft dedicated to life-science experiments. 'M' in the mission's name stands for 'modernized'; the epithet was equally applicable to the spacecraft and the research program. The principal animal species for physiological studies in this mission was the mouse (Mus musculus). Unlike more recent space experiments that used female mice, males were flown in the Bion-M 1 mission. The challenging task of supporting mice in space for this unmanned, automated, 30-day-long mission, was made even more so by the requirement to house the males in groups.",
        
        "Russian biomedical research in space traditionally has employed dogs, rats, monkeys, and more recently Mongolian gerbils. The flight of Laika in 1957 was one of the early dog experiments and became world famous for demonstrating that a living organism can withstand rocket launch and weightlessness, thus paving the way for the first human spaceflight. Laika's success also promoted biomedical research with other non-human animals in space that culminated with the Bion biosatellites program. A total of 212 rats and 12 monkeys were launched on 11 satellites and exposed in microgravity for 5.0–22.5 days between 1973 and 1997. Animal experiments on the Bion missions have contributed comprehensive data on adaptive responses of sensorimotor, cardiovascular, muscle, bone and other systems to spaceflight conditions and the mechanisms underlying these adaptations [1], [2].",
        
        "The use of mice for space experiments offers numerous advantages. Probably the most apparent one is their small size and thus the possibility of utilizing more animals per flight, thus increasing scientific output and the cost-efficiency ratio. Comparisons of data obtained with mice, with those obtained from larger species or humans can also reveal how factors affecting adaptation to spaceflight conditions depend on the size of the organism. The mouse has become the most prevalent 'mammalian model' in biomedical research, with a fully described genome and an established role in genetically engineered mutants. While mice are preferred mammalian models for molecular biology studies, their small size is a debated limitation rather than an advantage for physiological studies. Miniaturization of scientific hardware has reduced some of the disadvantages of the species small size. Finally, the use of genetically controlled mice offers a means to reduce inter-individual variability and obtain potentially more consistent results.",
        
        "Despite the advantages of the mouse as a model organism for space research, their use was rather limited (apart from a number of experiments with mice during early space exploration in the 1950's and 1960's, which were aimed primarily at testing if living organisms can survive the launch or a brief exposure in microgravity) [3]. Flight experiments with mice were performed aboard STS-90 ('NeuroLab'), STS-108, STS-129, STS-131 ('Mouse Immunology I'), STS-133 ('Mouse Immunology II'), and STS-135 with exposure times ranging from 12 to 16 days. Research programs of these flights were largely focused on studies of muscle, bone/tendon/cartilage, nervous, and cardiovascular systems, and innate and acquired immune responses. Experiments were performed with groups of 30 or fewer female C57BL/6J mice, which were dissected typically shortly after return [4]–[9].",
        
        "The Mice Drawer System (MDS) experiment of the Italian Space Agency is by far the longest spaceflight of mice to date [10]. In this mission, 6 mice were exposed for 91 days aboard the International Space Station. The advantages offered by the possibility of genetic manipulations with mice were utilized in this experiment; three mice were transgenic with pleiotrophin overexpression (C57BLJ10/ PTN) and three mice were their wild-type counterparts. The MDS habitats required periodic replenishment and servicing by by astronauts. Sadly, half of the mice died during the course of this mission due to various estimated reasons.",
        
        "In the present paper we aim to present a brief overview of the Bion-M 1 mission scientific goals and experimental design. Of particular interest we will focus on the program of mouse training and selection for the experiments, and some outcomes of the Bion-M 1 mission."
    ]
    
    # Initialize embeddings
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    # Create embeddings for text
    text_vectors = embeddings.embed_documents(text)
    
    return text, embeddings, text_vectors

def get_relevant_context(question, text, embeddings, text_vectors):
    question_vector = embeddings.embed_query(question)
    scores = cosine_similarity([question_vector], text_vectors)[0]
    
    # Get top 2 most relevant chunks
    top_indices = np.argsort(scores)[-2:][::-1]
    relevant_text = text[top_indices[0]] + "\n\n" + text[top_indices[1]]
    
    return relevant_text

def generate_response(question, style, level, max_token, temperature):
    try:
        # Initialize RAG components
        text, embeddings, text_vectors = initialize_rag()
        
        # Get relevant context
        relevant_context = get_relevant_context(question, text, embeddings, text_vectors)
        
        # Initialize LLM
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            task="text-generation",
            temperature=temperature,
        )
        model = ChatHuggingFace(llm=llm)
        
        # Create prompt
        template_text = """
        System : You are a biology rag chatbot, you will be given a context and a question from that context and you have to answer that question with {style}, and considering him as {level}
        User :  Explain the following query -
                {query} with {style} and assuming user to be {level}
        Context regarding query: {required_text_for_query}

        Only give answer the question when you are sure of it and it is present in Context somewhere.
        Also consider the max token limit to be {max_token},do not go over this token limit.
        """
        
        prompt = PromptTemplate(
            template=template_text,
            input_variables=['query', 'style', 'level', 'max_token', 'required_text_for_query']
        )
        
        template = prompt.invoke({
            'query': question,
            'style': style,
            'level': level,
            'max_token': max_token,
            'required_text_for_query': relevant_context,
        })
        
        result = model.invoke(template)
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
            with st.spinner("Thinking..."):
                response = generate_response(prompt, style, level, max_token, temperature)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

else:
    st.info("Please initialize the RAG system using the sidebar configuration.")
    st.markdown("""
    ### Getting Started:
    1. Enter your Hugging Face API token in the sidebar
    2. Configure your preferred settings
    3. Click "Initialize RAG System"
    4. Start asking questions about space biology experiments!
    
    ### Example Questions:
    - "What is the aim of the experiment?"
    - "What animals were used in space research?"
    - "What are the advantages of using mice in space experiments?"
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and LangChain 🚀")
