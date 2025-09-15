# graphrag_demo.py: Basic GraphRAG Demo for EduGraph AI
# Uses OpenAI embeddings to create a sample knowledge graph from NCERT text.
# Requires: pip install openai langchain langchain-openai langchain-community faiss-cpu
# Set your OpenAI API key as an environment variable: export OPENAI_API_KEY='your-key-here'

import os
from langchain_openai import OpenAIEmbeddings  # Updated import for compatibility
from langchain_community.vectorstores import FAISS  # Simple vector store for demo (simulate graph)
from langchain_core.documents import Document  # Updated schema import
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

# Sample NCERT Text (Physics example: Newton's Laws)
ncert_texts = [
    "Newton's First Law: An object at rest stays at rest and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force.",
    "Newton's Second Law: The acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass. F = ma.",
    "Newton's Third Law: For every action, there is an equal and opposite reaction.",
    "Friction: Opposition to motion, e.g., in daily life like walking on Indian roads during monsoon."
]

# Initialize OpenAI Embeddings (from workshop: Building GraphRAG)
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("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"))

# Create documents for embedding (simulate nodes in knowledge graph)
docs = [Document(page_content=text) for text in ncert_texts]

# Build a simple vector store (FAISS) to simulate GraphRAG retrieval
vector_store = FAISS.from_documents(docs, embeddings)

# Sample Query with Prompt Engineering (multilingual example)
query = "Explain Newton's Laws in Odia, linking to real-world friction examples from Indian villages."  # Prompt-engineered for causal reasoning

# Set up QA chain (Retrieval-Augmented Generation)
llm = OpenAI(temperature=0.7)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())

# Run query
result = qa_chain.run(query)

print("GraphRAG Demo Result:")
print(result)

# In full app: Expand to Neo4j graph, add Whisper/DALL·E integration.
