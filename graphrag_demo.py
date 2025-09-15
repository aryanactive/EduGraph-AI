# graphrag_demo.py: Basic GraphRAG Demo for EduGraph AI
# Uses OpenAI embeddings to create a sample knowledge graph from NCERT text.
# Requires: pip install openai langchain
# Set your OpenAI API key as an environment variable: export OPENAI_API_KEY='your-key-here'

import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS  # Simple vector store for demo (simulate graph)
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Sample NCERT Text (Physics example: Newton's Laws)
ncert_texts = [
    "Newton's First Law: An object at rest stays at rest and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force.",
    "Newton's Second Law: The acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass. F = ma.",
    "Newton's Third Law: For every action, there is an equal and opposite reaction.",
    "Friction: Opposition to motion, e.g., in daily life like walking on Indian roads during monsoon."
]

# Initialize OpenAI Embeddings (from workshop: Building GraphRAG)
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

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
