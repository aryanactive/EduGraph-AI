# EduGraph-AI: Personalized AI Tutor for Inclusive Indian Education

## Project Overview
EduGraph AI is a GenAI-powered tutor app for the NxtWave x OpenAI Academy Buildathon (Theme: AI in Education). It uses GraphRAG built with OpenAI APIs to create dynamic knowledge graphs from Indian curricula (e.g., NCERT texts). This addresses education gaps in Tier 2/3 and rural areas by providing multilingual, adaptive learning.

### Key Features
- **Voice-Enabled Tutoring**: Speak queries in Odia/Hindi; get audio responses and visuals.
- **Adaptive Paths**: Personalized recommendations via prompt-engineered chains.
- **Collaborative Mode**: Shared graphs for team learning.
- **Scalable**: Mobile/web, low-cost with OpenAI free tier.

### Problem It Solves
In India, 42% of rural youth struggle with English (ASER 2023), leading to knowledge silos. EduGraph democratizes STEM education for 250M+ learners.

### Build Plan Snippet
- Week 1: Set up GraphRAG with NCERT data.
- Tech: OpenAI (GPT-4o, Whisper, DALL·E), LangChain, Python.

### Demo Code
See `graphrag_demo.py` for a basic GraphRAG skeleton:
- Embeds sample NCERT physics text.
- Builds a simple knowledge graph.
- Queries with prompt engineering.

Run it: `python graphrag_demo.py` (requires OpenAI API key).

### Workshop Ties
Based on NxtWave x OpenAI learnings: Generative AI, Prompt Engineering, GraphRAG.

For full submission details, see our Buildathon doc. Contact: Aryan Raj (aryanactive).
