from src.retrieval.retriever import HybridRetriever
from src.generation.answer_generator import LocalLLMGenerator

retriever = HybridRetriever()
generator = LocalLLMGenerator(model_name="llama3.2:3b")

query = "What are the recommended treatments for malaria?"

retrieved_chunks = retriever.retrieve(query, top_k=5)

answer = generator.generate(query, retrieved_chunks)

print("\n===== ANSWER =====\n")
print(answer)

