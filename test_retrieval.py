from src.retrieval.retriever import HybridRetriever

retriever = HybridRetriever()

query = "malaria treatment guidelines"
results = retriever.retrieve(query, top_k=3)

for r in results:
    print("SOURCE:", r["source"])
    print("DOC:", r["document"])
    print("PAGES:", r["page_start"], "-", r["page_end"])
    print("FUSED SCORE:", r["fused_score"])
    print(r["text"][:300])
    print("-" * 80)
