import subprocess
from typing import List, Dict


class LocalLLMGenerator:
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model_name = model_name

    def generate(self, query, retrieved_chunks, max_chunks=5):
        context_blocks = []

        for i, c in enumerate(retrieved_chunks[:max_chunks], start=1):
            block = (
                f"[SOURCE {i}]\n"
                f"Source: {c['source']}\n"
                f"Document: {c['document']}\n"
                f"Pages: {c['page_start']}–{c['page_end']}\n"
                f"Text: {c['text']}\n"
            )
            context_blocks.append(block)

        context = "\n\n".join(context_blocks)

        prompt = f"""
You are a public health assistant.

You must answer the user's question using ONLY the sources provided below.
Do NOT use any external knowledge.
If the answer is not fully supported by the sources, say:
"I do not have enough information from WHO or CDC guidelines to answer this."

Cite sources using [SOURCE X] notation.

User Question:
{query}

Sources:
{context}

Answer:
"""

        result = subprocess.run(
            ["ollama", "run", self.model_name],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        output = result.stdout + result.stderr

        return output.decode("utf-8", errors="replace").strip()
