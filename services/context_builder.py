"""
services/context_builder.py
Healthcare KIS - Context Builder

Assembles the final LLM prompt from ranked context chunks.
Handles disclaimer injection and context truncation.
"""

from typing import List, Tuple
from langchain_core.documents import Document


MEDICAL_DISCLAIMER = (
    "\n\n⚠️ Medical Disclaimer: This information is for educational purposes only "
    "and should not replace professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare provider for medical concerns."
)

SYSTEM_PROMPT = """You are a knowledgeable healthcare assistant for the Knowledge Infrastructure System (KIS).
Answer questions accurately using ONLY the provided context.
If the context does not contain enough information, say so clearly.
Be concise, factual, and cite the source of your information when possible.
Do not speculate or provide information beyond what the context supports."""


class ContextBuilder:
    """
    Builds the final prompt to send to the LLM.
    """

    MAX_CONTEXT_CHARS = 3000  # Prevent token overflow

    def build_prompt(
        self,
        question: str,
        ranked_docs: List[Tuple[Document, float]],
        requires_disclaimer: bool = False,
    ) -> str:
        """
        Builds a structured prompt with system instruction + context + question.
        """
        context_parts = []
        char_count = 0

        for i, (doc, score) in enumerate(ranked_docs):
            chunk_text = doc.page_content.strip()
            chunk_header = f"[Source {i+1} | Relevance: {score:.2f}]"
            chunk = f"{chunk_header}\n{chunk_text}\n"

            if char_count + len(chunk) > self.MAX_CONTEXT_CHARS:
                break

            context_parts.append(chunk)
            char_count += len(chunk)

        context_text = "\n".join(context_parts) if context_parts else "No context available."

        prompt = f"""{SYSTEM_PROMPT}

--- CONTEXT START ---
{context_text}
--- CONTEXT END ---

Question: {question}

Answer:"""

        return prompt

    def build_no_context_prompt(self, question: str) -> str:
        """Used when no relevant context was found."""
        return (
            f"{SYSTEM_PROMPT}\n\n"
            "No relevant context was found in the knowledge base.\n\n"
            f"Question: {question}\n\n"
            "Answer: I could not find relevant information in the knowledge base to answer this question. "
            "Please consult a healthcare professional."
        )

    def get_disclaimer(self) -> str:
        return MEDICAL_DISCLAIMER

    def inject_disclaimer(self, answer: str) -> str:
        """Appends the medical disclaimer to the answer."""
        return answer + MEDICAL_DISCLAIMER


# Singleton
context_builder = ContextBuilder()