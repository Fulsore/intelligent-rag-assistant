import os
from groq import Groq


class LLMService:

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in .env"
            )

        self.client = Groq(api_key=api_key)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512
    ) -> str:

        try:

            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful healthcare knowledge assistant. "
                            "Answer only using the provided context."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:

            print(f"[LLM ERROR] {e}")

            return (
                "I encountered an error generating a response. "
                "Please try again."
            )


llm_service = LLMService()