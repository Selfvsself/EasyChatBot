import httpx


class LLMClient:

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model

    async def chat(self, messages: list[dict]) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages
                }
            )

            response.raise_for_status()

            data = response.json()

            return data["choices"][0]["message"]["content"]