import logging

from core.config import settings


class ChatMemoryService:
    def __init__(self, message_repo, chat_repo, llm_client):
        self.message_repo = message_repo
        self.chat_repo = chat_repo
        self.llm = llm_client

    @staticmethod
    def _build_messages_block(messages):
        lines = []
        for message in messages:
            role = getattr(message, "role", "user")
            text = str(getattr(message, "text", "")).strip().replace("\n", " ")
            if text:
                lines.append(f"{role}: {text}")
        return "\n".join(lines)

    @staticmethod
    def _trim_memory(memory: str) -> str:
        max_chars = settings.CHAT_MEMORY_MAX_CHARS
        if len(memory) <= max_chars:
            return memory
        return memory[-max_chars:]

    async def _summarize(self, current_memory: str, messages_to_archive) -> str:
        messages_block = self._build_messages_block(messages_to_archive)
        if not messages_block:
            return current_memory

        prompt = [
            {
                "role": "system",
                "content": (
                    "You maintain a compact memory of a chat. "
                    "Keep only important facts, decisions, constraints, preferences and pending tasks. "
                    "Do not include greetings, small talk, or duplicated details. "
                    "Return plain text in one language used by user."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Current memory:\n"
                    f"{current_memory or '(empty)'}\n\n"
                    "New archived dialogue chunk:\n"
                    f"{messages_block}\n\n"
                    "Update memory:"
                ),
            },
        ]
        summary = await self.llm.chat(prompt)
        return self._trim_memory((summary or "").strip())

    async def compress_if_needed(self, chat_id):
        active_count = self.message_repo.count_active_by_chat(chat_id)
        if active_count <= settings.CHAT_MAX_ACTIVE_MESSAGES:
            return

        messages_to_archive = self.message_repo.get_oldest_active_by_chat(
            chat_id=chat_id,
            limit=settings.CHAT_ARCHIVE_BATCH_SIZE,
        )
        if not messages_to_archive:
            return

        chat = self.chat_repo.get_by_id(chat_id)
        if not chat:
            return

        current_memory = (chat.memory_summary or "").strip()

        try:
            updated_memory = await self._summarize(current_memory, messages_to_archive)
        except Exception:
            logging.exception("Failed to summarize archived messages for chat %s", chat_id)
            fallback_part = self._build_messages_block(messages_to_archive)
            updated_memory = self._trim_memory(
                f"{current_memory}\n{fallback_part}".strip()
            )

        self.chat_repo.update_memory_summary(chat_id, updated_memory)
        self.message_repo.mark_as_archived([message.id for message in messages_to_archive])
