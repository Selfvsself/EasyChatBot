import json


class SearchResult:
    def __init__(self, title: str, url: str, text: str):
        self.title = title
        self.url = url
        self.text = text

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "text": self.text
        }

    def __str__(self) -> str:
        return f"Заголовок: {self.title}\nСсылка: {self.url}\nТекст: {self.text}"

    def __repr__(self) -> str:
        return f"SearchResult(title='{self.title}', url='{self.url}')"


class StepContext:
    def __init__(
            self,
            user_input: str,
            chat_memory: str,
            system_prompt: str,
            history: list[dict] = None,
            answer: str = None,
            user_intent: str = None,
            action: str = None,
            validation_condition: str = None,
            next_step_query: str = None,
            search_results: list[SearchResult] = None):
        self.user_input = user_input
        self.chat_memory = chat_memory
        self.history = history
        self.system_prompt = system_prompt
        self.answer = answer
        self.user_intent = user_intent
        self.action = action
        self.validation_condition = validation_condition
        self.next_step_query = next_step_query
        self.search_results = search_results

    @classmethod
    def from_context(cls, context):
        return cls(
            user_input=context.user_input,
            chat_memory=context.chat_memory,
            history=context.history,
            system_prompt=context.system_prompt,
            answer=context.answer,
            user_intent=context.user_intent,
            action=context.action,
            validation_condition=context.validation_condition,
            next_step_query=context.next_step_query,
            search_results=context.search_results
        )

    def to_json_string(self) -> str:
        return json.dumps(self.__dict__, indent=4, ensure_ascii=False, default=str)
