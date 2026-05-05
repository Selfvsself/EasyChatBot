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
            system_prompt: str,
            history: list[dict] = None,
            internal_messages: list[str] = None,
            answer: str = None,
            action: str = None,
            validation_condition: str = None,
            next_step_query: str = None,
            search_results: list[SearchResult] = None):
        self.user_input = user_input
        self.history = history
        self.internal_messages = internal_messages
        self.system_prompt = system_prompt
        self.answer = answer
        self.action = action
        self.validation_condition = validation_condition
        self.next_step_query = next_step_query
        self.search_results = search_results

    @classmethod
    def from_context(cls, context):
        return cls(
            user_input=context.user_input,
            history=context.history,
            internal_messages=context.internal_messages,
            system_prompt=context.system_prompt,
            answer=context.answer,
            action=context.action,
            validation_condition=context.validation_condition,
            next_step_query=context.next_step_query,
            search_results=context.search_results
        )

    def to_json_string(self) -> str:
        return json.dumps(self.__dict__, indent=4, ensure_ascii=False, default=str)
