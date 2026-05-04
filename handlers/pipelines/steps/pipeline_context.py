class PipelineContext:
    def __init__(
            self,
            user_input: str,
            system_prompt: str,
            history: list[dict]=None,
            answer: str=None,
            search_queries: list[str]=None,
            validation_condition: str=None,
            intent: str=None,
            validation_issues: list[str]=None,
            suggestions: str=None,
            search_results: list[dict]=None,
            sources: list[str] = None):
        self.user_input = user_input
        self.history = history
        self.system_prompt = system_prompt
        self.answer = answer
        self.search_queries = search_queries
        self.validation_condition = validation_condition
        self.intent = intent
        self.validation_issues = validation_issues
        self.suggestions = suggestions
        self.search_results = search_results
        self.sources = sources

    @classmethod
    def from_context(cls, context):
        # Создаем новый экземпляр, копируя данные из старого
        return cls(
            user_input=context.user_input,
            history=context.history,
            system_prompt=context.system_prompt,
            answer=context.answer,
            search_queries=context.search_queries,
            validation_condition=context.validation_condition,
            intent=context.intent,
            validation_issues=context.validation_issues,
            suggestions=context.suggestions,
            search_results=context.search_results,
            sources=context.sources
        )
