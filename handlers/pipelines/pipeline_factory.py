from handlers.pipelines.steps.generate_step import GenerateStep
from handlers.pipelines.steps.generate_json_step import GenerateJsonStep
from handlers.pipelines.steps.search_plan_step import SearchPlanStep
from handlers.pipelines.steps.search_web_step import SearchWebStep
from handlers.pipelines.steps.search_page_source_step import SearchPageSourceStep
# from handlers.pipelines.steps.search_all_summary_step import SearchSummaryStep
from handlers.pipelines.steps.search_agr_summary_step import SearchSummaryStep
from handlers.pipelines.steps.search_response_step import SearchResponseStep

class PipelineFactory:

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def simple_pipeline(self):
        return [
            GenerateStep(self.llm_client)
        ]

    def simple_translate_pipeline(self):
        return [
            GenerateJsonStep(self.llm_client)
        ]

    def web_search_pipeline(self):
        return [
            SearchPlanStep(self.llm_client),
            SearchWebStep(self.llm_client, max_results=15),
            SearchPageSourceStep(self.llm_client, max_results=8),
            SearchSummaryStep(self.llm_client),
            SearchResponseStep(self.llm_client)
        ]