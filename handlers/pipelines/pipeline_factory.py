from handlers.pipelines.agent_web_page_analys_pipeline import WebPageAnalysAgentPipeline
from handlers.pipelines.agent_web_search_pipeline import WebAgentPipeline
from handlers.pipelines.steps.generate_json_step import GenerateJsonStep
from handlers.pipelines.steps.generate_step import GenerateStep
from handlers.pipelines.steps.plan_step import PlanStep
from handlers.pipelines.steps.search_page_source_step import SearchPageSourceStep
from handlers.pipelines.steps.search_rag_step import SearchRagStep
from handlers.pipelines.steps.search_summary_step import SearchSummaryStep
from handlers.pipelines.steps.search_web_step import SearchWebStep
from handlers.pipelines.agent_pipeline import AgentPipeline
from handlers.pipelines.steps.validation_criteria_step import ValidationCriteriaStep
from handlers.pipelines.steps.search_query_step import SearchQueryStep


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
            PlanStep(self.llm_client),
            SearchWebStep(self.llm_client, max_results=15),
            SearchPageSourceStep(self.llm_client, max_results=8),
            SearchRagStep(self.llm_client),
            SearchSummaryStep(self.llm_client),
            GenerateStep(self.llm_client)
        ]

    def web_step(self):
        return [
            PlanStep(self.llm_client),
            ValidationCriteriaStep(self.llm_client),
            SearchQueryStep(self.llm_client)
        ]

    def main_agent(self):
        return [
            AgentPipeline(self.llm_client)
        ]

    def web_search_agent(self):
        return [
            WebAgentPipeline(self.llm_client)
        ]
