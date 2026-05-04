from handlers.pipelines.steps.pipeline_context import PipelineContext


class StepResult:
    def __init__(self, context: PipelineContext, stop: bool = False, success: bool = True):
        self.context = context
        self.stop = stop
        self.success = success

    def stage_data(self) -> dict:
        pass
