from handlers.pipelines.steps2.step_context import StepContext


class StepResult:
    def __init__(self, context: StepContext, stop: bool = False, success: bool = True):
        self.context = context
        self.stop = stop
        self.success = success

    def stage_data(self) -> dict:
        pass
