from handlers.pipelines.steps.pipeline_step import PipelineStep

class PipelineOrchestrator:

    def __init__(self):
        pass

    @staticmethod
    async def _emit_stage(stage_callback, stage: str, metadata: dict | None = None):
        if stage_callback:
            await stage_callback(stage=stage, metadata=metadata or {})

    async def run(self, pipeline: list[PipelineStep], context=None, stage_callback=None):
        if not context:
            raise ValueError("Pipeline context is missing or empty")
        result = None
        last_context = context
        for step in pipeline:
            await self._emit_stage(stage_callback, step.stage(), step.stage_metadata())
            result = await step.execute(last_context, stage_callback)
            if result.stop:
                break
            last_context = result.context
        return result.context.answer
