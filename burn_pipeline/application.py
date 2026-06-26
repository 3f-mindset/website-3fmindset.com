from __future__ import annotations

from pathlib import Path

from .domain import (
    BurnContext,
    FileStorePort,
    GenerateCommand,
    InferencePort,
    InferenceRequest,
    PipelineSpec,
    StepSpec,
    build_generation_prompt,
    validate_generated_content,
)


class BurnPipeline:
    def __init__(self, files: FileStorePort, inference: InferencePort) -> None:
        self._files = files
        self._inference = inference

    def generate_step(self, step: StepSpec, context: BurnContext, force: bool) -> str:
        prompt_template = self._files.read_text(step.prompt_file)
        named_inputs = self._load_inputs(step)
        prompt = build_generation_prompt(
            GenerateCommand(
                step=step,
                context=context,
                prompt_template=prompt_template,
                named_inputs=named_inputs,
            )
        )
        content = self._inference.generate(
            InferenceRequest(prompt=prompt, output_format=step.format)
        )
        validate_generated_content(content, step.format)
        self._files.write_text(step.output, content, force=force)
        return content

    def run_pipeline(self, spec: PipelineSpec, force: bool) -> None:
        completed: set[str] = set()
        steps_by_id = {step.id: step for step in spec.steps}

        for step in spec.steps:
            missing = [step_id for step_id in step.depends_on if step_id not in completed]
            unknown = [step_id for step_id in step.depends_on if step_id not in steps_by_id]
            if unknown:
                raise ValueError(f"Step {step.id} depends on unknown step(s): {', '.join(unknown)}")
            if missing:
                raise ValueError(
                    f"Step {step.id} ran before dependency step(s): {', '.join(missing)}"
                )

            self.generate_step(step=step, context=spec.context, force=force)
            completed.add(step.id)

    def _load_inputs(self, step: StepSpec) -> dict[str, str]:
        named_inputs: dict[str, str] = {}
        for path in step.inputs:
            named_inputs[str(path)] = self._files.read_text(path)
        return named_inputs


def step_from_paths(
    *,
    step_id: str,
    output_format: str,
    prompt_file: Path,
    output: Path,
    inputs: list[Path],
) -> StepSpec:
    return StepSpec(
        id=step_id,
        format=output_format,
        prompt_file=prompt_file,
        output=output,
        inputs=inputs,
    )
