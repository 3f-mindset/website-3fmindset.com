# SteadyBurn Weekly Bundle Model Comparisons

These case studies measure the practical cost of producing one complete SteadyBurn weekly content bundle with models that were active and available at the time of each run.

The local-model run is the baseline. It represents the capability already available on this machine: local text generation through llama-swap and a local hero-image model. The OpenRouter runs use the same seed, pipeline structure, and optional-track settings so their outputs and provider-reported costs can be compared against that baseline.

Each model directory contains one isolated run of the weekly bundle: context, lesson, instructions, worksheet copy, worksheet and masked worksheet, long-form letter, newsletter email, community post, cover prompt, rendered hero image, and the pipeline configuration used. OpenRouter case studies also include `OPENROUTER_USAGE.jsonl` and `MODEL_COMPARISON.md` with the provider-reported usage and cost.

These are comparison artifacts, not publishable site content. They stay under `automation/case-studies` so evaluating model quality and operating cost does not change the live `content` collection.

## Cost Charts

- [Linear scale](weekly-bundle-costs.svg): preserves absolute cost differences.
- [Logarithmic scale](weekly-bundle-costs-log.svg): makes the lower-cost models easier to compare.
