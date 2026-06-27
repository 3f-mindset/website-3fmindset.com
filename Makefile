CODEX_BURN_OUT ?= POC.md
CODEX_BURN_TITLE ?= Harness POC
CODEX_BURN_DATE ?= $(shell date +%F)
CODEX_BURN_SLUG ?= harness-poc
CODEX_BURN_MODEL ?= active
CODEX_BURN_FORCE ?=
CODEX_BURN_FORCE_ARG = $(if $(CODEX_BURN_FORCE),--force,)
BURN_PROVIDER ?= openai-compatible
BURN_BASE_URL ?= http://localhost:11434/v1
BURN_API_KEY_ENV ?= LLAMA_ROX
BURN_CODEX_BIN ?= 
BURN_PROVIDER_ARGS = --provider "$(BURN_PROVIDER)" --api-key-env "$(BURN_API_KEY_ENV)" --codex-bin "$(BURN_CODEX_BIN)" $(if $(CODEX_BURN_MODEL),--model "$(CODEX_BURN_MODEL)",) $(if $(BURN_BASE_URL),--base-url "$(BURN_BASE_URL)",)

.PHONY: poc poc-dry-run all burn-require-dir

poc:
	uv run automation/burn-pipeline $(BURN_PROVIDER_ARGS) generate-step \
		--format markdown \
		--prompt-file prompts/burn/poc-markdown.md \
		--output "$(CODEX_BURN_OUT)" \
		--title "$(CODEX_BURN_TITLE)" \
		--slug "$(CODEX_BURN_SLUG)" \
		--date "$(CODEX_BURN_DATE)" \
		--force

dry-run:
	uv run automation/burn-pipeline $(BURN_PROVIDER_ARGS) run \
		--pipeline pipelines/burn-poc.toml \
		--title "$(CODEX_BURN_TITLE)" \
		--slug "$(CODEX_BURN_SLUG)" \
		--date "$(CODEX_BURN_DATE)" \
		--force \
		--dry-run

burn-require-dir:
	@test -n "$(BURN_DIR)" || (echo 'Usage: make all BURN_DIR=content/letters/YYYY-MM-DD-slug CODEX_BURN_FORCE=1' >&2; exit 2)
	@test -d "$(BURN_DIR)" || (echo 'BURN_DIR does not exist: $(BURN_DIR)' >&2; exit 2)

all: burn-require-dir
	uv run automation/burn-pipeline $(BURN_PROVIDER_ARGS) generate-step --step-id lesson --format markdown --prompt-file prompts/burn/lesson.md --input "$(BURN_DIR)/index.md" --output "$(BURN_DIR)/LESSON.md" --title "$(CODEX_BURN_TITLE)" --slug "$(CODEX_BURN_SLUG)" --date "$(CODEX_BURN_DATE)" $(CODEX_BURN_FORCE_ARG)
	uv run automation/burn-pipeline $(BURN_PROVIDER_ARGS) generate-step --step-id instructions --format markdown --prompt-file prompts/burn/instructions.md --input "$(BURN_DIR)/LESSON.md" --output "$(BURN_DIR)/INSTRUCTIONS.md" --title "$(CODEX_BURN_TITLE)" --slug "$(CODEX_BURN_SLUG)" --date "$(CODEX_BURN_DATE)" $(CODEX_BURN_FORCE_ARG)
	uv run automation/burn-pipeline $(BURN_PROVIDER_ARGS) generate-step --step-id context --format markdown --prompt-file prompts/burn/context.md --input "$(BURN_DIR)/LESSON.md" --output "$(BURN_DIR)/CONTEXT.md" --title "$(CODEX_BURN_TITLE)" --slug "$(CODEX_BURN_SLUG)" --date "$(CODEX_BURN_DATE)" $(CODEX_BURN_FORCE_ARG)
	uv run automation/burn-pipeline $(BURN_PROVIDER_ARGS) generate-step --step-id gpt --format markdown --prompt-file prompts/burn/gpt.md --input "$(BURN_DIR)/CONTEXT.md" --output "$(BURN_DIR)/GPT.md" --title "$(CODEX_BURN_TITLE)" --slug "$(CODEX_BURN_SLUG)" --date "$(CODEX_BURN_DATE)" $(CODEX_BURN_FORCE_ARG)
	uv run automation/burn-pipeline $(BURN_PROVIDER_ARGS) generate-step --step-id worksheet --format svg --prompt-file prompts/burn/worksheet-svg.md --input "$(BURN_DIR)/LESSON.md" --output "$(BURN_DIR)/WORKSHEET.svg" --title "$(CODEX_BURN_TITLE)" --slug "$(CODEX_BURN_SLUG)" --date "$(CODEX_BURN_DATE)" $(CODEX_BURN_FORCE_ARG)
	uv run automation/burn-pipeline $(BURN_PROVIDER_ARGS) generate-step --step-id worksheet-masked --format svg --prompt-file prompts/burn/worksheet-masked-svg.md --input "$(BURN_DIR)/WORKSHEET.svg" --output "$(BURN_DIR)/WORKSHEET_MASKED.svg" --title "$(CODEX_BURN_TITLE)" --slug "$(CODEX_BURN_SLUG)" --date "$(CODEX_BURN_DATE)" $(CODEX_BURN_FORCE_ARG)
	uv run automation/burn-pipeline $(BURN_PROVIDER_ARGS) generate-step --step-id index --format markdown --prompt-file prompts/burn/index.md --input "$(BURN_DIR)/LESSON.md" --output "$(BURN_DIR)/index.md" --title "$(CODEX_BURN_TITLE)" --slug "$(CODEX_BURN_SLUG)" --date "$(CODEX_BURN_DATE)" $(CODEX_BURN_FORCE_ARG)
