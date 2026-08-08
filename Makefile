PYTHON ?= python3
DEFAULT_BURN_DATE ?= $(shell $(PYTHON) -c "from datetime import date; print(date.today().isoformat())")

BURN_STATE_FILE ?= tmp/current-burn.mk
BURN_STATE_SCRIPT ?= scripts/burn_make_state.py

-include $(BURN_STATE_FILE)

CODEX_BURN_TITLE ?= $(BURN_STATE_TITLE)
CODEX_BURN_DATE ?= $(if $(BURN_STATE_DATE),$(BURN_STATE_DATE),$(DEFAULT_BURN_DATE))
CODEX_BURN_SLUG ?= $(BURN_STATE_SLUG)
CODEX_BURN_MODEL ?= active

BURN_PROVIDER ?= openai-compatible
BURN_BASE_URL ?= $(if $(filter openrouter,$(BURN_PROVIDER)),https://openrouter.ai/api/v1,http://localhost:11434)
BURN_API_KEY_ENV ?= $(if $(filter openrouter,$(BURN_PROVIDER)),OPENROUTER_API_KEY,OPENAI_API_KEY)
BURN_CODEX_BIN ?=
BURN_TARGET_ROOT ?= content/letters

BURN_DIR ?= $(if $(BURN_STATE_DIR),$(BURN_STATE_DIR),$(BURN_TARGET_ROOT)/$(CODEX_BURN_DATE)-$(CODEX_BURN_SLUG))

BURN_PIPELINE_CMD = $(PYTHON) scripts/burn-pipeline.py
BURN_PROVIDER_ARGS = --provider "$(BURN_PROVIDER)" $(if $(CODEX_BURN_MODEL),--model "$(CODEX_BURN_MODEL)",) $(if $(BURN_BASE_URL),--base-url "$(BURN_BASE_URL)",) $(if $(BURN_API_KEY_ENV),--api-key-env "$(BURN_API_KEY_ENV)",) $(if $(BURN_CODEX_BIN),--codex-bin "$(BURN_CODEX_BIN)",)

BURN_SEED_FILE = $(BURN_DIR)/SEED.md
BURN_CONTEXT_FILE = $(BURN_DIR)/CONTEXT.md
BURN_PIPELINE_FILE = $(BURN_DIR)/pipeline.toml
BURN_INDEX_FILE = $(BURN_DIR)/index.md
BURN_LESSON_FILE = $(BURN_DIR)/LESSON.md
BURN_INSTRUCTIONS_FILE = $(BURN_DIR)/INSTRUCTIONS.md
BURN_WORKSHEET_FILE = $(BURN_DIR)/WORKSHEET.svg
BURN_WORKSHEET_MASKED_FILE = $(BURN_DIR)/WORKSHEET_MASKED.svg
BURN_PROMO_FILE = $(BURN_DIR)/PROMO_PROMPT.md
BURN_PROMO_IMAGE_FILE = $(BURN_DIR)/promo.png
BURN_BANNER_FILE = $(BURN_DIR)/BANNER_PROMPT.md
BURN_BANNER_IMAGE_FILE = $(BURN_DIR)/banner.png
BURN_COVER_FILE = $(BURN_DIR)/COVER_PROMPT.md
BURN_COVER_IMAGE_FILE = $(BURN_DIR)/cover.png
BURN_PAGE_COPY_FILE = $(BURN_DIR)/PAGE_COPY.md
BURN_LANDING_PAGE_FILE = $(BURN_DIR)/LANDING_PAGE.html
BURN_WORKSHEET_PAGE_FILE = $(BURN_DIR)/WORKSHEET_PAGE.md
BURN_GPT_FILE = $(BURN_DIR)/GPT.md
BURN_NEWSLETTER_FILE = $(BURN_DIR)/NEWSLETTER_EMAIL.md
BURN_COMMUNITY_POST_FILE = $(BURN_DIR)/COMMUNITY_POST.md

BURN_GENERATED_FILES = \
	$(BURN_CONTEXT_FILE) \
	$(BURN_PIPELINE_FILE) \
	$(BURN_INDEX_FILE) \
	$(BURN_LESSON_FILE) \
	$(BURN_INSTRUCTIONS_FILE) \
	$(BURN_WORKSHEET_FILE) \
	$(BURN_WORKSHEET_MASKED_FILE) \
	$(BURN_PROMO_FILE) \
	$(BURN_PROMO_IMAGE_FILE) \
	$(BURN_BANNER_FILE) \
	$(BURN_BANNER_IMAGE_FILE) \
	$(BURN_COVER_FILE) \
	$(BURN_COVER_IMAGE_FILE) \
	$(BURN_PAGE_COPY_FILE) \
	$(BURN_LANDING_PAGE_FILE) \
	$(BURN_WORKSHEET_PAGE_FILE) \
	$(BURN_GPT_FILE) \
	$(BURN_NEWSLETTER_FILE) \
	$(BURN_COMMUNITY_POST_FILE)

.PHONY: all burn-start burn-seed burn-context burn-lesson burn-instructions burn-worksheet burn-worksheet-masked burn-promo burn-promo-image burn-banner burn-banner-image burn-cover burn-cover-image burn-page-copy burn-landing-page burn-worksheet-page burn-gpt burn-index burn-newsletter burn-community-post burn-all burn-dry-run burn-show burn-reset burn-clear-state burn-reseed

all: burn-all

burn-start:
	@rm -f "$(BURN_STATE_FILE)"
	@$(MAKE) --no-print-directory burn-context

burn-seed: $(BURN_SEED_FILE)

burn-context: $(BURN_CONTEXT_FILE)

burn-lesson: $(BURN_LESSON_FILE)

burn-instructions: $(BURN_INSTRUCTIONS_FILE)

burn-worksheet: $(BURN_WORKSHEET_FILE)

burn-worksheet-masked: $(BURN_WORKSHEET_MASKED_FILE)

burn-promo: $(BURN_PROMO_FILE)

burn-promo-image: $(BURN_PROMO_IMAGE_FILE)

burn-banner: $(BURN_BANNER_FILE)

burn-banner-image: $(BURN_BANNER_IMAGE_FILE)

burn-cover: $(BURN_COVER_FILE)

burn-cover-image: $(BURN_COVER_IMAGE_FILE)

burn-page-copy: $(BURN_PAGE_COPY_FILE)

burn-landing-page: $(BURN_LANDING_PAGE_FILE)

burn-worksheet-page: $(BURN_WORKSHEET_PAGE_FILE)

burn-gpt: $(BURN_GPT_FILE)

burn-index: $(BURN_INDEX_FILE)

burn-newsletter: $(BURN_NEWSLETTER_FILE)

burn-community-post: $(BURN_COMMUNITY_POST_FILE)

burn-all: $(BURN_COMMUNITY_POST_FILE)

burn-dry-run: $(BURN_CONTEXT_FILE)
	$(BURN_PIPELINE_CMD) $(BURN_PROVIDER_ARGS) run \
		--pipeline "$(BURN_PIPELINE_FILE)" \
		--dry-run

burn-show: $(BURN_STATE_FILE)
	@printf '%s\n' "Title: $(CODEX_BURN_TITLE)"
	@printf '%s\n' "Date: $(CODEX_BURN_DATE)"
	@printf '%s\n' "Slug: $(CODEX_BURN_SLUG)"
	@printf '%s\n' "Directory: $(BURN_DIR)"

burn-reset: $(BURN_STATE_FILE)
	@rm -f $(BURN_GENERATED_FILES)
	@printf '%s\n' "Cleared generated files for $(BURN_DIR)"

burn-clear-state:
	@rm -f "$(BURN_STATE_FILE)"
	@printf '%s\n' "Cleared active burn state"

burn-reseed: $(BURN_STATE_FILE)
	@rm -f "$(BURN_SEED_FILE)" "$(BURN_CONTEXT_FILE)" "$(BURN_PIPELINE_FILE)" "$(BURN_INDEX_FILE)"
	@$(MAKE) --no-print-directory burn-context

$(BURN_STATE_FILE):
	@mkdir -p "$(dir $(BURN_STATE_FILE))"
	@$(PYTHON) "$(BURN_STATE_SCRIPT)" \
		--state-file "$(BURN_STATE_FILE)" \
		--target-root "$(BURN_TARGET_ROOT)" \
		--default-date "$(DEFAULT_BURN_DATE)"

$(BURN_SEED_FILE): $(BURN_STATE_FILE)
	@mkdir -p "$(BURN_DIR)"
	@printf '%s\n' "Paste seed content for $(CODEX_BURN_TITLE)." "Finish with a line containing only EOF."
	@{ \
		while IFS= read -r line; do \
			[ "$$line" = "EOF" ] && break; \
			printf '%s\n' "$$line"; \
		done; \
	} > "$@"
	@printf '%s\n' "Wrote: $@"

$(BURN_CONTEXT_FILE): $(BURN_SEED_FILE)
	$(BURN_PIPELINE_CMD) $(BURN_PROVIDER_ARGS) seed-production \
		--seed-file "$(BURN_SEED_FILE)" \
		--title "$(CODEX_BURN_TITLE)" \
		--slug "$(CODEX_BURN_SLUG)" \
		--date "$(CODEX_BURN_DATE)" \
		--target-root "$(BURN_TARGET_ROOT)" \
		--force

$(BURN_LESSON_FILE): $(BURN_CONTEXT_FILE)
	$(BURN_PIPELINE_CMD) $(BURN_PROVIDER_ARGS) run \
		--pipeline "$(BURN_PIPELINE_FILE)" \
		--step-id lesson \
		--force

$(BURN_INSTRUCTIONS_FILE): $(BURN_LESSON_FILE)
	$(BURN_PIPELINE_CMD) $(BURN_PROVIDER_ARGS) run \
		--pipeline "$(BURN_PIPELINE_FILE)" \
		--step-id instructions \
		--force

$(BURN_WORKSHEET_FILE): $(BURN_INSTRUCTIONS_FILE)
	$(BURN_PIPELINE_CMD) $(BURN_PROVIDER_ARGS) run \
		--pipeline "$(BURN_PIPELINE_FILE)" \
		--step-id worksheet \
		--force

$(BURN_WORKSHEET_MASKED_FILE): $(BURN_WORKSHEET_FILE)
	$(BURN_PIPELINE_CMD) $(BURN_PROVIDER_ARGS) run \
		--pipeline "$(BURN_PIPELINE_FILE)" \
		--step-id worksheet_masked \
		--force

$(BURN_PROMO_FILE): $(BURN_WORKSHEET_MASKED_FILE)
	$(BURN_PIPELINE_CMD) $(BURN_PROVIDER_ARGS) run \
		--pipeline "$(BURN_PIPELINE_FILE)" \
		--step-id promo \
		--force

$(BURN_PROMO_IMAGE_FILE): $(BURN_PROMO_FILE)
	$(BURN_PIPELINE_CMD) $(BURN_PROVIDER_ARGS) run \
		--pipeline "$(BURN_PIPELINE_FILE)" \
		--step-id promo_image \
		--force

$(BURN_BANNER_FILE): $(BURN_PROMO_IMAGE_FILE)
	$(BURN_PIPELINE_CMD) $(BURN_PROVIDER_ARGS) run \
		--pipeline "$(BURN_PIPELINE_FILE)" \
		--step-id banner \
		--force

$(BURN_BANNER_IMAGE_FILE): $(BURN_BANNER_FILE)
	$(BURN_PIPELINE_CMD) $(BURN_PROVIDER_ARGS) run \
		--pipeline "$(BURN_PIPELINE_FILE)" \
		--step-id banner_image \
		--force

$(BURN_COVER_FILE): $(BURN_BANNER_IMAGE_FILE)
	$(BURN_PIPELINE_CMD) $(BURN_PROVIDER_ARGS) run \
		--pipeline "$(BURN_PIPELINE_FILE)" \
		--step-id cover \
		--force

$(BURN_COVER_IMAGE_FILE): $(BURN_COVER_FILE)
	$(BURN_PIPELINE_CMD) $(BURN_PROVIDER_ARGS) run \
		--pipeline "$(BURN_PIPELINE_FILE)" \
		--step-id cover_image \
		--force

$(BURN_PAGE_COPY_FILE): $(BURN_COVER_IMAGE_FILE)
	$(BURN_PIPELINE_CMD) $(BURN_PROVIDER_ARGS) run \
		--pipeline "$(BURN_PIPELINE_FILE)" \
		--step-id page_copy \
		--force

$(BURN_LANDING_PAGE_FILE): $(BURN_PAGE_COPY_FILE)
	$(BURN_PIPELINE_CMD) $(BURN_PROVIDER_ARGS) run \
		--pipeline "$(BURN_PIPELINE_FILE)" \
		--step-id landing_page \
		--force

$(BURN_WORKSHEET_PAGE_FILE): $(BURN_LANDING_PAGE_FILE)
	$(BURN_PIPELINE_CMD) $(BURN_PROVIDER_ARGS) run \
		--pipeline "$(BURN_PIPELINE_FILE)" \
		--step-id worksheet_page \
		--force

$(BURN_GPT_FILE): $(BURN_WORKSHEET_PAGE_FILE)
	$(BURN_PIPELINE_CMD) $(BURN_PROVIDER_ARGS) run \
		--pipeline "$(BURN_PIPELINE_FILE)" \
		--step-id gpt \
		--force

$(BURN_INDEX_FILE): $(BURN_GPT_FILE)
	$(BURN_PIPELINE_CMD) $(BURN_PROVIDER_ARGS) run \
		--pipeline "$(BURN_PIPELINE_FILE)" \
		--step-id index \
		--force

$(BURN_NEWSLETTER_FILE): $(BURN_INDEX_FILE)
	$(BURN_PIPELINE_CMD) $(BURN_PROVIDER_ARGS) run \
		--pipeline "$(BURN_PIPELINE_FILE)" \
		--step-id newsletter_email \
		--force

$(BURN_COMMUNITY_POST_FILE): $(BURN_NEWSLETTER_FILE)
	$(BURN_PIPELINE_CMD) $(BURN_PROVIDER_ARGS) run \
		--pipeline "$(BURN_PIPELINE_FILE)" \
		--step-id community_post \
		--force
