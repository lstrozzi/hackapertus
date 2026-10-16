# Hack Apertus submission — project root Makefile.
# `make run` builds and starts the project in Docker from a clean checkout.
#
# Two run plans, one `make run` target (docs/RUN_PLANS.md):
#   make run              Plan Local (default) — full local stack: this app container plus the
#                         official UNMODIFIED Thalabus platform + model images, REST-only
#                         access, air-gapped at runtime. Needs Docker and a Linux ARM64
#                         host of the tested class (NVIDIA DGX Spark: GB10, >=128 GB RAM,
#                         >=96 GB GPU memory); images pull at start. Preflight hard-fails
#                         with the tested-platform banner when the host does not qualify.
#   make run PLAN=SaaS    Plan SaaS — app container only, one env var points the same app at a
#                         running Thalabus instance (default https://demo.thalabus.ai).
#                         Needs an instance token and LLM Lite key; requires network access.
#
# Targets:
#   make run | run-local | run-saas | run-a | run-b (legacy aliases) | build | stop | logs | health | keys | lint
#
# Configuration precedence: `make run VAR=…` > process env > .env > plan default.

.PHONY: run run-local run-saas run-a run-b build stop logs health preflight keys lint verify

PORT ?= 8000
THALABUS_PORT ?= 8080
COMPOSE ?= docker compose
PLAN ?= local
# Canonical values are `local` and `SaaS`; `a`/`b` are accepted as legacy aliases.
PLAN_KIND := $(if $(filter SaaS saas b,$(PLAN)),saas,$(if $(filter local a,$(PLAN)),local,$(PLAN)))

# The only difference between the plans: which compose files are loaded (i.e. whether
# the platform is local) and which endpoint the app is configured with.
ifeq ($(PLAN_KIND),local)
COMPOSE_FILES := -f docker-compose.yml -f docker-compose.plan-local.yml
THALABUS_PLAN_DEFAULT := local
THALABUS_BASE_URL_DEFAULT := http://thalabusweb:8080
THALABUS_UI_URL_DEFAULT := http://localhost:$(THALABUS_PORT)
PLAN_NOTE := Plan Local — full air-gapped local stack (platform images pulled at start)
else ifeq ($(PLAN_KIND),saas)
COMPOSE_FILES := -f docker-compose.yml -f docker-compose.plan-saas.yml
THALABUS_PLAN_DEFAULT := saas
THALABUS_BASE_URL_DEFAULT := https://demo.thalabus.ai
THALABUS_UI_URL_DEFAULT := https://demo.thalabus.ai
PLAN_NOTE := Plan SaaS — app only, platform at $(THALABUS_BASE_URL_DEFAULT) (needs reachability)
else
$(error Unknown PLAN '$(PLAN)' — use PLAN=local or PLAN=SaaS)
endif

export PORT THALABUS_PORT EMBEDDINGS_URL RERANKER_URL
export SAAS_EMBEDDINGS_URL SAAS_EMBEDDINGS_MODEL SAAS_RERANKER_URL SAAS_RERANKER_MODEL
export THALABUS_PLAN ?= $(THALABUS_PLAN_DEFAULT)
export THALABUS_BASE_URL ?= $(THALABUS_BASE_URL_DEFAULT)
export THALABUS_UI_URL ?= $(THALABUS_UI_URL_DEFAULT)
export THALABUS_TOKEN NEO4J_PASSWORD

run: preflight
	@sh scripts/hardware-check.sh --banner
	@echo ""
	$(COMPOSE) $(COMPOSE_FILES) up --build -d
	@echo ""
	@echo "$(PLAN_NOTE)"
	@echo "GUI:      http://localhost:$(PORT)"
	@echo "Platform: $(THALABUS_BASE_URL)    (browser: $(THALABUS_UI_URL))"
	@echo "Check:    make health   — then press 'Test model — Hello World' in the GUI"

run-local:
	@$(MAKE) run PLAN=local

run-saas:
	@$(MAKE) run PLAN=SaaS

# Legacy aliases (former plan names A and B).
run-a:
	@$(MAKE) run PLAN=local

run-b:
	@$(MAKE) run PLAN=SaaS

preflight:
	@command -v docker >/dev/null 2>&1 || { echo "ERROR: docker is required but was not found in PATH"; exit 1; }
ifeq ($(PLAN_KIND),local)
	@sh scripts/hardware-check.sh
	@if [ -z "$$NEO4J_PASSWORD" ] && ! grep -qs '^NEO4J_PASSWORD=.' .env; then \
		echo "ERROR: NEO4J_PASSWORD is required by Plan Local — set it in .env or 'make run NEO4J_PASSWORD=…'"; exit 1; \
	fi
	@if [ ! -s keys/python-api-jwt-private.pem ]; then echo "NOTE: keys/ not found — run 'make keys' first (Plan Local needs the platform JWT key pair)"; fi
	@if test -n "$$(docker network ls -q)" && docker network inspect $$(docker network ls -q) 2>/dev/null | grep -qs '172\.20\.0\.0'; then echo "NOTE: 172.20.0.0/24 is already claimed (a local dev platform stack or a leftover hackathon stack) — stop it: Plan Local creates ThalabusHackathonNetwork on that subnet"; fi
endif
ifeq ($(PLAN_KIND),saas)
	@if [ -z "$$THALABUS_TOKEN" ] && ! grep -qs '^THALABUS_TOKEN=.' .env; then echo "NOTE: Plan SaaS has no THALABUS_TOKEN — /api/platform/status will report unauthenticated; instance access is by agreement with T2B"; fi
	@echo "NOTE: Plan SaaS depends on a remote instance — it is the documented fallback, Plan Local is the submitted run path"
endif

build:
	$(COMPOSE) $(COMPOSE_FILES) build

stop:
	$(COMPOSE) -f docker-compose.yml -f docker-compose.plan-local.yml down --remove-orphans

logs:
	$(COMPOSE) $(COMPOSE_FILES) logs -f app

health:
	@curl -fsS http://localhost:$(PORT)/api/health; echo ""
	@curl -fsS http://localhost:$(PORT)/api/platform/status; echo ""

# Plan Local mounts a platform JWT key pair; generate it locally, never commit it.
keys:
	mkdir -p keys
	openssl genrsa -out keys/python-api-jwt-private.pem 2048
	openssl rsa -in keys/python-api-jwt-private.pem -pubout -out keys/python-api-jwt-public.pem

# Documentation versus code, which is the failure mode this project actually had: routes and
# settings were written into the README that the app never served, and nothing noticed because
# the app simply 404s. Every route named in the docs has to exist, every name in .env.example
# has to be read, and with the app up the two databases have to be answered apart.
verify:
	@python3 scripts/verify_submission.py --live

# Release gate: every *mirror* image (name ends in
# `-hackathon`) must be digest-pinned; the only other `thalabus/` images allowed
# are the three PUBLIC images (neo4j, embeddings_server, vllm_server) — they carry
# no platform code and float on :latest. Nothing may be vendored.
# Image lines only are inspected -- a comment must never satisfy a gate.
lint:
	@if grep -Eq '^[[:space:]]*image:.*-hackathon' docker-compose.plan-local.yml && ! grep -Eq '^[[:space:]]*image:.*-hackathon(:[^@[:space:]]+)?@sha256:[0-9a-f]{64}' docker-compose.plan-local.yml; then \
		echo "ERROR: Plan Local -hackathon mirror images are not pinned to a 64-hex digest - required before submission"; exit 1; \
	fi
	@if grep -E '^[[:space:]]*image:' docker-compose.plan-local.yml | grep -E 'thalabus/' | grep -vE 'thalabus/(neo4j|embeddings_server|vllm_server)(:|@|$)' | grep -q .; then \
		echo "ERROR: unexpected thalabus/ image - allowed are *-hackathon mirrors plus public neo4j/embeddings_server/vllm_server only"; exit 1; \
	fi
	@if grep -Eq 'image:.*hackathon-' docker-compose.plan-local.yml; then \
		echo "ERROR: '-hackathon' appears in a tag - the word belongs in the image name only"; exit 1; \
	fi
	@if find . -name '*.tar' -o -name '*.tar.gz' | grep -q .; then echo "ERROR: vendored image archives in tree"; exit 1; fi
	@echo "lint ok"
