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
# The GUI port is the one exception: it is a literal in the compose files, so no
# command-line variable, environment variable, or .env line can move it.

.PHONY: run run-local run-saas run-a run-b build stop logs health preflight keys lint verify

# The GUI port is a literal in the compose files (base 8082, overridden per plan in
# the plan overlay), not a Makefile setting — it is the one value `make run VAR=…`
# therefore cannot override. The two numbers below mirror those literals for the
# banner and `make health` only; compose never reads them.
APP_GUI_PORT := 8082
PORT := $(APP_GUI_PORT)
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

# PORT and NEO4J_PASSWORD are deliberately NOT exported. A plain `export` publishes
# even an unset or mirrored name into compose's process environment, where it outranks
# .env: exporting an empty NEO4J_PASSWORD made the app fall back to the compose default
# password and the graph store answered `unauthorized` (observed live). A file that
# declares a setting should not be silently outranked by an unset process variable.
export THALABUS_PORT EMBEDDINGS_URL RERANKER_URL
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
	@echo "GUI:      http://localhost:$(APP_GUI_PORT)   (compose publishes it; the Makefile holds no port default)"
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

# The published GUI port is read from `compose config`, so the plan overlay decides it
# and this file mirrors nothing — no second copy to fall out of step. Two Makefile traps
# shaped this line, both of which fail silently: a comma inside $(shell …) is read as a
# separator of the enclosing call, and a literal double quote closes the recipe's own
# quoting (observed: the file argument arrived truncated to "-f"). Hence no double quotes
# and no commas anywhere below, and `sed -n /…/-1p` in place of `| head -1`.
GUI_PORT = $(shell $(COMPOSE) $(COMPOSE_FILES) config 2>/dev/null | sed -n -e '/target: 8000/{$!{n;p}}' | sed -n -e '/published/{s/[^0-9]//g;p;q}' 2>/dev/null)

health:
	@test -n "$(GUI_PORT)" || { echo 'ERROR: could not read the published port — is docker running? pass the same PLAN as make run'; exit 1; }
	@echo '# published GUI port: $(GUI_PORT)  (resolved by compose for PLAN=$(PLAN))'
	@curl -fsS http://localhost:$(GUI_PORT)/api/health; echo ""
	@curl -fsS http://localhost:$(GUI_PORT)/api/platform/status; echo ""

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
