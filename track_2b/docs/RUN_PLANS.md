# Run plans — what `make run` starts

One target, two run plans. The submitted run path is **Plan Local** because the track
requires on-premise / air-gapped / sovereign-Swiss deployability; **Plan SaaS** is the
documented fallback for a machine that cannot host the full stack.

|                       | Plan Local (`make run`)         | Plan SaaS (`make run PLAN=SaaS`)                |
|-----------------------|---------------------------------|--------------------------------------------------|
| Containers            | app + own Neo4j + platform + model servers | app + own Neo4j                     |
| Thalabus endpoint     | `http://thalabusweb:8080`       | `https://demo.thalabus.ai` (or your own instance) |
| Runtime network       | LLM endpoint only               | LLM, embeddings and reranker endpoints + the remote platform |
| Air-gapped claim      | holds (no other egress)         | does not hold — needs reachability               |
| Host prerequisites    | Docker, Linux **ARM64**, ≥128 GB RAM, NVIDIA GPU ≥96 GB (tested on DGX Spark) | Docker, session token, LLM Lite API key |
| Judges need           | nothing but Docker             | credentials obtained by agreement               |

Under Plan SaaS, step 2 displays and probes the configured Apertus chat endpoint
(`LLM_NAME` / `LLM_BASE_URL`); step 3 probes embeddings and reranking at the configured
`SAAS_*` URLs and model IDs, authenticating with `LLM_API_KEY`. Those routes and model IDs
must be enabled for the key by the LiteLLM provider; failures are shown by the wizard and
do not count as verified. Under Plan SaaS, step 4 reads `not required in this hackathon`
and offers Continue without sending an Engine request. Plan Local continues to
use its in-network model service containers.

## Tested hardware — and what happens on any other machine

> The provided images have been tested on NVIDIA DGX Spark machines with ARM64 GB10
> CPU+GPU with 128GB RAM. We cannot guarantee that they work on any other machine or
> architecture and we have provided a SaaS running instance on said NVIDIA machine.

> The solution can run fully locally, but the job to adapt to different hardware may
> require 1 to 5 days of work and are out of scope of the hackathon, first of all
> because your hardware specifications are unknown, secondly because of the needed
> time, thirdly because an environment is not provided.

Plan Local in numbers: **Linux, ARM64 (aarch64), ≥128 GB system RAM, NVIDIA GPU with
≥96 GB memory, Docker with Compose v2+.** The architecture requirement is hard rather
than stylistic: the vLLM model images publish a `linux/arm64` manifest only.

`make run` enforces all of this *before pulling or starting anything*, via
`scripts/hardware-check.sh` (invoked by `preflight`): an unqualified host receives one
ERROR block that names every unmet requirement, repeats this banner, and points at the
fallback above. It never half-starts. The fallback and escape hatches:

```sh
make run PLAN=SaaS               # SaaS plan — runs on any Docker host (needs credentials)
THALABUS_HW_CHECK=warn make run  # keep the check, downgrade the ERROR to a WARNING
THALABUS_HW_CHECK=0 make run     # skip the check (organisers / known-good CI)
```

## Why a single `make run` covers both

`make run` is one target with one switch, `PLAN`. Everything else follows from it,
because the difference between the plans is **only configuration**:

1. **Same image.** `src/Dockerfile` builds the app once; it contains no platform code.
2. **Two compose files, one variable.** `PLAN=local` passes `-f docker-compose.yml
   -f docker-compose.plan-local.yml` (the overlay *adds* the platform services and puts the
    app on their network *as well as* the default one, because its own graph store lives
    there); `PLAN=SaaS` passes the base file and SaaS defaults
    overlay. Compose merges files, so the app service is declared once.
3. **One app-side switch.** The app resolves the endpoint from `THALABUS_PLAN`
   (see `src/app/thalabus.py`): plan `local` → the in-network service name, plan `saas` →
   the remote instance. An explicit `THALABUS_BASE_URL` still wins, so a third
   instance is one env var away without touching code.
4. **The platform's own settings are applied over the API, not by mounting files.**
   Exactly one plan does this from the GUI: **Plan SaaS**, where the model card posts the
   `SAAS_CFG_*` preset to `POST /v1/api/platform/config/model` on the remote instance and
   reads the redacted result back. **Plan Local never calls that route** -- the platform
   there is this submission's own stack, its LLM is configured by the environment the
   container is created with, and the local card is read-only by decision. Nothing in this
   tree edits or ships platform configuration files.
   *(Superseded wording kept on record: this item once claimed the app reconfigured a local
   platform after boot from the `LLM_*` trio. That was never implemented.)*
**Two databases, asked about separately.** Plan Local runs two graph databases — this submission's
own (`graph`, database `neo4j`) and the platform's (`neo4j`, database `platform`). The app answers
for both, and separates four states that a single flag would flatten:

| State at the platform's address | Reported as |
|---------------------------------|-------------|
| `platform` present and queried | reached |
| a server answers, `platform` absent | "not available there", with the databases that *are* visible, and the note that the platform creates it on first start |
| nothing answers | unreachable |
| the address is this app's own store | `same_store_as_app`, and the probe is refused |

The second row is the state a freshly created platform stack is really in, and treating it as
"unreachable" would be wrong in a way a reader could act on incorrectly: it suggests restarting
a service that is running. The refusal matters because both stacks ship with identical default
credentials, so a copy-paste of `NEO4J_URI` into `NEO4J_PLATFORM_URI` would otherwise return a
truthful answer about the wrong database. Under Plan SaaS the field is present with
`requested: false`, never omitted, because an absent field invites a reader to invent the reason.Two graph databases exist in Plan Local, and the app answers for each one separately. Its own
(`NEO4J_URI`, service `graph`) is where this submission stores its single wizard-state node;
the platform's (`NEO4J_PLATFORM_URI`, default `bolt://neo4j:7687`, database `platform`) holds
the platform's cases. `GET /api/neo4j/status` reports both, and one flag could not cover them:
a reachable server and an existing `platform` database are two facts, and the submission's
store contains `neo4j` and `system` only, so a check that borrowed the app's own answer would
report the platform's database as absent on a deployment that has one. Pointing
`NEO4J_PLATFORM_URI` at `NEO4J_URI` is therefore refused outright (`same_store_as_app`), and
under Plan SaaS the field is still returned with `requested: false` rather than omitted.
Consequences worth knowing, and how they are handled:

* **The platform may not be up yet when the app starts.** No `depends_on` condition
  is used (that needs a recent Compose version); the app answers `/api/health` with
  `200` regardless and reports the platform state under `"platform"`, so a first-run
  "not reachable yet" line is a waiting state, not a failure. `make health` prints
  the app view and the resolved platform view.
* **A small or wrong-architecture machine cannot host Plan Local.** The `preflight`
  target runs `scripts/hardware-check.sh`, which hard-fails a non-ARM64 host, a host
  below the RAM threshold, or a host without a ≥96 GB NVIDIA GPU — always printing
  the tested-platform banner and the `make run PLAN=SaaS` fallback, never a silent
  half-start (see “Tested hardware” above).
* **Plan Local is stricter about secrets.** `preflight` refuses to start Plan Local without
  `NEO4J_PASSWORD` and hints at `make keys` for the platform JWT key pair, while
  teardown (`make stop`) works without any secret — the compose file therefore uses
  a fallback default for the database password rather than a hard requirement.
* **What must not be assumed.** Plan Local creates and owns its own network
  `ThalabusHackathonNetwork` and never joins a pre-existing platform network,
  but it still needs `172.20.0.0/24` (two bridge networks may not share a subnet)
  and its published ports to be free, and it assumes the public event mirrors of
  the platform images are pullable anonymously.

## Platform images (Plan Local)

`docker-compose.plan-local.yml` references official, **unmodified** images by name (and,
for the two mirror services, eventually pinned digest) only — it builds nothing,
vendors no archive, and pulls at start time. `make lint` enforces the digest pinning
of the mirrors and the no-archives rule before release, so it is **red today**: the
two mirror digests arrive with the one-off mirror publication and land here in the
same commit that records them.

| Role                          | Reference                                                     | Publication mode                                   |
|-------------------------------|---------------------------------------------------------------|----------------------------------------------------|
| Web/API front end             | `thalabus/thalabusweb-hackathon:2026-10@sha256:…` (pending)   | frozen one-off `-hackathon` mirror                    |
| Python services               | `thalabus/pythonserver-hackathon:2026-10@sha256:…` (pending)  | frozen one-off `-hackathon` mirror                    |
| Graph database                | `thalabus/neo4j:latest`                                       | **public repository** (unmodified upstream software)  |
| Embeddings (vLLM)             | `thalabus/embeddings_server:latest`                           | **public repository** (no platform code in the image) |
| Reranking (vLLM)              | `thalabus/vllm_server:latest`                                 | **public repository** (no platform code in the image) |

The `-hackathon` repositories are one-off public mirrors of the official release
build at a fixed digest; the main repositories stay private, so publishing them
does not release anything beyond that digest. Rebuilds and forks are deliberately
not used — a rebuilt image would no longer be the official artefact.

**Two model services, two images:** the embeddings server and the reranker server
are separate vLLM-based services and never re-use one another's image:
`thalabus/embeddings_server` and `thalabus/vllm_server`. Both wrappers add no
platform code (public NVIDIA vLLM base + apt utilities + pip fixes), so they are
published **publicly at `:latest`** instead of getting frozen mirrors — which also
lets anyone verify and rebuild them from their public Dockerfiles. The reranker's
container start command in `docker-compose.plan-local.yml` is the matching vLLM pooling
serve (`vllm serve BAAI/bge-reranker-v2-m3 … --runner pooling`, port 8364, API
`POST /v1/score`).

What still follows and is **not yet settled**:

* **The two mirror digests are missing.** The mirrors are published as a separate
  one-off step; until then the two platform defaults are tag-pinned only,
  so `make lint` is **expected to fail** and `make run` Plan Local cannot pull. Each
  reference becomes `…-hackathon:2026-10@sha256:…` in the same commit that records
  the published digest. (The three public images — neo4j and the two model
  wrappers — need no digest: floating `:latest` is their accepted form.)
* **Model weights must be pre-baked.** Plan Local is air-gapped at runtime, so the vLLM
  services mount `./models-cache` as the Hugging Face cache; whoever prepares a
  machine populates it once (`Qwen/Qwen3-Embedding-0.6B`, `BAAI/bge-reranker-v2-m3`).

## Run-preconditions (both plans)

* A **Thalabus instance**: started by `make run` in Plan Local, or access to a running
  one in Plan SaaS. Access is by separate agreement with the platform owner and is not
  part of this submission.
* An **OpenAI-compatible endpoint** for `LLM_NAME` / `LLM_BASE_URL` / `LLM_API_KEY`.
  Plan SaaS defaults to `apertus-1.5-70b` at `https://llm.thalabus.ai/v1` — that id is
  what the endpoint's own catalogue returns, and the `/v1` path is part of the value
  because the app appends `/chat/completions` to it. Supply `LLM_API_KEY` in `.env`.
  Explicit environment or `.env` values override the defaults.
* A **graph store**: `docker compose` starts `neo4j:5-community` as service `graph` and
  the app connects to `NEO4J_URI` (default `bolt://graph:7687`, password
  `NEO4J_PASSWORD`, shared with the platform's database in Plan Local). Nothing to
  provision — but the image is ~700 MB to download and the port must be free, so stop
  any Neo4j already running on this host or point `NEO4J_URI` elsewhere.
* `curl` and `openssl` for `make health` / `make keys` respectively.

## Commands

```sh
make run                 # Plan Local — full local stack (submitted path)
make run PLAN=SaaS       # Plan SaaS — app + its own graph store, remote platform
make health              # app health + resolved platform status (add the same PLAN you ran)
make logs                # follow the app container
make stop                # stop everything, no secrets needed
make keys                # generate the local platform JWT key pair (git-ignored)
make lint                # release gate: pinned digests, nothing vendored
```
