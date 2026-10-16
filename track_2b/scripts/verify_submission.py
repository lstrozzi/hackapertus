#!/usr/bin/env python3
"""Fail-closed checks over one thing: does the documentation still match the code?

    python3 scripts/verify_submission.py            # static checks, no server needed
    python3 scripts/verify_submission.py --live     # also probe a running app

Why this file exists. Several statements written into the documentation during development
described routes and settings that were never implemented (`/internal/restart`, `/ready`,
`GET /api/config`, a per-action feature flag, and a claim that Plan Local reconfigures the
platform it is talking to). None of them broke a test at the time, because the app never
served a route it did not implement -- the documentation simply drifted ahead of it. The
static half below closes that gap permanently: every `/api/...` the documentation instructs a
reader to call has to exist in `src/app/main.py`, and every environment name it tells a reader
to set has to be read by the code.

The live half asserts the behaviours that were changed most recently and are otherwise only
visible in a browser: that a second database is reported as a separate fact rather than
merged into the first, that secrets come back as a set/unset view instead of a mask with two
meanings, and that progress can move backwards but never forwards.

No third-party dependency, no Docker SDK, nothing that needs an API key: this has to run on a
judge's machine as easily as `make run`.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

TRACK_ROOT: Path = Path(__file__).resolve().parent.parent
MAIN_PY: Path = TRACK_ROOT / "src" / "app" / "main.py"
ENV_EXAMPLE: Path = TRACK_ROOT / ".env.example"
DOC_FILES: List[Path] = [TRACK_ROOT / "README.md", TRACK_ROOT / "docs" / "RUN_PLANS.md",
                         ENV_EXAMPLE, TRACK_ROOT / "src" / "app" / "static" / "index.html"]
PAGE: Path = TRACK_ROOT / "src" / "app" / "static" / "index.html"
# The submission must stand alone, so the host project's stylesheet is consulted only when this
# tree still sits inside it; the colour check below says which case it found.
TBS_CSS: Path = TRACK_ROOT.parents[2] / "ThalabusWeb" / "wwwroot" / "css" / "tbs-design.css"

failures: List[str] = []
checks: int = 0


def report(name: str, ok: bool, detail: str = "") -> None:
    """Count one check, and remember it if it failed."""
    global checks
    checks += 1
    if not ok:
        failures.append(f"{name}{(': ' + detail) if detail else ''}")
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" -- {detail}" if detail and not ok else ""))


def read_text(path: Path) -> str:
    """Read a tracked file, or an empty string when it is absent (reported as a failure)."""
    if not path.is_file():
        report(f"file present: {path.name}", False, "not found")
        return ""
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------- static: routes

def documented_routes() -> List[Tuple[str, str]]:
    """Every backticked `/api/...` path the documentation or the page tells someone to call.

    Paths under the platform's own `/v1/` prefix are excluded: those belong to the platform,
    are not served by this app, and are checked against the platform's OpenAPI file instead.
    """
    found: List[Tuple[str, str]] = []
    pattern = re.compile(r"`((?:/v\d)?/api/[A-Za-z0-9_{}/.-]*)`")
    for doc in DOC_FILES:
        for match in pattern.finditer(read_text(doc)):
            path = match.group(1)
            if path.startswith("/v"):
                continue
            found.append((doc.name, path))
    return found


def served_routes() -> List[str]:
    """Route paths registered in `main.py`, templates normalised to `{...}`."""
    text = read_text(MAIN_PY)
    paths: List[str] = [m.group(2).strip() for m in
                        re.compile(r"@app\.(get|post|put|delete)\(\s*[\"']([^\"']+)[\"']").finditer(text)]
    return [re.sub(r"\{[^}]*\}", "{}", p) for p in paths]


def check_routes() -> None:
    served = served_routes()
    pairs = documented_routes()
    report("documentation names at least one route", bool(pairs), "found none")
    for name, path in pairs:
        normalised = re.sub(r"\{[^}]*\}", "{}", path)
        report(f"{name}: {path} is served by the app", normalised in served,
               f"registered: {', '.join(sorted(served))}")
    # Anti-fabrication: these were written into the docs during development and never built.
    for ghost in ("/internal/restart", "/ready", "`/api/config`", "`/api/graph/status`"):
        for doc in DOC_FILES:
            if doc.name == "technical_report.md":
                continue
            report(f"{doc.name}: no reference to {ghost}", ghost not in read_text(doc))


def code_that_reads_configuration() -> str:
    """Everything able to consume an environment name: app code, compose, Makefile, scripts.

    A name does not have to be read by Python to be real. `NEO4J_URI` and the image overrides
    are consumed by compose variable expansion, `PORT` by the Makefile, and refusing them
    because they are not in `src/app` would report working configuration as a phantom.
    """
    sources: List[Path] = sorted((TRACK_ROOT / "src" / "app").rglob("*.py"))
    sources += sorted(TRACK_ROOT.glob("docker-compose*.yml"))
    sources += [TRACK_ROOT / "Makefile"]
    sources += sorted((TRACK_ROOT / "scripts").glob("*"))
    text: str = ""
    for source in sources:
        if source.is_file():
            text += source.read_text(encoding="utf-8", errors="replace")
    return text


def check_env_names() -> None:
    """Every upper-case name `.env.example` presents as settable must be read somewhere."""
    sample: str = read_text(ENV_EXAMPLE)
    code: str = code_that_reads_configuration()
    for line in sample.splitlines():
        match = re.match(r"(#?\s*)([A-Z][A-Z0-9_]{3,})\s*=", line)
        if match is None:
            continue
        name = match.group(2)
        if name in {"NODE", "PATH", "DEBIAN_FRONTEND", "PIP_NO_CACHE_DIR", "PYTHONUNBUFFERED"}:
            continue
        report(f".env.example: {name} is read somewhere", name in code)


def check_superseded_claims() -> None:
    """Claims that were disproved must stay disproved, and must be marked as history."""
    plans: str = read_text(TRACK_ROOT / "docs" / "RUN_PLANS.md")
    report("RUN_PLANS.md does not claim Plan Local reconfigures the platform",
           "In Plan Local the app configures the platform" not in plans)
    report("the superseded claim is kept on record, not silently deleted", "Superseded" in plans)


# ---------------------------------------------------------------- static: recent fixes

def check_restart_poll() -> None:
    """The first probe has to be later than the restart itself.

    A container comes back in about a second and a process that is being asked to stop still
    answers one final request. Probing at 2 s therefore observes the *old* process succeed,
    and the next probe fails: the button once reported a dead container for a restart that had
    worked. Invariants, not wording: the first wait is at least twice the length of a later
    wait, at least four seconds, and the figure shown to the reader is accumulated rather than
    derived from the attempt index, which would be wrong by the size of that first wait.
    """
    page: str = read_text(TRACK_ROOT / "src" / "app" / "static" / "index.html")
    first_wait = re.search(r"attempt\s*===\s*0\s*\?\s*(\d+)\s*:\s*(\d+)", page)
    report("the first probe waits for the restart to finish", first_wait is not None,
           "no longer-than-the-restart initial wait found")
    if first_wait is not None:
        leading = int(first_wait.group(1))
        interval = int(first_wait.group(2))
        report(f"that initial wait ({leading} s) outlasts the restart", leading >= 4, f"{leading} s")
        report(f"the initial wait ({leading} s) exceeds the repeat interval ({interval} s)",
               leading > interval)
    report("elapsed time is measured rather than inferred from the attempt index",
           "(attempt + 1) * 2" not in page)
    report("the restart poll is bounded", re.search(r"attempt\s*<\s*(\d{2,})", page) is not None)


def check_no_dead_markup() -> None:
    """Screen 5 states that it is empty; markup hinting at operations must not survive next to it."""
    page: str = read_text(TRACK_ROOT / "src" / "app" / "static" / "index.html")
    for ghost in ("opsRunForm", "opsBuildForm", "opsDataset"):
        report(f"no unreferenced {ghost} in the page", ghost not in page)


def check_second_store_cannot_be_conflated() -> None:
    """The guard that refuses to ask the platform's question of the submission's own store.

    Both stacks ship an identical, empty password-less-by-default database, so answering
    ``platform_database.reachable`` from the app's own store would be trivially true and
    completely false at the same time. The code refuses that configuration outright, so the
    refusal has to exist in the source and be reachable from the endpoint.
    """
    store: str = read_text(TRACK_ROOT / "src" / "app" / "graphstore" / "neo4j.py")
    endpoint: str = read_text(MAIN_PY)
    page: str = read_text(TRACK_ROOT / "src" / "app" / "static" / "index.html")
    report("the same-address refusal exists", "same_store_as_app" in store)
    report("the platform probe is served by the status endpoint", "platform_database" in endpoint
           and "probe_platform_database" in endpoint)
    report("the GUI distinguishes answered from unasked", "not asked" in page)
    report("the GUI does not shorten the answer to a single flag",
           "platform_database.exists" not in page or "reachable" in page)


def check_evidence_does_not_cross_plans() -> None:
    """A step waived on one plan cannot arrive pre-verified on the other.

    The live check is this submission's own history: state written while running Plan SaaS,
    where step 3 is waived, was resumed by a Plan Local process and reported step 3 as
    "Verified in this run" -- a completed test that had never been run. The restore now compares
    the plan the state was written on with the live one, drops the steps that only the other
    plan could waive, and reports them.
    """
    code: str = read_text(TRACK_ROOT / "src" / "app" / "wizard.py")
    report("the plan a state was written on is read back", 'w.plan AS plan' in code)
    report("restoring takes the live plan as an argument",
           re.search(r"async def load_state\(neo4j: Neo4jConfig,\s*plan: str\)", code) is not None)
    report("cross-plan evidence is dropped", "forgotten" in code and "stored_plan" in code)
    report("the reader is rewound rather than left past the frontier",
           re.search(r"_STATE\.current_step > min\(waived\)", code) is not None)
    page: str = read_text(TRACK_ROOT / "src" / "app" / "static" / "index.html")
    report("the GUI reports a step it had to forget", "forgotten_steps" in page)


def inline_scripts(page: str) -> List[str]:
    """Every inline `<script>` body in the page."""
    return re.findall(r"<script[^>]*>(.*?)</script>", page, flags=re.DOTALL)


def check_page_script_parses() -> None:
    """The page's own JavaScript has to be parseable, because a syntax error is silent.

    HTML and CSS keep rendering while the script is dead, so a page looks fine and lies: the
    breadcrumb row, the plan line and the restored state all stay at their placeholders. One
    submission had a plain double quote where a backtick belonged, which ended a template
    literal early; every interactive part of the screen was broken and every endpoint in this
    file still answered correctly, because an endpoint says nothing about its client. No
    JavaScript engine exists on this host or in this image, so this cannot be a real parse -- an
    odd number of unescaped backticks is the parity check that catches exactly the class of
    mistake found here, and a syntax error anywhere else must still be caught by opening the page.
    """
    page: str = read_text(PAGE)
    scripts: List[str] = inline_scripts(page)
    report("the page carries an inline script", bool(scripts))
    for number, body in enumerate(scripts, start=1):
        stray: int = len(re.findall(r"(?<!\\)`", body))
        report(f"script {number}: template literals close evenly ({stray} backticks)", stray % 2 == 0)


def check_welcome_screen_marks_its_plan() -> None:
    """The plan in force is visible on the screen that explains the two plans.

    Both plans are described in full on screen 1 and only one of them is running; which one is
    decided by the server, so the highlight is taken from the wizard payload rather than made
    clickable, and cannot be nudged into disagreeing with the tests the guide runs next.
    """
    page: str = read_text(PAGE)
    report("both plan cards are addressable", page.count("data-plan-card=") == 2)
    report("the frame is decided by the plan the server reports",
           re.search(r"dataset\.planCard === WIZ\.plan", page) is not None)
    report("the cards no longer touch", re.search(r"\.grid2\s*\{[^}]*gap", page) is not None)
    report("the welcome button says what it starts", ">Start Environment Verification</button>" in page)
    report("and stands centred on its own line",
           re.search(r"#step1\s+\.step-actions\s*\{[^}]*justify-content:\s*center", page) is not None)
    report("and only that one button was renamed", page.count(">Continue</button>") == 4)
    frame: str = (re.search(r"\.plan-card--current\s*\{([^}]*)\}", page) or re.search(r"(x)", "x")).group(1)
    report("the frame is a frame and not a hint of one", "4px solid" in frame)
    # The first version of this rule was `.step > .card.plan-card--current`, which matches nothing:
    # the cards are inside the .grid2 wrapper. A rule that does not apply still tests green unless
    # the selector is checked against the markup it is meant to reach.
    selectors: List[str] = re.findall(r"([^{}]*)\.plan-card--current[^{]*\{", page)
    report("the frame selector can reach the card it styles",
           bool(selectors) and all(">" not in s for s in selectors),
           " ; ".join(s.strip() for s in selectors)[:90])
    painted: List[str] = [c.lower() for c in re.findall(r"#[0-9a-fA-F]{6}", frame)]
    design: str = read_text(TBS_CSS) if TBS_CSS.is_file() else ""
    active: Optional[re.Pattern[str]] = re.search(r"\.tbs-step-card--active\s*\{([^}]*)\}", design)
    expected: List[str] = [c.lower() for c in re.findall(r"#[0-9a-f]{6}", active.group(1))] if active else []
    if expected:
        report("the frame's blue is the design system's active blue",
               bool(painted) and all(c in expected for c in painted), "tbs-step-card--active: " + ", ".join(expected))
    else:
        report("the frame is painted with one colour (design system stylesheet not alongside this tree)",
               len(painted) == 1)


# The first bytes that decide what an image actually is, so a file's name can be checked against
# its content instead of trusted. A slide exported as PNG but named .jpg would be served as
# `image/jpeg`, and a browser would render it anyway -- leaving the wrong type as a surprise for
# the next reader, cache, or non-sniffing client.
MAGIC_NUMBERS: Dict[str, bytes] = {".png": b"\x89PNG\r\n\x1a\n", ".jpg": b"\xff\xd8\xff",
                                   ".jpeg": b"\xff\xd8\xff", ".gif": b"GIF8", ".webp": b"RIFF"}


def fetch_bytes(path: str, timeout: float = 8.0) -> Tuple[int, str, bytes]:
    """GET one URL as bytes, returning its status, declared content type and body."""
    try:
        with urllib.request.urlopen(base_url() + path, timeout=timeout) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Content-Type", ""), b""
    except (urllib.error.URLError, OSError):
        return 0, "", b""


def check_slides_carry_their_command_and_their_caveat() -> None:
    """Each slide says which command produces it, and the risk is written down somewhere readable.

    Three claims, none of them about pixels:

        * The exact invocation is printed below each plan card; `PLAN=SaaS` is case-sensitive, and
          lowercase `plan=saas` does not select the SaaS compose overlay.
    * The caveat about the cost of Plan Local was drawn onto the slide, where a screen reader, a
      renderer with images off, and this script cannot see it. In text it is legible to all three.
    * The plan commands are status-only radio labels; the server-reported plan is checked.
    """
    page: str = read_text(PAGE)
    report("both plan commands appear in the outside-card radio group",
           page.count('class="plan-run-option"') == 2
           and page.index('class="plan-run-options"') > page.index('class="grid2"'))
    report("the local command label is exact", "<span>make run</span>" in page)
    report("the saas command label uses the exact Make setting",
           "<span>make run PLAN=SaaS</span>" in page)
    report("the obsolete This run summary is removed", 'id="welcomePlan"' not in page)
    report("the plan selector reflects the server plan",
           'radio.checked = radio.value === WIZ.plan' in page)
    risk_rule: str = (re.search(r"\.plan-risk\s*\{([^}]*)\}", page) or re.search(r"(x)", "x")).group(1)
    risk_text: str = page[page.find("plan-risk"):] if "plan-risk" in page else ""
    report("the stated cost is text, not paint",
           '"make run" might take up to one hour to configure and validate' in risk_text
           and "untested and unconfigured host environment" in risk_text)
    # The estimate was a claim twice over -- "2-5 days", then "up to one day", both milder -- and
    # each time the slide, the check and the README had to move together. The old numbers may not
    # survive in any of them, so a drift reappearing as prose is caught here rather than admired.
    report("and no retired estimate survived the rewording",
           "2-5 days" not in page and "up to one day" not in page)
    report("and it is set apart as a caution rather than as prose",
           "background" in risk_rule and "border" in risk_rule)
    report("and it is only attached to the plan that carries it",
           page.count("class=\"plan-risk\"") == 1)
    report("the local card still describes what runs there", "embeddings server and a reranker" in page)
    report("the welcome screen stopped quoting Cypher at the reader",
           re.search(r"the node is \$\{p\.node\}", page) is None
           and "Progress is stored in the graph store; the node" not in page)


def check_diagrams_are_shipped() -> None:
    """The two deployment slides are drawn on the welcome screen, and are what they claim to be.

    "Pictures were added to the page" is not one claim but four, and the markup alone proves none
    of them: the files have to exist where the app mounts `/static`, their names have to agree with
    their bytes, each one has to be described for someone who cannot see it, and the CSS has to
    limit their size without editing their shape. The live check under `--live` adds the fifth --
    that a running container serves the same bytes -- because the image bakes its source in, and a
    browser shows a broken frame rather than saying which build it came from.
    """
    page: str = read_text(PAGE)
    references: List[str] = sorted(set(re.findall(r"/static/([\w./+-]+)", page)))
    slides: List[str] = [r for r in references if "Deployment" in r]
    report("the page draws both deployment slides", len(slides) == 2, ", ".join(slides))
    for reference in references:
        on_disk: Path = TRACK_ROOT / "src" / "app" / "static" / reference
        report(f"{reference} is where /static is mounted", on_disk.is_file())
        if not on_disk.is_file():
            continue
        suffix: str = on_disk.suffix.lower()
        body: bytes = on_disk.read_bytes()
        expected: bytes = MAGIC_NUMBERS.get(suffix, b"")
        if not expected:  # a format with no fixed signature, such as svg: nothing to compare
            report(f"{reference} carries {len(body)} bytes", len(body) > 0)
            continue
        report(f"{reference} is really a {suffix} file", body.startswith(expected),
               f"{len(body)} bytes, starts {body[:4].hex()}")
    report("every image says what it shows",
           bool(re.findall(r"<img", page)) and not re.findall(r'\salt="\s*"', page))
    rule: str = (re.search(r"\.plan-diagram\s*\{([^}]*)\}", page) or re.search(r"(x)", "x")).group(1)
    report("the slides are capped at 720 px", "max-width: 720px" in rule)
    # Asserted as an absence, because side by side is what the shared .grid2 rule already does: the
    # thing that broke it was a #step1 override forcing one column, so that override is the regression.
    report("and no rule forces the two slides onto separate rows",
           re.search(r"#step1\s*\.grid2\s*\{", page) is None)
    report("and their ratio comes from the file, not from here",
           "height: auto" in rule and page.count('width="1600" height="900"') == 2)


def check_the_wide_column_stays_readable() -> None:
    """The page is wide for the diagrams, which must not become wide for the sentences.

    The column was widened so the deployment slides could be read, and a paragraph inherits every
    pixel of a column it is given: at 1600 px a line runs past 200 characters and no one can follow
    it back to the next line. So two things have to hold at once -- one width for the whole page,
    because `main` and `.crumbs` each carrying their own is how the breadcrumb came to drift out
    from under the content -- and a measure on the prose that is written in characters rather than
    pixels, so it stays the same measure however wide the page is allowed to become.
    """
    page: str = read_text(PAGE)
    declared: List[str] = re.findall(r"--page-width:\s*([^;]+);", page)
    report("the page has one width, declared once", len(declared) == 1, " ; ".join(declared))
    report("that width is generous but bounded",
           bool(declared) and re.search(r"min\(\s*1[2-9]\d\dpx\s*,\s*100%\s*\)", declared[0]) is not None,
           declared[0] if declared else "no --page-width")
    report("and it does not count a scrollbar it cannot use", "100vw" not in (declared[0] if declared else ""))
    for rule_name in ("main", ".crumbs"):
        # Not `\b`: there is no word boundary before a `.` selector, and a matcher that cannot find
        # a rule reports it as missing. The guard also stops `.crumbs li {` reading as `.crumbs {`.
        block: Optional[re.Match[str]] = re.search(
            r"(?<![\w.])" + re.escape(rule_name) + r"\s*\{([^}]*)\}", page)
        body: str = block.group(1) if block else ""
        report(f"{rule_name} takes that width", "max-width: var(--page-width)" in body)
        report(f"{rule_name} keeps its padding inside it", "box-sizing: border-box" in body)
    # Not "860px does not appear": the comment above explains why it was dropped, and prose that
    # names a superseded number is documentation rather than a second source of truth.
    report("no rule still sets the old 860 px column",
           not re.search(r"(max-)?width:\s*860px", page))
    prose_block: Optional[re.Match[str]] = re.search(r"\.step\s*>\s*p\s*\{([^}]*)\}", page)
    prose: str = prose_block.group(1) if prose_block else ""
    report("prose is capped by a measure, not by the page width",
           re.search(r"max-width:\s*\d+ch", prose) is not None, prose.strip()[:60])


# ---------------------------------------------------------------- live

def base_url() -> str:
    """Same rule the page uses: an explicit URL, else the port published by compose."""
    explicit = os.environ.get("APP_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    port = ""
    if ENV_EXAMPLE.parent.joinpath(".env").is_file():
        for line in (ENV_EXAMPLE.parent / ".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("APP_PORT=") or line.startswith("PORT="):
                port = line.split("=", 1)[1].strip()
    return f"http://localhost:{port or '8300'}"


def get(path: str, timeout: float = 8.0) -> Optional[Dict[str, Any]]:
    """GET one endpoint, or None when it cannot be reached."""
    try:
        with urllib.request.urlopen(base_url() + path, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
        report(f"GET {path}", False, str(error))
        return None


def get_text(path: str, timeout: float = 8.0) -> Optional[str]:
    """GET one endpoint as text, or None when it cannot be reached."""
    try:
        with urllib.request.urlopen(base_url() + path, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except (urllib.error.URLError, OSError, UnicodeDecodeError) as error:
        report(f"GET {path}", False, str(error))
        return None


def check_live() -> None:
    print(f"\n  target: {base_url()}")
    health = get("/api/health")
    if health is None:
        report("app is reachable", False, f"is it running?  make run PLAN=SaaS  (or pass APP_URL)")
        return
    # The endpoint answers `status: "ok"`. Asserting a field it does not send is exactly the
    # mistake the restart poll used to make, so the check names the field it reads.
    report("GET /api/health reports status ok", health.get("status") == "ok", json.dumps(health)[:160])

    served_page: Optional[str] = get_text("/")
    if served_page is not None:
        # The image bakes the source in, so a container left on an older build serves the old page
        # while every endpoint keeps answering correctly -- which is how a fix can be "verified"
        # against an API and still be missing from the screen. This is the only check that compares
        # what the browser is handed with what this tree says it should be.
        def squash(value: str) -> str:
            return re.sub(r"\s+", " ", value).strip()

        report("the running app serves the page from this tree",
               squash(served_page) == squash(read_text(PAGE)),
               f"served {len(served_page)} characters; rebuild with make build if this fails")

        for reference in sorted(set(re.findall(r"/static/([\w./+-]+)", served_page))):
            status, declared, body = fetch_bytes("/static/" + reference)
            report(f"the running app serves {reference}", status == 200 and len(body) > 0,
                   f"HTTP {status}, declared {declared or 'no type'}, {len(body)} bytes")
            on_disk: Path = TRACK_ROOT / "src" / "app" / "static" / reference
            # A container on an older build serves a page naming diagrams it does not contain, and
            # a browser shows a broken frame rather than saying so.
            report(f"the served {reference} is the file in this tree",
                   on_disk.is_file() and body == on_disk.read_bytes(),
                   f"served {len(body)} bytes, on disk "
                   f"{on_disk.stat().st_size if on_disk.is_file() else 'missing'}")
            suffix: str = on_disk.suffix.lower()
            declared_type: str = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                                  ".gif": "image/gif", ".svg": "image/svg+xml"}.get(suffix, "")
            report(f"{reference} is announced as the type its name promises",
                   declared.split(";")[0].strip() == declared_type,
                   f"declared {declared or 'no type'} for a file named {suffix}")

    state = get("/api/wizard")
    if state is not None:
        report("wizard serves six steps", len(state.get("steps", []) or []) == 6)
        engine_step = next((s for s in state["steps"] if s.get("number") == 4), {})
        report("step 4 is the Docker Engine screen",
               engine_step.get("title") == "Docker Engine"
               and engine_step.get("status") in {"locked", "available", "current", "completed", "skipped"})
        report("the engine flag is exposed to the GUI", "engine_verified" in state)
        report("wizard state can be restored from the graph store",
               state.get("persistence", {}).get("graph_verified") is True,
               json.dumps(state.get("persistence", {}))[:200])
        unlocked = int(state.get("unlocked_through", 0) or 0)
        current = int(state.get("current_step", 0) or 0)
        report("the open screen is never ahead of the unlocked frontier", current <= unlocked,
               f"current_step={current} unlocked_through={unlocked}")

    store_status = get("/api/neo4j/status")
    if store_status is not None:
        report("graph store probe reports its own database", bool(store_status.get("target")))
        report("the app's own store answers", store_status.get("reachable") is True,
               json.dumps(store_status.get("error"))[:160])

        # The platform's database is a second store, and the endpoint answers for it under every
        # plan -- present, and stating that it was not asked, rather than omitted. One shared
        # `reachable` flag could not express "server up, platform database not there", which is
        # the state this submission is actually in when both stacks are running.
        probe = store_status.get("platform_database")
        report("the platform's database has its own answer in the payload",
               isinstance(probe, dict), "the endpoint did not return platform_database")
        if isinstance(probe, dict):
            for required in ("requested", "configured", "reachable", "database", "databases",
                             "exists", "error", "hint"):
                report(f"platform_database reports {required}", required in probe)
            if probe.get("requested") is False:
                report("an unasked probe says so instead of staying silent", bool(probe.get("hint")))
            else:
                report("the probe was not answered with the app's own answer",
                       probe.get("uri") != store_status.get("target"),
                       "the two questions were asked of the same database")

    saas = get("/api/saas/model/config")
    if saas is not None and saas.get("ok"):
        report("a configured secret reads back as set, not as a value",
               saas.get("api_key_masked") == "***" and isinstance(saas.get("api_key_set"), bool),
               json.dumps(saas)[:200])


def check_the_engine_step() -> None:
    """Screen 5: a read-only probe on the platform's own two variables, renumbered honestly.

    The point of this step -- what makes it more than a curl alias -- is that it cannot
    mutate anything, cannot flatter the host with a second address, and cannot turn a
    waiver into a pass. Each of those is falsifiable here rather than in review.
    """
    engine: str = read_text(TRACK_ROOT / "src" / "app" / "docker_engine.py")
    wizard: str = read_text(TRACK_ROOT / "src" / "app" / "wizard.py")
    entry: str = read_text(TRACK_ROOT / "src" / "app" / "main.py")
    page: str = read_text(PAGE)
    overlay: str = read_text(TRACK_ROOT / "docker-compose.plan-local.yml")
    base: str = read_text(TRACK_ROOT / "docker-compose.yml")
    readme: str = read_text(TRACK_ROOT / "README.md")

    report("the Engine step is step 4 and skippable by plan",
           re.search(r'StepDefinition\(4, "docker-engine".*saas_skippable=True', wizard, re.S) is not None)
    report("a renumbered flow migrates instead of guessing",
           "FLOW_VERSION" in wizard and 'result["migration"] = migration' in wizard
           and "migration_note" in entry)
    report("the Engine flag reaches the GUI", '"engine_verified": _STATE.engine_verified' in wizard)
    for kind in ("post", "put", "delete", "patch"):
        report(f"the probe issues no {kind.upper()}s", re.search(rf"\.{kind}\(", engine) is None)
    report("the probe issues exactly the two documented GETs",
           engine.count("client.get(") == 2 and "/_ping" in engine and "/version" in engine)
    report("timeout is named as a firewall question, not swallowed",
           "firewall" in engine and "bridge subnet" in engine.lower())
    for route in ("@app.get(\"/api/docker/status\")", "@app.post(\"/api/wizard/docker-test\")",
                  "@app.post(\"/api/wizard/docker/waive\")"):
        report(f"{route.split(chr(34))[1]} is registered", route in entry)
    report("a waiver is persisted through the real writer, not a stub",
           re.search(r"waive_engine_step[\s\S]{0,200}?save_state", entry) is not None)
    report("the overlay pins the platform's own address for the app too",
           re.search(r"app:\s*\n\s*environment:[\s\S]*?DOCKER_ENGINE_ADDR: host\.docker\.internal"
                     r"[\s\S]*?DOCKER_ENGINE_PORT: \"2375\"", overlay) is not None)
    report("and the platform service of the same overlay -- one source of truth",
           overlay.count("DOCKER_ENGINE_ADDR") >= 2)
    report("the base compose passes the same names through from .env",
           "${DOCKER_ENGINE_ADDR:-host.docker.internal}" in base and "${DOCKER_ENGINE_PORT:-2375}" in base)
    report("the app resolves the gateway name the same way the platform does",
           re.search(r'app:[\s\S]*?extra_hosts:[\s\S]*?host\.docker\.internal:host-gateway', base) is not None)
    report("the README names the same pair the code reads", "`DOCKER_ENGINE_ADDR`" in readme
           and "docker-compose.plan-local.yml" in readme)
    report("the waiver reason cannot exceed what is stored",
           page.find('maxlength="200"') > 0)
    report("navigation was renumbered around the new screen",
           re.search(r'n <= 6', page) is not None
           and 'completeStep(4)' in page and 'completeStep(5)' in page
           and 'gotoStep(5)' in page and 'step4Waive' in page)
    report("the screen states the firewall honestly on the machine that has one",
           "firewall" in page.lower() or "UFW" in page)


def main(argv: List[str]) -> int:
    print("verify_submission: documentation versus code")
    print("\n  routes")
    check_routes()
    print("\n  configuration contract")
    check_env_names()
    print("\n  superseded claims")
    check_superseded_claims()
    print("\n  the page can run its own script")
    check_page_script_parses()
    print("\n  welcome screen")
    check_welcome_screen_marks_its_plan()
    print("\n  slide captions and caveats")
    check_slides_carry_their_command_and_their_caveat()
    print("\n  deployment diagrams")
    check_diagrams_are_shipped()
    print("\n  the wide column")
    check_the_wide_column_stays_readable()
    print("\n  evidence cannot cross plans")
    check_evidence_does_not_cross_plans()
    print("\n  the engine step")
    check_the_engine_step()
    print("\n  recently changed behaviour")
    check_restart_poll()
    check_no_dead_markup()
    check_second_store_cannot_be_conflated()
    if "--live" in argv:
        print("\n  live endpoints")
        check_live()

    print(f"\n{checks} checks, {len(failures)} failure(s)")
    for failure in failures:
        print(f"  - {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
