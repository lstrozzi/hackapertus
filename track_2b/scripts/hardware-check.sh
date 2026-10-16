#!/bin/sh
# Fully local hardware gate — Hack Apertus 2026, Track 2B.
#
# Why a hard gate: the vLLM model images publish a linux/arm64 manifest only, and
# the platform was built, tuned and tested on one machine class (NVIDIA DGX Spark,
# ARM64 GB10 CPU+GPU, 128 GB unified memory). On any other machine the stack does
# not fail cleanly — it fails late or half-starts, which is worse for a judge than
# an instant ERROR that says what is required and what the fallback is.
#
# Modes:
#   sh scripts/hardware-check.sh                gate: exit 1 when requirements unmet
#   sh scripts/hardware-check.sh --banner       print the tested-platform banner only
#   THALABUS_HW_CHECK=warn make run             same report, but WARNING + continue
#   THALABUS_HW_CHECK=0    make run             skip the gate (organisers known-good CI)

banner() {
    echo "The provided images have been tested on NVIDIA DGX Spark machines with ARM64"
    echo "GB10 CPU+GPU with 128GB RAM. We cannot guarantee that they work on any other"
    echo "machine or architecture and we have provided a SaaS running instance on said"
    echo "NVIDIA machine."
    echo ""
    echo "Adapting local deployment to other hardware architectures may take 1–5 days. Given the hackathon"
    echo "timeframe and the lack of representative alternative hardware for testing, that"
    echo "work is outside scope."
}

if [ "$1" = "--banner" ]; then
    banner
    exit 0
fi

env_file_has_value() {
    [ -f .env ] || return 1
    awk -F= -v key="$1" '
        {
            name = $1
            gsub(/^[ \t]+|[ \t]+$/, "", name)
            if (name != key) next
            value = substr($0, index($0, "=") + 1)
            gsub(/^[ \t]+|[ \t]+$/, "", value)
            if (value != "" && value != "\"\"" && value != "\047\047" && value != "replace-me" && value !~ /^#/) configured = 1
        }
        END { exit !configured }
    ' .env
}

llm_value_is_set() {
    case "$1" in
        LLM_NAME)
            value="${LLM_NAME:-}"
            if [ -z "$value" ] && { [ "${THALABUS_PLAN:-}" = "saas" ] || [ "${THALABUS_PLAN:-}" = "b" ]; }; then value="apertus-70b"; fi
            ;;
        LLM_BASE_URL)
            value="${LLM_BASE_URL:-}"
            if [ -z "$value" ] && { [ "${THALABUS_PLAN:-}" = "saas" ] || [ "${THALABUS_PLAN:-}" = "b" ]; }; then value="https://llm.thalabus.ai"; fi
            ;;
        LLM_API_KEY) value="${LLM_API_KEY:-}" ;;
        *) return 1 ;;
    esac
    if [ -n "$value" ] && [ "$value" != "replace-me" ]; then
        return 0
    fi
    env_file_has_value "$1"
}

missing_llm_values=""
for llm_name in LLM_NAME LLM_BASE_URL LLM_API_KEY; do
    if ! llm_value_is_set "$llm_name"; then
        if [ -n "$missing_llm_values" ]; then
            missing_llm_values="$missing_llm_values, "
        fi
        missing_llm_values="$missing_llm_values$llm_name"
    fi
done
if [ -n "$missing_llm_values" ]; then
    if [ "${THALABUS_PLAN:-}" = "saas" ] || [ "${THALABUS_PLAN:-}" = "b" ]; then
        printf '\033[33mWARNING: Missing %s. Add the SaaS API key to .env; model and endpoint default to Apertus 70B at LLM Lite.\033[0m\n\n' "$missing_llm_values"
    else
        printf '\033[33mWARNING: Missing %s. Add the SaaS-provided values to .env.\033[0m\n\n' "$missing_llm_values"
    fi
else
    printf '\033[33mUsing configured SaaS-provided values for LLM_NAME, LLM_BASE_URL, and LLM_API_KEY (values hidden).\033[0m\n'
fi

mode="${THALABUS_HW_CHECK:-}"
if [ "$mode" = "0" ]; then
    echo "NOTE: Fully local hardware check skipped (THALABUS_HW_CHECK=0)."
    exit 0
fi

os=$(uname -s 2>/dev/null || echo unknown)
machine=$(uname -m 2>/dev/null || echo unknown)

problems=""
add_problem() {
    problems="${problems}  - $1
"
}

# --- architecture: the model images exist for linux/arm64 only -------------------
case "$machine" in
    aarch64|arm64) : ;;
    *) add_problem "architecture '$machine' is not ARM64 — the model images publish linux/arm64 manifests only (tested on NVIDIA DGX Spark ARM64 GB10 architecture with 128GB unified RAM)" ;;
esac
if [ "$os" != "Linux" ]; then
    add_problem "OS '$os' is not Linux — the stack is tested on Linux/ARM64 only"
fi

# --- system RAM: >=128 GB, measured as MemTotal >= 115 GiB (kernel/carve-out) ----
ram_info="unknown"
if [ -r /proc/meminfo ]; then
    mem_kb=$(awk '/^MemTotal/ {print $2; exit}' /proc/meminfo)
    if [ -n "$mem_kb" ]; then
        ram_gib=$((mem_kb / 1048576))
        ram_info="${ram_gib} GiB"
        if [ "$ram_gib" -lt 115 ]; then
            add_problem "system RAM is ${ram_gib} GiB (gate: 115 GiB MemTotal, i.e. a 128 GB machine after kernel carve-out)"
        fi
    else
        add_problem "cannot read MemTotal from /proc/meminfo"
    fi
else
    add_problem "/proc/meminfo not readable — hardware gate needs a Linux host"
fi

# --- GPU: NVIDIA, driver visible, memory >=96 GB (gate: 92000 MiB total) ---------
gpu_info="none visible"
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk -F, '{ gsub(/ /, "", $1); s += $1 } END { print s + 0 }')
    gpu_mib=${gpu_mib:-0}
    gpu_info="${gpu_mib} MiB NVIDIA (nvidia-smi)"
    if [ "$gpu_mib" -lt 92000 ]; then
        add_problem "NVIDIA GPU memory totals ${gpu_mib} MiB (gate: 92000 MiB, i.e. one 96 GB+ GPU after driver reserve)"
    fi
else
    add_problem "no NVIDIA driver found (nvidia-smi missing) — the fully local option requires an NVIDIA GPU with >=96 GB memory"
fi

if [ -n "$problems" ]; then
    if [ "$mode" = "warn" ]; then
        echo "WARNING: Fully local hardware requirements are not met on this host."
    else
        printf '\033[31mERROR: Fully local hardware requirements are NOT met on this host:\033[0m\n'
    fi
    printf '%s' "$problems"
    echo ""
    printf '\033[33m'
    echo "Adapting local deployment to other hardware architectures may take 1–5 days. Given the hackathon"
    echo "timeframe and the lack of representative alternative hardware for testing, that"
    echo "work is outside scope."
    echo ""
    printf '\033[0m'
    printf "\033[32mSolution: Local + SaaS — run the app against the provided SaaS instance in the tested T2B data center:\n    \033[1mmake run PLAN=SaaS\033[22m\n\nThe thalabus platform still can be deployed fully locally, based on hardware requirements, for full data privacy and sovereignty. We are running fully local deployments (incl. platform and LLMs) by our clients' data centers.\033[0m\n"
    echo ""
    echo "Additional information:"
    echo "This host:    OS=$os  arch=$machine  RAM=$ram_info  GPU=$gpu_info"
    echo "Requirements (tested configuration):"
    echo "  - Linux on ARM64 (aarch64); the vLLM model images publish linux/arm64 only"
    echo "  - >=128 GB system RAM; >=96 GB NVIDIA GPU memory"
    echo "  - Tested: NVIDIA DGX Spark ARM64 GB10 architecture with 128GB unified RAM"
    echo "  - Docker with Compose v2+ and GNU make"
    echo ""
    echo "Escape hatches (organisers / known-good hosts only):"
    echo "    THALABUS_HW_CHECK=warn make run    # keep the check, downgrade to WARNING"
    echo "    THALABUS_HW_CHECK=0 make run       # skip the check"
    if [ "$mode" = "warn" ]; then
        exit 0
    else
        exit 1
    fi
fi

echo "hardware-check: OK ($os/$machine, $ram_info RAM, $gpu_info)."
exit 0
