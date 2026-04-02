#!/usr/bin/env bash
# download silesia corpus, enwik8, enwik9, calgary corpus into the datasets directory
# check before downloading if the files already exist and the md5sum matches
# after downloads, extract the files
# always try mattmahoney.net first

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="${SCRIPT_DIR}/datasets"

mkdir -p "${DATASETS_DIR}"

# ── helpers ──────────────────────────────────────────────────────────────────

log()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
err()  { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; }

check_deps() {
    local missing=()
    for cmd in wget md5sum tar unzip; do
        command -v "$cmd" &>/dev/null || missing+=("$cmd")
    done
    if (( ${#missing[@]} )); then
        err "Missing required tools: ${missing[*]}"
        exit 1
    fi
}

# download_file URL DEST
#   Downloads URL to DEST, returns 0 on success.
download_file() {
    local url="$1" dest="$2"
    log "Downloading ${url} -> ${dest}"
    wget --tries=3 --waitretry=5 -q --show-progress -O "${dest}" "${url}"
}

# compute_md5 FILE
#   Prints the md5 of a file.
compute_md5() {
    md5sum "$1" | awk '{print $1}'
}

# verify_md5 FILE MD5_FILE
#   Returns 0 if the file's md5 matches the cached value.
verify_md5() {
    local file="$1" md5_file="$2"
    [[ -f "${md5_file}" ]] || return 1
    local expected actual
    expected="$(cat "${md5_file}")"
    actual="$(compute_md5 "${file}")"
    [[ "${actual}" == "${expected}" ]]
}

# save_md5 FILE
#   Computes and caches the md5 of a file alongside it (file.md5).
save_md5() {
    local file="$1"
    compute_md5 "${file}" > "${file}.md5"
    log "Saved checksum for $(basename "${file}")"
}

# fetch FILE_PATH URL [FALLBACK_URL...]
#   Skips download when a valid file already exists (verified via cached .md5).
#   On first download, computes and saves the md5 for future runs.
#   Tries each URL in order until one succeeds.
fetch() {
    local dest="$1"
    shift
    local urls=("$@")
    local md5_file="${dest}.md5"

    if [[ -f "${dest}" ]]; then
        if verify_md5 "${dest}" "${md5_file}"; then
            log "Already have $(basename "${dest}") (md5 OK) – skipping"
            return 0
        elif [[ -f "${md5_file}" ]]; then
            warn "$(basename "${dest}") exists but md5 mismatch – re-downloading"
        else
            log "$(basename "${dest}") exists but no checksum cached – computing"
            save_md5 "${dest}"
            return 0
        fi
    fi

    for url in "${urls[@]}"; do
        if download_file "${url}" "${dest}"; then
            save_md5 "${dest}"
            log "Downloaded $(basename "${dest}") ✓"
            return 0
        else
            warn "Failed to download from ${url}"
            rm -f "${dest}"
        fi
    done

    err "All sources failed for $(basename "${dest}")"
    return 1
}

# ── dataset definitions ──────────────────────────────────────────────────────

# enwik8  –  first 10^8 bytes of English Wikipedia (zipped)
ENWIK8_URLS=(
    "https://mattmahoney.net/dc/enwik8.zip"
    "https://data.deepai.org/enwik8.zip"
)

# enwik9  –  first 10^9 bytes of English Wikipedia (zipped)
ENWIK9_URLS=(
    "https://mattmahoney.net/dc/enwik9.zip"
    "https://data.deepai.org/enwik9.zip"
)

# silesia corpus  –  standard compression benchmark
SILESIA_URLS=(
    "https://mattmahoney.net/dc/silesia.zip"
    "https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip"
)

# calgary corpus  –  classic compression test suite
CALGARY_URLS=(
    "https://mattmahoney.net/dc/calgary.zip"
    "https://corpus.canterbury.ac.nz/resources/calgary.zip"
)

# ── download ─────────────────────────────────────────────────────────────────

check_deps

FAILED=0

fetch "${DATASETS_DIR}/enwik8.zip"  "${ENWIK8_URLS[@]}"  || ((FAILED++))
fetch "${DATASETS_DIR}/enwik9.zip"  "${ENWIK9_URLS[@]}"  || ((FAILED++))
fetch "${DATASETS_DIR}/silesia.zip" "${SILESIA_URLS[@]}" || ((FAILED++))
fetch "${DATASETS_DIR}/calgary.zip" "${CALGARY_URLS[@]}" || ((FAILED++))

# ── extract ──────────────────────────────────────────────────────────────────

extract_zip() {
    local archive="$1" dest_dir="$2"
    if [[ ! -f "${archive}" ]]; then
        warn "$(basename "${archive}") not found – skipping extraction"
        return
    fi
    if [[ -d "${dest_dir}" && "$(ls -A "${dest_dir}" 2>/dev/null)" ]]; then
        log "$(basename "${dest_dir}") already extracted – skipping"
        return
    fi
    log "Extracting $(basename "${archive}") -> ${dest_dir}"
    mkdir -p "${dest_dir}"
    unzip -qo "${archive}" -d "${dest_dir}"
}

extract_zip "${DATASETS_DIR}/enwik8.zip"  "${DATASETS_DIR}/enwik8"
extract_zip "${DATASETS_DIR}/enwik9.zip"  "${DATASETS_DIR}/enwik9"
extract_zip "${DATASETS_DIR}/silesia.zip" "${DATASETS_DIR}/silesia"
extract_zip "${DATASETS_DIR}/calgary.zip" "${DATASETS_DIR}/calgary"

# ── summary ──────────────────────────────────────────────────────────────────

echo
if (( FAILED == 0 )); then
    log "All datasets downloaded and extracted successfully!"
    log "Location: ${DATASETS_DIR}"
    du -sh "${DATASETS_DIR}"/*/ 2>/dev/null || true
else
    err "${FAILED} dataset(s) failed to download. See warnings above."
    exit 1
fi
