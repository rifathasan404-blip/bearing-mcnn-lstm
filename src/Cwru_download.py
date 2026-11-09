#!/usr/bin/env python3
"""Download the CWRU Bearing Data Center dataset and label every file.

The file list is read live from the four official data pages, so the labels
come from the same tables you would cite. Each file is saved as
<out>/<category>/<label>__<file number>.mat, and manifest.csv records its
fault type, fault size, motor load, speed, sampling rate, URL and SHA-256.

Safe to re-run: finished files are skipped and partial files are resumed.

    python3 download_cwru.py --dry-run             # list files and total size
    python3 download_cwru.py                       # everything (161 files)
    python3 download_cwru.py --only normal 12k_de  # just these categories
    python3 download_cwru.py --verify              # also open files with scipy
"""
import argparse
import csv
import hashlib
import html
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SITE = "https://engineering.case.edu/bearingdatacenter/"
ALLOWED_HOST = "engineering.case.edu"
USER_AGENT = "cwru-dataset-downloader/1.0"
CHUNK = 1024 * 256
MAX_TRIES = 8

# key: (page slug, folder name, sampling rate in Hz, fault end)
# The normal rate is blank on purpose: the CWRU site does not state it and
# published sources disagree, so confirm it from the data before relying on it.
CATEGORIES = {
    "normal": ("normal-baseline-data", "normal", "", "none"),
    "12k_de": ("12k-drive-end-bearing-fault-data", "12k_drive_end", 12000, "drive_end"),
    "48k_de": ("48k-drive-end-bearing-fault-data", "48k_drive_end", 48000, "drive_end"),
    "12k_fe": ("12k-fan-end-bearing-fault-data", "12k_fan_end", 12000, "fan_end"),
}

FIELDS = [
    "category", "label", "file_number", "path", "sampling_rate_hz", "fault_end",
    "fault_type", "fault_diameter_in", "outer_race_position", "motor_load_hp",
    "approx_rpm", "url", "bytes", "sha256",
]

LABEL_RE = re.compile(r"^(IR|B|OR)(\d{3})(?:@(\d+))?_(\d)$")
NORMAL_RE = re.compile(r"^Normal_(\d)$")
TYPE_NAMES = {"IR": "inner_race", "B": "ball", "OR": "outer_race"}


def strip_tags(fragment):
    text = re.sub(r"(?s)<[^>]+>", " ", fragment)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def request(url, method="GET", headers=None):
    return urllib.request.Request(
        url, method=method, headers={"User-Agent": USER_AGENT, **(headers or {})}
    )


def backoff(attempt):
    time.sleep(min(60, 2 ** attempt) + random.random())


def fetch_text(url):
    for attempt in range(MAX_TRIES):
        try:
            with urllib.request.urlopen(request(url), timeout=60) as resp:
                return resp.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError) as err:
            if attempt == MAX_TRIES - 1:
                raise RuntimeError(f"could not read {url}: {err}") from err
            backoff(attempt)


def parse_label(label):
    """Return fault fields from a table label such as IR007_0 or OR014@6_1."""
    match = NORMAL_RE.match(label)
    if match:
        return {"fault_type": "normal", "fault_diameter_in": "",
                "outer_race_position": "", "motor_load_hp": match.group(1)}
    match = LABEL_RE.match(label)
    if match:
        kind, size, position, load = match.groups()
        return {"fault_type": TYPE_NAMES[kind],
                "fault_diameter_in": f"0.{size}",
                "outer_race_position": position or "",
                "motor_load_hp": load}
    return None


def parse_page(key):
    slug, folder, rate, fault_end = CATEGORIES[key]
    page = fetch_text(SITE + slug)
    tables = re.findall(r"(?is)<table.*?</table>", page)
    if not tables:
        raise RuntimeError(f"no data table on {SITE + slug}; the page layout may have changed")

    records = []
    for row in re.findall(r"(?is)<tr.*?</tr>", tables[0]):
        cells = [strip_tags(c) for c in re.findall(r"(?is)<t[dh][^>]*>(.*?)</t[dh]>", row)]
        # The motor speed is the only 4-digit whole-number cell in a data row.
        rpm = next((c for c in cells if re.fullmatch(r"\d{4}", c)), "")
        for href, inner in re.findall(r'(?is)<a[^>]+href="([^"]+\.mat)"[^>]*>(.*?)</a>', row):
            url = urllib.parse.urljoin(SITE, html.unescape(href))
            if urllib.parse.urlparse(url).hostname != ALLOWED_HOST:
                raise RuntimeError(f"unexpected download host: {url}")
            number = Path(urllib.parse.urlparse(url).path).stem
            label = re.sub(r"[^A-Za-z0-9@_.-]", "", strip_tags(inner))
            fields = parse_label(label)
            if fields is None:
                print(f"  note: unrecognised label {label!r} for {url}", file=sys.stderr)
                fields = {"fault_type": "unknown", "fault_diameter_in": "",
                          "outer_race_position": "", "motor_load_hp": ""}
            records.append({
                "category": folder, "label": label, "file_number": number,
                "path": f"{folder}/{label}__{number}.mat",
                "sampling_rate_hz": rate, "fault_end": fault_end,
                "approx_rpm": rpm, "url": url, "bytes": "", "sha256": "", **fields,
            })
    return records


def remote_size(url):
    for attempt in range(MAX_TRIES):
        try:
            with urllib.request.urlopen(request(url, "HEAD"), timeout=60) as resp:
                return int(resp.headers["Content-Length"])
        except (urllib.error.URLError, TimeoutError, TypeError) as err:
            if attempt == MAX_TRIES - 1:
                raise RuntimeError(f"no size for {url}: {err}") from err
            backoff(attempt)


def looks_like_html(path):
    with open(path, "rb") as handle:
        return handle.read(64).lstrip().lower().startswith((b"<!doctype", b"<html"))


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, dest, expected):
    """Fetch url into dest, resuming from dest.part. Returns True if bytes moved."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size == expected:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    for attempt in range(MAX_TRIES):
        have = part.stat().st_size if part.exists() else 0
        if have > expected:
            part.unlink()
            have = 0
        try:
            if have < expected:
                headers = {"Range": f"bytes={have}-"} if have else {}
                with urllib.request.urlopen(request(url, headers=headers), timeout=120) as resp:
                    # A plain 200 means the server ignored Range, so start over.
                    mode = "ab" if have and resp.status == 206 else "wb"
                    with open(part, mode) as out:
                        for block in iter(lambda: resp.read(CHUNK), b""):
                            out.write(block)
            size = part.stat().st_size
            if size == expected and not looks_like_html(part):
                part.replace(dest)
                return True
            # Keep a short file so the next attempt resumes it; discard bad content.
            if size >= expected:
                part.unlink()
            raise OSError(f"bad download: got {size} bytes, expected {expected}")
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            if attempt == MAX_TRIES - 1:
                raise RuntimeError(f"{url}: {err}") from err
            backoff(attempt)
    return False


def verify_mat(path):
    """Return an error string if scipy cannot read the file or finds no signal."""
    import numpy as np
    from scipy.io import loadmat

    try:
        data = loadmat(path)
    except Exception as err:  # scipy raises many types on corrupt files
        return f"unreadable: {err}"
    signals = [k for k in data if k.endswith(("_DE_time", "_FE_time", "_BA_time"))]
    if not signals:
        return "no _DE/_FE/_BA signal variables"
    for name in signals:
        if data[name].size == 0 or not np.isfinite(data[name]).all():
            return f"bad values in {name}"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="cwru_data", help="output folder (default: cwru_data)")
    parser.add_argument("--only", nargs="+", choices=list(CATEGORIES), help="categories to fetch")
    parser.add_argument("--workers", type=int, default=2, help="parallel downloads (default: 2)")
    parser.add_argument("--dry-run", action="store_true", help="list files and size, download nothing")
    parser.add_argument("--verify", action="store_true", help="open every file with scipy afterwards")
    args = parser.parse_args()

    out = Path(args.out)
    records = []
    for key in args.only or CATEGORIES:
        rows = parse_page(key)
        print(f"{key:7s} {len(rows):3d} files  ({SITE}{CATEGORIES[key][0]})")
        records.extend(rows)

    print("Reading file sizes (the server is slow, so this can take several minutes)...")
    with ThreadPoolExecutor(args.workers) as pool:
        sizes = list(pool.map(remote_size, [r["url"] for r in records]))
    for record, size in zip(records, sizes):
        record["bytes"] = size
    print(f"Total: {len(records)} files, {sum(sizes) / 1e9:.2f} GB")
    if args.dry_run:
        return 0

    failures = []
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(download, r["url"], out / r["path"], r["bytes"]): r for r in records}
        for count, job in enumerate(as_completed(jobs), 1):
            record = jobs[job]
            try:
                state = "downloaded" if job.result() else "already present"
            except RuntimeError as err:
                failures.append(record["path"])
                state = f"FAILED: {err}"
            print(f"[{count}/{len(records)}] {record['path']}: {state}", flush=True)

    for record in records:
        path = out / record["path"]
        if path.exists():
            record["sha256"] = sha256_of(path)
    with open(out / "manifest.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(records)

    if args.verify:
        print("Checking files with scipy...")
        for record in records:
            path = out / record["path"]
            if path.exists():
                problem = verify_mat(path)
                if problem:
                    failures.append(record["path"])
                    print(f"  {record['path']}: {problem}")

    print(f"Done: {len(records) - len(failures)}/{len(records)} OK. Manifest: {out / 'manifest.csv'}")
    if failures:
        print("Problem files (re-run to retry downloads):", *failures, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
