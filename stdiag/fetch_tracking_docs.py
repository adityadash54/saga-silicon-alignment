#!/usr/bin/env python3
"""Build the local ACTS and Millepede-II snapshot read by tracking_expert.py.

The tracking expert has no network access: it reads only this snapshot,
the coresoftware checkout and the literature folder. The snapshot is kept
in tracking_docs/ and refreshed one part at a time, only when asked:

  acts/docs/...            Markdown/Doxygen sources of the ACTS docs
  acts/Core/...            ACTS core headers and sources (ground truth for
                           conventions)
  acts/Plugins/{Root,Json}/..., acts/Examples/{Framework,Detectors}/...
                           headers and sources of the ActsPlugins and
                           ActsExamples parts that sPHENIX code includes
  acts/version_number      ACTS version of the snapshot
  millepede/draftman_page.html, draftman_page.txt
                           Millepede-II draft manual; the .txt keeps the
                           page's anchors as {#anchor} markers for citing
  SOURCES.json             where each part came from (ref, commit, file
                           checksum, time)

ACTS comes from the sPHENIX fork (the version sPHENIX builds against), not
upstream main. It is downloaded only with --update-acts, into a temporary
folder that replaces the old copy only when the download succeeded.

The Millepede-II manual is built from a copy of the page saved by hand
(default docs/Millepede/draftman_page.html, STDIAG_MILLEPEDE_HTML), e.g.
  curl -L --proto-redir =https --max-time 300 -A "Mozilla/5.0" \\
    -o docs/Millepede/draftman_page.html <MILLEPEDE_URL>
or the browser's "Save Page As, HTML only". Only text and anchors are
kept; script and style content is dropped.

Usage:
  python -m stdiag.fetch_tracking_docs --update-acts       # first time
  python -m stdiag.fetch_tracking_docs                     # Millepede only
  python -m stdiag.fetch_tracking_docs --update-acts --acts-ref <full hash>
  python -m stdiag.fetch_tracking_docs --check             # status, offline
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html.parser
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from stdiag.paths import MILLEPEDE_HTML, TRACKING_DOCS

ACTS_REPO = "https://github.com/sPHENIX-Collaboration/acts.git"
# Parts of ACTS kept; sPHENIX code includes Acts/*, ActsExamples/* and
# ActsPlugins/{Root,Json}/*. Folders missing at a given ref are skipped.
ACTS_PATHS = ["docs", "Core", "Plugins/Root", "Plugins/Json",
              "Examples/Framework", "Examples/Detectors"]
ACTS_SUFFIXES = {".md", ".dox", ".bib", ".hpp", ".ipp", ".h", ".cpp"}
MILLEPEDE_URL = "https://www.desy.de/~kleinwrt/MP2/doc/html/draftman_page.html"
# The docs map cites this anchor; a page without it is not the manual.
MILLEPEDE_MARKER = "ssec-meapar"


class TextWithAnchors(html.parser.HTMLParser):
    """HTML to plain text, marking id/name anchors as {#anchor}."""

    BREAK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4",
                  "h5", "h6", "pre", "table"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        anchor = dict(attrs).get("id") or dict(attrs).get("name")
        if anchor:
            self.parts.append(f" {{#{anchor}}} ")
        if tag in self.BREAK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip:
            self.skip -= 1
        if tag in self.BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self.parts)
                 .splitlines())
        return "\n".join(line for line in lines if line) + "\n"


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def run(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clone_acts(ref: str, clone: Path) -> None:
    """Shallow sparse checkout (cone mode) of a branch, tag or full hash."""
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        run(["git", "init", "-q", str(clone)])
        run(["git", "remote", "add", "origin", ACTS_REPO], cwd=clone)
        run(["git", "fetch", "-q", "--depth", "1", "origin", ref], cwd=clone)
        run(["git", "sparse-checkout", "set", "--cone", *ACTS_PATHS],
            cwd=clone)
        run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=clone)
    else:
        run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none",
             "--sparse", "--branch", ref, ACTS_REPO, str(clone)])
        run(["git", "sparse-checkout", "set", "--cone", *ACTS_PATHS],
            cwd=clone)


def copy_acts(clone: Path, destination: Path) -> dict[str, int]:
    """Copy the kept files; return the number of files per ACTS part."""
    counts = {}
    for part in ACTS_PATHS:
        counts[part] = 0
        for path in (clone / part).rglob("*"):
            if path.is_file() and path.suffix in ACTS_SUFFIXES:
                target = destination / path.relative_to(clone)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                counts[part] += 1
    shutil.copy2(clone / "version_number", destination / "version_number")
    return counts


def fetch_acts(ref: str, destination: Path) -> dict:
    """Download ACTS at ref; replace destination only if every part has
    files (a part with none means the folder layout is not as expected)."""
    staging = destination.with_name(destination.name + ".new")
    if staging.exists():
        shutil.rmtree(staging)
    try:
        with tempfile.TemporaryDirectory() as temporary:
            clone = Path(temporary) / "acts"
            clone_acts(ref, clone)
            commit = run(["git", "rev-parse", "HEAD"], cwd=clone)
            date = run(["git", "log", "-1", "--format=%cI"], cwd=clone)
            version = (clone / "version_number").read_text().strip()
            staging.mkdir(parents=True)
            counts = copy_acts(clone, staging)
        empty = [part for part, count in counts.items() if not count]
        if empty:
            raise RuntimeError(
                f"no files in {', '.join(empty)} at ACTS {ref} ({commit[:12]}"
                "); the folder layout differs from ACTS_PATHS in "
                "fetch_tracking_docs.py")
    except subprocess.CalledProcessError as error:
        shutil.rmtree(staging, ignore_errors=True)
        sys.exit(f"ERROR: ACTS download failed ({' '.join(error.cmd[:3])}"
                 f"): {(error.stderr or '').strip()[-500:]}\n"
                 "The existing ACTS copy is unchanged.")
    except (OSError, RuntimeError) as error:
        shutil.rmtree(staging, ignore_errors=True)
        sys.exit(f"ERROR: {error}\nThe existing ACTS copy is unchanged.")
    if destination.exists():
        shutil.rmtree(destination)
    staging.rename(destination)
    return {"repository": ACTS_REPO, "ref": ref, "commit": commit,
            "commit_date": date, "version_number": version,
            "files": counts, "fetched_utc": now_utc()}


def check_millepede_html(source: Path) -> str:
    """The page's text, or exit with how to get it."""
    if not source.is_file():
        sys.exit(f"ERROR: no Millepede-II page at {source}. Save "
                 f"{MILLEPEDE_URL} there (browser: Save Page As, HTML only; "
                 "or curl, see --help), or pass --millepede-html.")
    page = source.read_bytes().decode("utf-8", errors="replace")
    if MILLEPEDE_MARKER not in page:
        sys.exit(f"ERROR: {source} has no '{MILLEPEDE_MARKER}' anchor; it "
                 "is not the Millepede-II draft manual page.")
    return page


def build_millepede(source: Path, page: str, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "draftman_page.html").write_text(page)
    parser = TextWithAnchors()
    parser.feed(page)
    (destination / "draftman_page.txt").write_text(parser.text())
    modified = dt.datetime.fromtimestamp(source.stat().st_mtime,
                                         dt.timezone.utc)
    return {"url": MILLEPEDE_URL, "local_file": source.name,
            "sha256": sha256_of(source),
            "file_modified_utc": modified.isoformat(timespec="seconds"),
            "built_utc": now_utc()}


def read_sources(out: Path) -> dict:
    path = out / "SOURCES.json"
    return json.loads(path.read_text()) if path.exists() else {}


def status(out: Path) -> list[str]:
    """Problems with the snapshot (empty when complete)."""
    sources = read_sources(out)
    problems = []
    if not (out / "acts" / "version_number").exists() or "acts" not in sources:
        problems.append("ACTS missing: run with --update-acts")
    else:
        empty = [part for part in ACTS_PATHS if not any(
            path.is_file() for path in (out / "acts" / part).rglob("*"))]
        if empty:
            problems.append(f"ACTS parts without files: {', '.join(empty)}; "
                            "run with --update-acts")
    if (not (out / "millepede" / "draftman_page.txt").exists()
            or "millepede" not in sources):
        problems.append("Millepede-II manual missing: save the page and run "
                        "without options")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=TRACKING_DOCS)
    parser.add_argument("--update-acts", action="store_true",
                        help="download ACTS (network: github.com)")
    parser.add_argument("--acts-ref", default="sPHENIX",
                        help="branch, tag or full commit hash of the fork")
    parser.add_argument("--millepede-html", type=Path, default=MILLEPEDE_HTML,
                        help="saved copy of the Millepede-II manual page")
    parser.add_argument("--check", action="store_true",
                        help="report the snapshot status; change nothing")
    args = parser.parse_args()

    if args.check:
        print(json.dumps(read_sources(args.out), indent=2))
        problems = status(args.out)
        print("\n".join(problems) or "snapshot complete")
        sys.exit(1 if problems else 0)

    # Check the input before anything in the snapshot is replaced.
    page = check_millepede_html(args.millepede_html)
    args.out.mkdir(parents=True, exist_ok=True)
    sources = read_sources(args.out)
    if args.update_acts:
        sources["acts"] = fetch_acts(args.acts_ref, args.out / "acts")
    sources["millepede"] = build_millepede(args.millepede_html, page,
                                           args.out / "millepede")
    sources.pop("fetched_utc", None)  # old single-fetch format
    (args.out / "SOURCES.json").write_text(json.dumps(sources, indent=2)
                                           + "\n")
    print(json.dumps(sources, indent=2))
    problems = status(args.out)
    if problems:
        sys.exit("\n".join(problems))


if __name__ == "__main__":
    main()
