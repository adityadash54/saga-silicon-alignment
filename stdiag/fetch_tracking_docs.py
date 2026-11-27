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
  millepede/draftman_page.txt
                           Millepede-II draft manual (doc/draftManual.md)
                           with its Doxygen section labels written as
                           {#label} markers for citing
  millepede/doc/*.md       the other Millepede-II documentation pages
                           (steering options, major changes, exit codes,
                           hard-coded limits, ...)
  millepede/src/*.f90, *.c the pede source code
  SOURCES.json             where each part came from (repository, ref,
                           commit, checksums, time)

ACTS comes from the sPHENIX fork (the version sPHENIX builds against), not
upstream main. It is downloaded only with --update-acts, into a temporary
folder that replaces the old copy only when the download succeeded.

Millepede-II comes from a local clone of the Millepede-II repository
(default docs/Millepede/millepede-ii, STDIAG_MILLEPEDE_REPO), read at a
release tag (default MILLEPEDE_REF) with `git show <ref>:<path>`, so the
clone's checked-out commit and any local edits do not matter. No network
access is needed. Get the clone once and check its tag against DESY:
  git clone https://github.com/millepede-project/millepede-ii.git \\
    docs/Millepede/millepede-ii
  git -C docs/Millepede/millepede-ii rev-parse <tag>
  git ls-remote https://gitlab.desy.de/millepede/millepede-ii.git \\
    refs/tags/<tag>                       # must print the same hash

Usage:
  python -m stdiag.fetch_tracking_docs --update-acts       # first time
  python -m stdiag.fetch_tracking_docs                     # Millepede only
  python -m stdiag.fetch_tracking_docs --millepede-ref <tag>
  python -m stdiag.fetch_tracking_docs --update-acts --acts-ref <full hash>
  python -m stdiag.fetch_tracking_docs --check             # status, offline
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from stdiag.paths import MILLEPEDE_REPO, TRACKING_DOCS

ACTS_REPO = "https://github.com/sPHENIX-Collaboration/acts.git"
# Parts of ACTS kept; sPHENIX code includes Acts/*, ActsExamples/* and
# ActsPlugins/{Root,Json}/*. Folders missing at a given ref are skipped.
ACTS_PATHS = ["docs", "Core", "Plugins/Root", "Plugins/Json",
              "Examples/Framework", "Examples/Detectors"]
ACTS_SUFFIXES = {".md", ".dox", ".bib", ".hpp", ".ipp", ".h", ".cpp"}
# Millepede-II release read from the local clone. The draft manual text is
# unchanged since it entered the repository (2025-11-18, first released in
# V04-18-00) except for notation (MathJax macros, figure format).
MILLEPEDE_REF = "V05-01-08"
MILLEPEDE_OFFICIAL = "https://gitlab.desy.de/millepede/millepede-ii.git"
MILLEPEDE_DOCS_URL = "https://millepede.pages.desy.de/millepede-ii/"
MILLEPEDE_MANUAL = "doc/draftManual.md"
# Files kept, by folder and suffix.
MILLEPEDE_PARTS = {"doc": (".md",), "src": (".f90", ".c")}
# Labels the docs map cites; the manual must define all of them.
MILLEPEDE_CITED_LABELS = ["ssec-meapar", "ssec_large", "sssec-outlow",
                          "ssec-lincon", "ssec-textfiles", "sssec-methods",
                          "ssec-globalfit", "ssec-align"]
HEADING_LEVELS = {"page": 1, "section": 2, "subsection": 3,
                  "subsubsection": 4, "paragraph": 5}
LABEL_LINE = re.compile(
    r"^\s*\\(page|section|subsection|subsubsection|paragraph)\s+(\S+)\s*(.*)$")
ANCHOR_LINE = re.compile(r"^\s*\\anchor\s+(\S+)\s*$")


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def run(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


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


def git_bytes(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True).stdout


def git_text(repo: Path, *args: str) -> str:
    return git_bytes(repo, *args).decode("utf-8", errors="replace").strip()


def manual_text(source: str) -> tuple[str, list[str]]:
    """The manual with each Doxygen label as a heading ending in {#label}."""
    lines, labels = [], []
    for line in source.splitlines():
        heading = LABEL_LINE.match(line)
        anchor = ANCHOR_LINE.match(line)
        if heading:
            kind, label, title = heading.groups()
            lines.append(f"{'#' * HEADING_LEVELS[kind]} {title} {{#{label}}}")
            labels.append(label)
        elif anchor:
            lines.append(f"{{#{anchor.group(1)}}}")
            labels.append(anchor.group(1))
        else:
            lines.append(line)
    return "\n".join(lines) + "\n", labels


def read_millepede(repo: Path, ref: str) -> tuple[dict[str, bytes], dict]:
    """Files and provenance of Millepede-II at ref; nothing is written.

    Files are read with `git show <commit>:<path>`, so neither the commit
    the clone has checked out nor local edits change the result.
    """
    if not (repo / ".git").exists():
        sys.exit(f"ERROR: no Millepede-II clone at {repo}; see --help for "
                 "how to get it, or pass --millepede-repo.")
    try:
        commit = git_text(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
        tag_object = git_text(repo, "rev-parse", "--verify", ref)
        date = git_text(repo, "log", "-1", "--format=%cI", commit)
        names = git_text(repo, "ls-tree", "-r", "--name-only", commit, "--",
                         *MILLEPEDE_PARTS).splitlines()
        files = {name: git_bytes(repo, "show", f"{commit}:{name}")
                 for name in names
                 if "/" in name and name.endswith(
                     MILLEPEDE_PARTS.get(name.split("/")[0], ()))}
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or b"").decode(errors="replace").strip()
        sys.exit(f"ERROR: cannot read {ref} from {repo}: {detail}")
    origin = subprocess.run(["git", "-C", str(repo), "remote", "get-url",
                             "origin"], capture_output=True,
                            text=True).stdout.strip()
    if MILLEPEDE_MANUAL not in files:
        sys.exit(f"ERROR: {MILLEPEDE_MANUAL} not found at {ref} in {repo}.")
    _, labels = manual_text(files[MILLEPEDE_MANUAL].decode("utf-8", "replace"))
    missing = [label for label in MILLEPEDE_CITED_LABELS if label not in labels]
    if missing:
        sys.exit(f"ERROR: the manual at {ref} lacks labels the docs map "
                 f"cites: {', '.join(missing)}.")
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode() + b"\0" + files[name] + b"\0")
    source = {
        "repository": origin, "official_repository": MILLEPEDE_OFFICIAL,
        "documentation": MILLEPEDE_DOCS_URL, "ref": ref, "commit": commit,
        "tag_object": None if tag_object == commit else tag_object,
        "commit_date": date,
        "files": {part: sum(name.startswith(part + "/") for name in files)
                  for part in MILLEPEDE_PARTS},
        "manual_sha256": hashlib.sha256(files[MILLEPEDE_MANUAL]).hexdigest(),
        "content_sha256": digest.hexdigest()}
    return files, source


def write_millepede(files: dict[str, bytes], destination: Path) -> None:
    """Replace destination with the files plus the citable manual text."""
    staging = destination.with_name(destination.name + ".new")
    if staging.exists():
        shutil.rmtree(staging)
    for name, content in files.items():
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    text, _ = manual_text(files[MILLEPEDE_MANUAL].decode("utf-8", "replace"))
    (staging / "draftman_page.txt").write_text(text)
    if destination.exists():
        shutil.rmtree(destination)
    staging.rename(destination)


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
    millepede = sources.get("millepede", {})
    if not (out / "millepede" / "draftman_page.txt").exists() or not millepede:
        problems.append("Millepede-II missing: run without options (reads "
                        "the local clone)")
    elif "commit" not in millepede:
        problems.append("Millepede-II is in the old saved-page format: run "
                        "without options to rebuild it from the clone")
    else:
        empty = [part for part in MILLEPEDE_PARTS
                 if not any((out / "millepede" / part).glob("*"))]
        if empty:
            problems.append(f"Millepede-II parts without files: "
                            f"{', '.join(empty)}; run without options")
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
    parser.add_argument("--millepede-repo", type=Path, default=MILLEPEDE_REPO,
                        help="local clone of the Millepede-II repository")
    parser.add_argument("--millepede-ref", default=MILLEPEDE_REF,
                        help=f"release tag or commit (default {MILLEPEDE_REF})")
    parser.add_argument("--check", action="store_true",
                        help="report the snapshot status; change nothing")
    args = parser.parse_args()

    if args.check:
        print(json.dumps(read_sources(args.out), indent=2))
        problems = status(args.out)
        print("\n".join(problems) or "snapshot complete")
        sys.exit(1 if problems else 0)

    # Read and check Millepede-II before anything in the snapshot changes.
    files, millepede = read_millepede(args.millepede_repo, args.millepede_ref)
    args.out.mkdir(parents=True, exist_ok=True)
    sources = read_sources(args.out)
    if args.update_acts:
        sources["acts"] = fetch_acts(args.acts_ref, args.out / "acts")
    write_millepede(files, args.out / "millepede")
    sources["millepede"] = millepede | {"built_utc": now_utc()}
    sources.pop("fetched_utc", None)  # old single-fetch format
    (args.out / "SOURCES.json").write_text(json.dumps(sources, indent=2)
                                           + "\n")
    print(json.dumps(sources, indent=2))
    problems = status(args.out)
    if problems:
        sys.exit("\n".join(problems))


if __name__ == "__main__":
    main()
