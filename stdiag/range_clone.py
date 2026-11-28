"""A coresoftware clone holding only what differs between two builds.

The coresoftware expert compares a reference and a comparison build made
from two explicit coresoftware versions. Commits that both builds contain
(the merge base and everything older) cannot explain a difference, so the
clone made here does not hold them:

- the work tree is checked out at the comparison version ("to"), so the
  agent reads the code that produced the comparison ntuples;
- refs `reference` and `comparison` point to the two versions;
- history is cut (shallow) just below the commits that are in one build
  but not the other (`git rev-list from...to`); `git blame` marks older
  lines with `^<hash>`;
- the cut commits themselves are in both builds, but with their parents
  gone git can no longer tell: each gets a ref under refs/shared/, and the
  commits between the builds are listed with
  `git log reference...comparison --not --glob=refs/shared/*` (SHARED_GLOB);
- with later=True, the commits of the source checkout's HEAD that are in
  neither build are kept too, under the ref `later` (fixes or reverts made
  after the comparison build; evidence, never a cause).

The source checkout is only read: the two versions (and HEAD) are fetched
from it into an empty repository. Clones are cached under clones_dir, one
per (from, to, later head). git runs without the user's git config (the
Claude Code sandbox refuses to read it); inside the sandbox, clones_dir must
be outside the workspace (e.g. under $TMPDIR), because the sandbox also
refuses writes to .git/config there.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

MARKER = "stdiag-range.json"
CLONE_FORMAT = 2
SHARED_GLOB = "refs/shared/*"
EXCLUDE_SHARED = ["--not", f"--glob={SHARED_GLOB}"]


def git_environment(cwd: Path) -> dict[str, str]:
    """Environment for git without the user's global config."""
    return {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull,
            "XDG_CONFIG_HOME": str(Path(cwd) / ".stdiag-no-user-config")}


def git(repo: Path, *arguments: str, check: bool = True
        ) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments], capture_output=True, text=True,
        check=check, env=git_environment(repo))


def resolve(repo: Path, revision: str) -> str | None:
    result = git(repo, "rev-parse", "--verify", "--quiet",
                 f"{revision}^{{commit}}", check=False)
    return result.stdout.strip() or None


def merge_bases(repo: Path, first: str, second: str) -> list[str]:
    return git(repo, "merge-base", "--all", first, second,
               check=False).stdout.split()


class RangeCloneError(RuntimeError):
    pass


def range_clone(source: Path, from_rev: str, to_rev: str, clones_dir: Path,
                later: bool = False) -> Path:
    """Return a cached clone of source with only from...to (and later).

    Raises RangeCloneError when a revision is missing or git fails.
    """
    source = Path(source).resolve()
    ends = {"from": resolve(source, from_rev), "to": resolve(source, to_rev)}
    missing = [f"{name} {rev}" for (name, full), rev
               in zip(ends.items(), (from_rev, to_rev)) if full is None]
    if missing:
        raise RangeCloneError(f"not a commit in {source}: "
                              + ", ".join(missing))
    head = resolve(source, "HEAD") if later else None
    key = f"{ends['from'][:12]}_{ends['to'][:12]}" + (
        f"_later{head[:12]}" if head else "")
    dest = Path(clones_dir).resolve() / key
    # "format" changes whenever the clone layout does, so older cached
    # clones are rebuilt.
    wanted = {"format": CLONE_FORMAT, "from": ends["from"], "to": ends["to"],
              "later": head}
    marker = dest / ".git" / MARKER
    if marker.exists() and json.loads(marker.read_text()) == wanted:
        return dest

    bases = merge_bases(source, ends["from"], ends["to"])
    tips = [ends["from"], ends["to"]] + ([head] if head else [])
    listing = git(source, "rev-list", "--boundary", *tips, "--not",
                  *bases).stdout.split()
    kept = [line for line in listing if not line.startswith("-")]
    # A tip that is itself in both builds (from == to, or HEAD older than
    # the builds) is cut too, or its whole history would come along.
    boundary = sorted({line[1:] for line in listing if line.startswith("-")}
                      | {tip for tip in tips if tip not in kept})

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    try:
        # No template: no sample hooks (the sandbox refuses .git/hooks).
        git(dest, "init", "-q", "--template=")
        git(dest, "fetch", "-q", "--no-tags", str(source), *dict.fromkeys(tips))
        git(dest, "checkout", "-q", "--detach", ends["to"])
        git(dest, "update-ref", "refs/heads/reference", ends["from"])
        git(dest, "update-ref", "refs/heads/comparison", ends["to"])
        if head:
            git(dest, "update-ref", "refs/heads/later", head)
        for number, commit in enumerate(boundary, start=1):
            git(dest, "update-ref", f"refs/shared/{number}", commit)
        (dest / ".git" / "FETCH_HEAD").unlink(missing_ok=True)
        if boundary:
            (dest / ".git" / "shallow").write_text("\n".join(boundary) + "\n")
        git(dest, "reflog", "expire", "--expire=now", "--all")
        git(dest, "gc", "-q", "--prune=now")
        present = int(git(dest, "rev-list", "--count", "--all").stdout)
        between = int(git(dest, "rev-list", "--count", "reference...comparison",
                          *EXCLUDE_SHARED).stdout)
    except (subprocess.CalledProcessError, OSError) as error:
        shutil.rmtree(dest, ignore_errors=True)
        detail = getattr(error, "stderr", "") or str(error)
        raise RangeCloneError(f"could not make the clone {dest}: "
                              f"{detail.strip()}") from error
    expected = len(set(kept) | set(boundary))
    expected_between = int(git(source, "rev-list", "--count",
                               f"{ends['from']}...{ends['to']}").stdout)
    if present != expected or between != expected_between:
        shutil.rmtree(dest, ignore_errors=True)
        raise RangeCloneError(
            f"clone {dest} has {present} commits ({expected} expected) and "
            f"{between} between the builds ({expected_between} expected)")
    marker.write_text(json.dumps(wanted) + "\n")
    return dest
