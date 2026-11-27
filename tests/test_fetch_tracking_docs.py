"""Local tracking docs snapshot (no network: ACTS download is faked or
local; Millepede-II is a small local git repository)."""

from __future__ import annotations

import fnmatch
import json
import shutil
import subprocess
import sys

import pytest

from stdiag import fetch_tracking_docs as fetch
from stdiag import coresoftware_expert, tracking_expert

MANUAL = "\n".join([
    r"\page draftman_page Millepede II - Draft Manual",
    r"\section sec_intro Introduction",
    "Text before the labels.",
    *[rf"\subsection {label} Title of {label}" + "\nBody of " + label
      for label in fetch.MILLEPEDE_CITED_LABELS],
    r"\anchor an-entries",
    "Residual z = y - f(x).", ""])


def git(repo, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t",
         "-c", "tag.gpgSign=false", "-c", "commit.gpgSign=false", *args],
        cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def mp_repo(tmp_path):
    """A small Millepede-II-like repository with release tag V05-01-08,
    a later commit and an uncommitted edit (neither may be read)."""
    repo = tmp_path / "millepede-ii"
    for name, text in {"doc/draftManual.md": MANUAL,
                       "doc/options.md": " ## entries\nentries cut\n",
                       "doc/fig1.tex": "not kept\n",
                       "src/pede.f90": "program pede\nend program\n",
                       "src/stat_compat.c": "int x;\n",
                       "src/CMakeLists.txt": "not kept\n",
                       "CMakeLists.txt": "not kept\n"}.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "release")
    git(repo, "tag", "-a", "V05-01-08", "-m", "V05-01-08")
    (repo / "doc" / "draftManual.md").write_text(MANUAL + "LATER COMMIT\n")
    git(repo, "commit", "-q", "-am", "later")
    (repo / "doc" / "options.md").write_text("UNCOMMITTED EDIT\n")
    return repo


@pytest.fixture
def fake_acts(monkeypatch):
    calls = []

    def fake(ref, destination):
        calls.append(ref)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "version_number").write_text("45.0.0\n")
        for part in fetch.ACTS_PATHS:
            (destination / part).mkdir(parents=True, exist_ok=True)
            (destination / part / "File.hpp").write_text("// header\n")
        return {"ref": ref, "commit": "a" * 40, "version_number": "45.0.0"}

    monkeypatch.setattr(fetch, "fetch_acts", fake)
    return calls


def run_main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["fetch_tracking_docs", *args])
    fetch.main()


def build(monkeypatch, out, repo, *extra):
    run_main(monkeypatch, "--out", str(out), "--millepede-repo", str(repo),
             *extra)


def test_manual_labels_become_anchors_on_the_same_lines():
    text, labels = fetch.manual_text(MANUAL)
    assert labels[:2] == ["draftman_page", "sec_intro"]
    assert set(fetch.MILLEPEDE_CITED_LABELS) <= set(labels)
    assert "an-entries" in labels
    lines = text.splitlines()
    assert len(lines) == len(MANUAL.splitlines())   # citations keep lines
    assert lines[0] == "# Millepede II - Draft Manual {#draftman_page}"
    assert "### Title of ssec-meapar {#ssec-meapar}" in lines
    assert "{#an-entries}" in lines


def test_first_build_reads_the_tag_only(tmp_path, mp_repo, fake_acts,
                                        monkeypatch):
    out = tmp_path / "docs"
    build(monkeypatch, out, mp_repo, "--update-acts")
    mp = out / "millepede"
    text = (mp / "draftman_page.txt").read_text()
    assert "{#ssec-meapar}" in text and "LATER COMMIT" not in text
    assert (mp / "doc" / "options.md").read_text() == " ## entries\nentries cut\n"
    assert sorted(p.name for p in (mp / "src").iterdir()) == [
        "pede.f90", "stat_compat.c"]
    assert not (mp / "doc" / "fig1.tex").exists()
    assert not (mp / "CMakeLists.txt").exists()
    sources = json.loads((out / "SOURCES.json").read_text())["millepede"]
    assert sources["ref"] == "V05-01-08"
    assert sources["commit"] == git(mp_repo, "rev-parse", "V05-01-08^{commit}")
    assert sources["tag_object"] == git(mp_repo, "rev-parse", "V05-01-08")
    assert sources["files"] == {"doc": 2, "src": 2}
    assert len(sources["content_sha256"]) == 64
    assert fetch.status(out) == []


def test_rebuild_keeps_acts_and_other_ref(tmp_path, mp_repo, fake_acts,
                                          monkeypatch):
    out = tmp_path / "docs"
    build(monkeypatch, out, mp_repo, "--update-acts")
    build(monkeypatch, out, mp_repo, "--millepede-ref", "HEAD")
    assert fake_acts == ["sPHENIX"]          # ACTS downloaded once only
    assert (out / "acts" / "version_number").exists()
    assert "LATER COMMIT" in (out / "millepede" / "draftman_page.txt").read_text()
    sources = json.loads((out / "SOURCES.json").read_text())
    assert sources["acts"]["commit"] == "a" * 40
    assert sources["millepede"]["tag_object"] is None


@pytest.mark.parametrize("problem", ["no_clone", "no_tag", "missing_label"])
def test_bad_millepede_input_leaves_snapshot_untouched(
        tmp_path, mp_repo, fake_acts, monkeypatch, problem):
    out = tmp_path / "docs"
    build(monkeypatch, out, mp_repo, "--update-acts")
    before = (out / "SOURCES.json").read_text()
    repo, ref = mp_repo, "V05-01-08"
    if problem == "no_clone":
        repo = tmp_path / "nowhere"
    elif problem == "no_tag":
        ref = "V99-99-99"
    else:
        (repo / "doc" / "draftManual.md").write_text(
            MANUAL.replace(r"\subsection ssec-lincon", "ssec-lincon"))
        git(repo, "commit", "-q", "-am", "drop a label")
        ref = "HEAD"
    with pytest.raises(SystemExit) as error:
        build(monkeypatch, out, repo, "--update-acts", "--millepede-ref", ref)
    assert {"no_clone": "no Millepede-II clone", "no_tag": "cannot read",
            "missing_label": "ssec-lincon"}[problem] in str(error.value)
    assert (out / "SOURCES.json").read_text() == before
    assert fake_acts == ["sPHENIX"]          # nothing downloaded on failure


def test_missing_acts_and_old_format_are_reported(tmp_path, mp_repo,
                                                  monkeypatch):
    out = tmp_path / "docs"
    with pytest.raises(SystemExit, match="--update-acts"):
        build(monkeypatch, out, mp_repo)
    with pytest.raises(SystemExit) as check:
        run_main(monkeypatch, "--out", str(out), "--check")
    assert check.value.code == 1
    sources = json.loads((out / "SOURCES.json").read_text())
    sources["millepede"] = {"local_file": "draftman_page.html", "sha256": "x"}
    (out / "SOURCES.json").write_text(json.dumps(sources))
    assert any("old saved-page format" in problem
               for problem in fetch.status(out))


def test_docs_map_cites_exactly_the_checked_labels():
    import re
    from stdiag.paths import DOCS_MAP
    millepede = DOCS_MAP.read_text().split("## ACTS")[0]
    placeholders = {"anchor", "label"}     # used in the map's own prose
    cited = set(re.findall(r"\{#([A-Za-z0-9_-]+)\}", millepede)) - placeholders
    assert cited == set(fetch.MILLEPEDE_CITED_LABELS)


def test_tracking_expert_has_no_network_tools():
    for session in (None, "abc"):
        command = tracking_expert.claude_command("claude", "m", "high",
                                                 session_id=session)
        tools = command[command.index("--tools") + 1]
        assert "Web" not in tools
        allowed = command[command.index("--allowedTools") + 1:
                          command.index("--disallowedTools")]
        assert not any("Web" in rule for rule in allowed)
        disallowed = command[command.index("--disallowedTools") + 1:
                             command.index("--permission-mode")]
        assert {"WebFetch", "WebSearch"} <= set(disallowed)
        prompt = command[command.index("--append-system-prompt") + 1]
        assert "no network access" in prompt
        assert "www.desy.de" not in prompt


def test_require_inputs_needs_a_complete_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking_expert, "DOCS_DIR", tmp_path / "empty")
    with pytest.raises(SystemExit, match="incomplete"):
        tracking_expert.require_inputs(tmp_path)


# --- ACTS download paths (git is faked) --------------------------------------

def fake_clone(parts):
    """A clone_acts stand-in that creates the given ACTS parts."""
    def clone(ref, clone_dir):
        clone_dir.mkdir(parents=True)
        (clone_dir / "version_number").write_text("45.0.0\n")
        for part in parts:
            (clone_dir / part / "include").mkdir(parents=True)
            (clone_dir / part / "include" / "A.hpp").write_text("//\n")
            (clone_dir / part / "notes.txt").write_text("not kept\n")
    return clone


@pytest.fixture
def old_acts(tmp_path):
    destination = tmp_path / "docs" / "acts"
    destination.mkdir(parents=True)
    (destination / "OLD").write_text("previous copy\n")
    return destination


def test_clone_commands_use_cone_mode(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(fetch, "run",
                        lambda command, cwd=None: commands.append(command) or "")
    fetch.clone_acts("sPHENIX", tmp_path / "a")
    assert commands[0][:2] == ["git", "clone"] and "--branch" in commands[0]
    assert commands[1][:4] == ["git", "sparse-checkout", "set", "--cone"]
    commands.clear()
    commit = "b" * 40
    fetch.clone_acts(commit, tmp_path / "b")
    assert ["git", "fetch", "-q", "--depth", "1", "origin", commit] in commands
    assert ["git", "sparse-checkout", "set", "--cone", *fetch.ACTS_PATHS] \
        in commands
    assert commands[-1] == ["git", "checkout", "-q", "FETCH_HEAD"]


def test_failed_download_keeps_old_copy(old_acts, monkeypatch):
    def fail(command, cwd=None):
        raise subprocess.CalledProcessError(128, command,
                                            stderr="fatal: unable to access")
    monkeypatch.setattr(fetch, "run", fail)
    with pytest.raises(SystemExit, match="unchanged"):
        fetch.fetch_acts("sPHENIX", old_acts)
    assert (old_acts / "OLD").exists()
    assert not old_acts.with_name("acts.new").exists()


def test_unexpected_layout_keeps_old_copy(old_acts, monkeypatch):
    monkeypatch.setattr(fetch, "clone_acts", fake_clone(["docs", "Core"]))
    monkeypatch.setattr(fetch, "run", lambda command, cwd=None: "c" * 40)
    with pytest.raises(SystemExit, match="no files in Plugins/Root"):
        fetch.fetch_acts("sPHENIX", old_acts)
    assert (old_acts / "OLD").exists()
    assert not old_acts.with_name("acts.new").exists()


def test_successful_download_replaces_and_counts(old_acts, monkeypatch):
    monkeypatch.setattr(fetch, "clone_acts", fake_clone(fetch.ACTS_PATHS))
    monkeypatch.setattr(fetch, "run", lambda command, cwd=None: "c" * 40)
    source = fetch.fetch_acts("sPHENIX", old_acts)
    assert not (old_acts / "OLD").exists()
    assert source["files"] == {part: 1 for part in fetch.ACTS_PATHS}
    assert (old_acts / "Core" / "include" / "A.hpp").exists()
    assert not (old_acts / "Core" / "notes.txt").exists()


def test_check_reports_an_empty_acts_part(tmp_path, mp_repo, fake_acts,
                                          monkeypatch):
    out = tmp_path / "docs"
    build(monkeypatch, out, mp_repo, "--update-acts")
    shutil.rmtree(out / "acts" / "Plugins" / "Json")
    assert fetch.status(out) == [
        "ACTS parts without files: Plugins/Json; run with --update-acts"]


# --- read-only git rules (approximate matcher; the real check is Claude
# Code's own matching, to be tested by the planned permissions eval) ------

def permitted(command: str) -> bool:
    def matches(rules):
        return any(fnmatch.fnmatchcase(f"Bash({command})", rule)
                   for rule in rules)
    return (matches(coresoftware_expert.ALLOWED_TOOLS)
            and not matches(coresoftware_expert.DISALLOWED_TOOLS))


@pytest.mark.parametrize("command", [
    "git log --oneline -5", "git show abc123:offline/x.cc",
    "git diff abc def -- offline", "git log -S localMeas --oneline",
    "git blame -L 10,20 x.cc", "git branch -a", "git branch -r",
    "git branch --contains abc123", "git status", "git ls-files offline",
])
def test_read_only_git_commands_are_permitted(command):
    assert permitted(command)


@pytest.mark.parametrize("command", [
    "git log --output=/tmp/x -1", "git log -1 --output /tmp/x",
    "git show --output=/tmp/x HEAD", "git diff --output=/tmp/x",
    "git diff HEAD --output=/tmp/x",
    "git grep pattern", "git grep -Osh pattern", "git grep -iOsh pattern",
    "git branch -D main", "git branch new-branch",
    "git branch --list -D main", "git branch --contains abc -m x y",
    "git fetch origin", "git pull", "git clone https://example.org/r.git",
    "git checkout main", "git reset --hard",
    "curl https://example.org",
])
def test_write_execute_and_network_commands_are_refused(command):
    assert not permitted(command)


@pytest.fixture
def local_acts_repo(tmp_path):
    """A small git repo with the ACTS folder layout, served as file://."""
    repo = tmp_path / "acts_origin"
    for part in [*fetch.ACTS_PATHS, "Fatras", "Examples/Python"]:
        (repo / part / "include").mkdir(parents=True)
        (repo / part / "include" / "A.hpp").write_text("//\n")
    (repo / "version_number").write_text("45.0.0\n")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t",
           "-c", "init.defaultBranch=sPHENIX"]
    subprocess.run([*git, "init", "-q"], cwd=repo, check=True)
    subprocess.run([*git, "add", "."], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "acts"], cwd=repo, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                            capture_output=True, text=True).stdout.strip()
    return repo.as_uri(), commit


@pytest.mark.parametrize("by_hash", [False, True])
def test_real_git_sparse_checkout(tmp_path, local_acts_repo, monkeypatch,
                                  by_hash):
    url, commit = local_acts_repo
    monkeypatch.setattr(fetch, "ACTS_REPO", url)
    destination = tmp_path / "docs" / "acts"
    source = fetch.fetch_acts(commit if by_hash else "sPHENIX", destination)
    assert source["commit"] == commit
    assert source["version_number"] == "45.0.0"
    assert source["files"] == {part: 1 for part in fetch.ACTS_PATHS}
    assert (destination / "Examples" / "Framework" / "include" / "A.hpp").exists()
    assert not (destination / "Fatras").exists()            # not kept
    assert not (destination / "Examples" / "Python").exists()
