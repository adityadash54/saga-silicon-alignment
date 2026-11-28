"""Range clones, side-labelled commit lists, later changes and local diffs.

A toy history (no real coresoftware):

    c0 - c1 - B ----- m1 ----- m2        main (source HEAD)
               \\
                r1                       ref branch

reference build = r1, comparison build = m1, m2 = a later fix.
"""

from __future__ import annotations

import json

import pytest

from stdiag import coresoftware_expert as ce
from stdiag import review
from stdiag.range_clone import RangeCloneError, git, range_clone

TIER1 = "offline/packages/TrackerMillepedeAlignment/MakeMilleFiles.cc"
TIER2 = "offline/packages/trackbase/ClusterErrorPara.cc"
OUTSIDE = "offline/packages/tpc/TpcClusterizer.cc"


def commit(repo, message, name):
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((path.read_text() if path.exists() else "") + message
                    + "\n")
    git(repo, "add", name)
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t",
        "-c", "commit.gpgSign=false", "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD").stdout.strip()


@pytest.fixture
def history(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    ids = {"c0": commit(repo, "old start", TIER1),
           "c1": commit(repo, "old error change", TIER2),
           "B": commit(repo, "base", OUTSIDE)}
    git(repo, "checkout", "-q", "-b", "ref")
    ids["r1"] = commit(repo, "branch selection change", TIER1)
    git(repo, "checkout", "-q", "main")
    ids["m1"] = commit(repo, "main error change", TIER2)
    ids["m2"] = commit(repo, "later fix of the error change", TIER2)
    return repo, ids


def commits_in(clone):
    return set(git(clone, "rev-list", "--all").stdout.split())


def present(clone, commit_id):
    return git(clone, "cat-file", "-e", f"{commit_id}^{{commit}}",
               check=False).returncode == 0


def test_branches_keep_both_sides_and_cut_shared_history(history, tmp_path):
    repo, ids = history
    clone = range_clone(repo, ids["r1"], ids["m1"], tmp_path / "clones")
    assert git(clone, "rev-parse", "HEAD").stdout.strip() == ids["m1"]
    refs = git(clone, "for-each-ref", "--format=%(refname:short) "
               "%(objectname)").stdout.split("\n")
    assert f"reference {ids['r1']}" in refs
    assert f"comparison {ids['m1']}" in refs
    assert commits_in(clone) == {ids["r1"], ids["m1"], ids["B"]}
    for gone in ("c0", "c1", "m2"):
        assert not present(clone, ids[gone])
    # lines older than the builds' shared base are marked ^ in blame
    blame = git(clone, "blame", TIER1).stdout
    assert blame.startswith("^")
    # the source is only read
    assert git(repo, "rev-parse", "HEAD").stdout.strip() == ids["m2"]
    assert git(repo, "status", "--porcelain").stdout == ""


def test_cut_commits_of_a_late_merged_branch_are_not_listed(history,
                                                            tmp_path):
    """A branch forked at c1 (an old in-scope commit) and merged after B:
    c1 becomes a cut commit reached from the comparison side; it is in
    both builds and must not be listed as comparison only."""
    repo, ids = history
    git(repo, "checkout", "-q", "-b", "side", ids["c1"])
    side = commit(repo, "side error change", TIER2 + ".side")
    git(repo, "checkout", "-q", "main")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "merge", "-q",
        "--no-edit", "--no-ff", "side")
    merged = git(repo, "rev-parse", "HEAD").stdout.strip()
    clone = range_clone(repo, ids["B"], merged, tmp_path / "clones")
    assert ids["c1"] in commits_in(clone)                  # cut commit, kept
    revision = {"from": ids["B"], "to": merged}
    subjects = [item["subject"] for tier in
                ce.traced_commits(clone, revision).values() for item in tier]
    assert "old error change" not in subjects
    assert sorted(subjects) == ["later fix of the error change",
                                "main error change", "side error change"]
    assert subjects == [item["subject"] for tier in
                        ce.traced_commits(repo, revision).values()
                        for item in tier]                  # same as source
    assert git(clone, "for-each-ref", "refs/shared").stdout.count("\n") == 2


def test_linear_range_and_cache(history, tmp_path):
    repo, ids = history
    clone = range_clone(repo, ids["c1"], ids["m1"], tmp_path / "clones")
    assert commits_in(clone) == {ids["c1"], ids["B"], ids["m1"]}
    assert not present(clone, ids["c0"])
    again = range_clone(repo, ids["c1"][:9], ids["m1"][:9],
                        tmp_path / "clones")
    assert again == clone


def test_same_version_twice_has_no_history(history, tmp_path):
    repo, ids = history
    clone = range_clone(repo, ids["m1"], ids["m1"], tmp_path / "clones")
    assert commits_in(clone) == {ids["m1"]}


def test_later_commits_only_on_request(history, tmp_path):
    repo, ids = history
    clone = range_clone(repo, ids["r1"], ids["m1"], tmp_path / "clones",
                        later=True)
    assert git(clone, "rev-parse", "later").stdout.strip() == ids["m2"]
    assert git(clone, "rev-parse", "HEAD").stdout.strip() == ids["m1"]
    revision = {"from": ids["r1"][:12], "to": ids["m1"][:12]}
    later = ce.later_commits(clone, revision)
    assert [item["subject"] for item in later] == [
        "later fix of the error change"]
    text = ce.traced_commits_section(clone, revision)
    assert "### Later changes" in text and "never a hypothesis" in text
    hidden = range_clone(repo, ids["r1"], ids["m1"], tmp_path / "clones")
    assert hidden != clone and not ce.has_later(hidden)
    assert "Later changes" not in ce.traced_commits_section(hidden, revision)


def test_commit_list_marks_the_side(history, tmp_path):
    repo, ids = history
    clone = range_clone(repo, ids["r1"], ids["m1"], tmp_path / "clones")
    revision = {"from": ids["r1"][:12], "to": ids["m1"][:12]}
    found = ce.traced_commits(clone, revision)
    assert [(i["subject"], i["side"]) for i in found["tier 1"]] == [
        ("branch selection change", "reference only")]
    assert [(i["subject"], i["side"]) for i in found["tier 2"]] == [
        ("main error change", "comparison only")]
    text = ce.traced_commits_section(clone, revision)
    assert "[reference only]" in text and "[comparison only]" in text
    assert "in one build but not the other" in text


def test_missing_version_is_an_error(history, tmp_path):
    repo, ids = history
    with pytest.raises(RangeCloneError, match="not a commit"):
        range_clone(repo, "deadbeef", ids["m1"], tmp_path / "clones")


def test_prepare_repo_uses_the_checkout_for_a_date_window(history):
    repo, _ = history
    assert ce.prepare_repo(repo, {"since": "2026-01-01"}) == repo


def test_prepare_repo_clones_into_the_configured_folder(
        history, scratch_clones_dir):
    repo, ids = history
    clone = ce.prepare_repo(repo, {"from": ids["c1"], "to": ids["m1"]})
    assert clone.parent == scratch_clones_dir
    assert ce.has_later(clone)                       # production default


def test_prompt_rules_for_versions_and_later_changes():
    prompt = ce.EXPERT_PROMPT
    assert "reference only" in prompt and "comparison only" in prompt
    assert "^<hash>" in prompt and "never a hypothesis" in prompt
    assert "### Later changes" in prompt
    assert "Local changes" in prompt


# --- local diffs --------------------------------------------------------------

DIFF = f"""diff --git a/{TIER1} b/{TIER1}
--- a/{TIER1}
+++ b/{TIER1}
@@ -1 +1 @@
-float m_minPt = 0.0;
+float m_minPt = 0.5;
diff --git a/{OUTSIDE} b/{OUTSIDE}
--- a/{OUTSIDE}
+++ b/{OUTSIDE}
@@ -1 +1 @@
-a
+b
"""


def test_filter_diff_keeps_in_scope_files():
    kept, others = ce.filter_diff(DIFF)
    assert "m_minPt = 0.5" in kept and "+b" not in kept
    assert others == [OUTSIDE]


def test_request_has_local_changes_but_not_the_path(history):
    repo, ids = history
    labels = {"A": {"label": "A", "local_diff": "/home/me/a.diff",
                    ce.LOCAL_DIFF_KEY: DIFF},
              "B": {"label": "B"}}
    text = ce.build_request({"id": "F1"}, {"since": "2026-01-01"}, labels,
                            repo)
    assert "## Local changes" in text
    assert "### Local changes in A" in text and "m_minPt = 0.5" in text
    assert "out of scope: " + OUTSIDE in text
    assert "/home/me/a.diff" not in text
    assert "Local changes in B" not in text


def test_read_labels_loads_local_diff_relative_to_the_csv(tmp_path):
    (tmp_path / "a.diff").write_text(DIFF)
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("label,coresoftware_commit,local_diff\n"
                          "A,abc,a.diff\nB,def,\n")
    labels = review.read_labels(labels_csv, ["A", "B"])
    assert labels["A"][ce.LOCAL_DIFF_KEY] == DIFF
    assert ce.LOCAL_DIFF_KEY not in labels["B"]
    labels_csv.write_text("label,local_diff\nA,missing.diff\n")
    with pytest.raises(SystemExit, match="not found"):
        review.read_labels(labels_csv, ["A"])


def test_check_reuses_the_recorded_clone(tmp_path):
    clone = tmp_path / "clone"
    (clone / ".git").mkdir(parents=True)
    findings = tmp_path / "findings_A_vs_B.json"
    assert review.recorded_repo(findings, {"repo": str(clone)}) == clone
    state = tmp_path / "dialogue.json"
    state.write_text(json.dumps({"repo": str(clone)}))
    assert review.recorded_repo(findings, {"state_file": state.name}) == clone
    assert review.recorded_repo(findings, {"repo": str(tmp_path / "gone")}) \
        is None
