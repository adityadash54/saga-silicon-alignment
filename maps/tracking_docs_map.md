# Tracking references for checking coresoftware expert answers

All sources are local; there is no network access. The snapshot in
`tracking_docs/` (made by `fetch_tracking_docs.py`) records in
`tracking_docs/SOURCES.json` the ACTS commit and the Millepede-II release
and commit it was built from. If something you need is not in the
snapshot or the coresoftware checkout, say so and use "cannot verify".

## Versions

- **ACTS 45.0.0**: `sPHENIX` branch of `github.com/sPHENIX-Collaboration/acts`
  (commit in `SOURCES.json`; confirm the version from
  `tracking_docs/acts/version_number`). This is the fork sPHENIX builds
  against; `coresoftware` itself does not pin a version. Upstream `main` is
  a newer, different API, so do not rely on remembered upstream behaviour.
- **Millepede-II V05-01-08** (release and commit in `SOURCES.json`): the
  official repository (DESY GitLab, mirrored on GitHub) at that release.
  The draft manual by V. Blobel (Doxygen edition by C. Kleinwort) entered
  the repository on 2025-11-18; since then only its notation (MathJax
  macros) and figure format have changed. It is the manual the earlier web
  editions (e.g. V04-17-06) showed, but it was not compared word by word
  with them.

## Millepede-II draft manual (`tracking_docs/millepede/draftman_page.txt`, anchors as `{#anchor}`)

Made from `doc/draftManual.md` with each Doxygen section label written as
`{#label}` at the end of its heading; line numbers are those of
`doc/draftManual.md`. The URLs are for citing only; they cannot be fetched.

| Topic | Anchor | URL |
|---|---|---|
| Measurements, residual z = y − f(x), local/global derivatives, sigma; sign consistency | `{#ssec-meapar}` | https://millepede.pages.desy.de/millepede-ii/draftman_page.html#ssec-meapar |
| Large problems: global vs local parameters | `{#ssec_large}` | ...#ssec_large |
| Outliers | `{#sssec-outlow}`, `outlierdownweighting`, `chisqcut`, `dwfractioncut` | ...#sssec-outlow |
| Linear constraints, weak modes | `{#ssec-lincon}` | ...#ssec-lincon |
| Text (steering, constraint, parameter) files; `entries` cut | `{#ssec-textfiles}` | ...#ssec-textfiles |
| Solution methods (inversion, diagonalization, MINRES, ...) | `{#sssec-methods}` | ...#sssec-methods |
| Global fit output and checks | `{#ssec-globalfit}` | ...#ssec-globalfit |
| Track-based alignment | `{#ssec-align}` | ...#ssec-align |

## Millepede-II documentation and `pede` source (`tracking_docs/millepede/doc/`, `src/`)

The manual describes the method (written 2007); the option pages and the
source show what the release actually does. Where they differ, the source
decides; say so.

| Topic | Where |
|---|---|
| Steering options (`entries`, `chisqcut`, `outlierdownweighting`, `dwfractioncut`, `method`, `constraint`, `regularisation`, ...) | `doc/options.md` (one `## <option>` heading each) |
| What changed relative to the draft manual | `doc/majorChanges.md`, `doc/changeLog.md`, `doc/releases.md` |
| Exit codes, hard-coded limits, troubleshooting | `doc/exitCodes.md`, `doc/hardcodedLimits.md`, `doc/troubleshooting.md` |
| Worked example (steering, constraints, output) | `doc/example.md` |
| `pede` main program: reading steering and binary files, entries cut (`mreqenf`), outlier treatment, `chisqcut`, constraints, solution | `src/pede.f90` (module data in `src/mpmod.f90`) |
| Fortran Mille writer (record format) | `src/mille.f90` |
| Numerics (inversion, diagonalization, constraint handling) | `src/mpnum.f90`, `src/mpqldec.f90`, `src/minres*.f90` |

The C++ `Mille.cc` writer is not part of this repository, so sPHENIX's copy
(`offline/packages/TrackerMillepedeAlignment/Mille.cc`) cannot be compared
with an upstream version here.

## ACTS 45.0.0 (`tracking_docs/acts/`)

| Topic | Snapshot path |
|---|---|
| Geometry context (how alignment enters transforms) | `docs/groups/context.md`; `Core/include/Acts/Geometry/GeometryContext.hpp` |
| Geometry, surfaces, local frames | `docs/groups/detector_descr/geometry.md`; `docs/old/core/geometry/surfaces.md`, `concepts.md`, `geometry_id.md`; `Core/include/Acts/Surfaces/` |
| Measurements and track parameters (loc0/loc1, bound parameters) | `docs/groups/eventdata/measurement.md`, `tracks.md`; `docs/old/core/eventdata/measurements.md`, `parametrization.md`; `Core/include/Acts/EventData/` |
| Track fitting (KF, GX2F, GSF) | `docs/groups/track_fitting.md`; `docs/old/core/reconstruction/track_fitting.md`; `Core/include/Acts/TrackFitting/` |
| Alignment example (ACTS's own approach) | `docs/old/examples/alignment.md` |
| Versioning policy | `docs/pages/versioning.md` |
| Implementations behind the core headers | `Core/src/` |
| `ActsExamples/...` includes used by sPHENIX (EventData, Framework, Utilities, MagneticField, TGeoDetector) | `Examples/Framework/`, `Examples/Detectors/` (Glob for the header name) |
| `ActsPlugins/Root`, `ActsPlugins/Json` includes | `Plugins/Root/`, `Plugins/Json/` |

## sPHENIX code that uses them (`coresoftware/`, read-only)

| What | Where |
|---|---|
| Mille binary writer (sPHENIX's own copy) | `offline/packages/TrackerMillepedeAlignment/Mille.cc`, `Mille.h` |
| Global parameter labels | `offline/packages/TrackerMillepedeAlignment/AlignmentDefs.h` (`getMvtxGlobalLabels`, `getInttGlobalLabels`, vertex labels 60000000–60000002) and `AlignmentDefs.cc` |
| Residuals and derivatives written to Mille and `ntp` | `offline/packages/TrackerMillepedeAlignment/MakeMilleFiles.cc`; `HelicalFitter.cc` |
| Residual definition from the ACTS fit | `offline/packages/trackreco/ActsAlignmentStates.cc` (`localMeas - localState`) |
| Alignment parameters into ACTS transforms | `offline/packages/trackbase/AlignmentTransformation.cc`, `alignmentTransformationContainer.cc` |
| ACTS geometry building and fitting | `offline/packages/trackreco/MakeActsGeometry.cc`, `PHActsTrkFitter.cc` |

## Alignment literature (`docs/TrackingLiterature/`, local hybrid search)

Papers and talks on track-based alignment and track fitting from other
experiments, kept locally by the user. The request
already contains the best passages for the finding under "Literature
passages". For more, run the search command named in your instructions with a
query, for example `search twist weak mode mass constraint`;
`--mode bm25` ranks by exact terms only. Each hit gives the title, pages and
PDF path: Read that page (the Read tool takes a `pages` range for PDFs) before
citing it, and cite as *title, p. N*. Use the literature for how other
experiments diagnosed and constrained similar effects (weak modes,
constraints, systematic distortions); it does not override ACTS,
Millepede-II or the sPHENIX code.
