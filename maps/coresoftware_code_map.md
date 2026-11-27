# coresoftware code map for residual diagnostics

Where the MVTX/INTT residuals in the `ntp` tree come from, traced back from
the module that writes them. Paths are relative to `coresoftware/`. Line
numbers were checked at commit 9c6cf8db3 (2026-11-25) and drift with later
commits; always confirm with `git show <rev>:<path>` or `git blame`. The
request also lists the commits of the range in each tier (tier 1 and tier 2
below); only those are candidates.

## Tier 1: `offline/packages/TrackerMillepedeAlignment/`

| File | What it does for the residuals |
|---|---|
| `MakeMilleFiles.cc` | Books the `ntp` (l. 250–252: `layer:stave:chip:residualX:residualY:...`) and fills it (l. 1859); reads the residual from the alignment state; applies the track selection (pT cut `m_minPt` at l. 597, set with `set_pt_cut`, default 0 in `.h` l. 164; MVTX/INTT hit and vertex cuts) and writes the vertex residual −DCA (l. 770–772). |
| `AlignmentDefs.cc`, `.h` | Global labels and alignment groupings (`mvtxGrp`: sensor, stave, layer, clamshell; `inttGrp`). |
| `Mille.cc`, `.h` | Mille binary writer (sPHENIX's own copy). |
| `HelicalFitter.cc` | Alternative residual producer with its own `ntp` (`crosshalfmvtx`, l. 134–138). |

## Tier 2: what MakeMilleFiles depends on

MakeMilleFiles reads five nodes; each is filled by other modules. Follow the
link that matches the finding.

| Node read by MakeMilleFiles | Filled by (tier 2) | What a change there does |
|---|---|---|
| `SvtxAlignmentStateMap` | `trackreco/PHActsTrkFitter.cc` → `trackreco/ActsAlignmentStates.cc` (l. 149 `localResidual = localMeas - localState`, smoothed state; stored at l. 221) | Residual definition and sign, derivatives, fit settings (outlier finder off by default: `PHActsTrkFitter.h` l. 289) |
| (inside that fit) | `trackreco/MakeSourceLinks.cc` (covariance l. 249–254) with `trackbase/ClusterErrorPara.cc` (`get_clusterv5_modified_error`, returns σ², l. 770; data-reco silicon factors l. 724–744) | Measurement errors: hit weights, so the widths of the biased residuals |
| `ActsGeometry` | `trackreco/MakeActsGeometry.cc` (`buildAllGeometry` l. 377, `buildActsSurfaces` l. 634) with `trackbase/AlignmentTransformation.cc` (`createMap` l. 35: local file or CDB `TRACKINGALIGNMENT` l. 96; parameter lines l. 140 and 159; `newMakeTransform` l. 388) and `trackbase/alignmentTransformationContainer.cc` | Sensor positions from the alignment parameters: mean shifts by sensor, stave, layer |
| `TRKR_CLUSTER` | `mvtx/`: `MvtxCombinedRawDataDecoder.cc` (hot-pixel masking l. 345–351, off by default), `MvtxClusterizer.cc` (`are_adjacent` l. 72, 94; `ClusterMvtx` l. 287; `ClusterMvtxRaw` l. 600), `MvtxClusterPruner.cc`; `intt/`: `InttCombinedRawDataDecoder.cc` (bad-channel map l. 328–351, permanent chip mask l. 354), `InttClusterizer.cc` (`ladder_are_adjacent` l. 55, 78) | Cluster positions, sizes and which hits exist: single-chip effects, missing or extra entries |
| `SvtxTrackMap`, `SvtxVertexMap` | The fit above; `trackreco/PHSimpleVertexFinder.cc` (standard vertexing), `trackreco/PHActsVertexPropagator.cc`; `trackbase_historic/TrackAnalysisUtils.cc` (`get_dca`: track − vertex, l. 215–255) | Track and vertex selection, DCA |

Also in tier 2 because MakeMilleFiles includes them: the rest of
`trackbase/` and `trackbase_historic/`, and `trackreco/ActsPropagator.cc`.

## Finding type → link to follow

| Finding | Start in tier 1 | Then the tier 2 link |
|---|---|---|
| Coherent stave/ladder- or layer-wide mean shift | `MakeMilleFiles.cc` fill and selection | `ActsGeometry`: `AlignmentTransformation.cc`, `MakeActsGeometry.cc` |
| Isolated chip/sensor mean shift | as above | `ActsGeometry` per-sensor transform; `TRKR_CLUSTER` positions (clusterizer, masking) |
| Width-only change | selection in `MakeMilleFiles.cc` | Measurement errors (`ClusterErrorPara.cc`, `MakeSourceLinks.cc`); fit settings |
| Missing, extra or low entries | selection in `MakeMilleFiles.cc` | `TRKR_CLUSTER`: decoders and their masks, clusterizers |
| Every chip shifted or sign flipped | `MakeMilleFiles.cc` fill | `ActsAlignmentStates.cc` residual definition |
| VERTEX (DCA) rows | vertex residual and pT cut in `MakeMilleFiles.cc` | `SvtxVertexMap`, `get_dca`, the fit |

If nothing in tier 1 or tier 2 explains the finding, the answer is: "The
reason could not be found from the changes in
offline/packages/TrackerMillepedeAlignment/ or the files it depends on."
Most residual changes between productions come from alignment constants,
conditions-database tags or macro settings rather than code; list those as
non-code causes to check.
