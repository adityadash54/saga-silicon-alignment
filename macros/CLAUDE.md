# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope

This directory holds the standalone ROOT macros of silicon-tracking-diagnostics.
The Python package `stdiag/` (e.g. `stdiag.compare_runs`), `tests/` and
`synthetic/` are in the parent directory and are not covered here.

The AGENTS.md files in the parent directories apply:
- Before editing, list the proposed changes with file and line numbers and ask.
- Ask, with the full path, before reading files outside `macros/`.
- Keep the generic default paths (`filelistN.txt:IterN`, `dca_fits_IterN.csv` in the
  macro signatures and the USER SETTINGS block of makeDcaZShiftedAlignment.C).
  Do not put real run numbers, sample names or iteration history in committed
  defaults; the repository is public.
- Do not commit generated ROOT/PDF/PNG/CSV output.

## Running

There is no build system, linter or test suite. The macros use only ROOT and the STL.
ROOT compiles them on the fly with ACLiC (`+`), and they run in batch mode.
To validate a change, compile with `+` without warnings, run on a short file list,
and compare the CSV and PDF with a previous run.

    root -l -b -q 'plot_fits_overlaid_stavechip.C+("list_Iter0.txt:Iter0,list_Iter1.txt:Iter1","out.pdf")'
    root -l -b -q 'plot_fits_overlaid_dcadistsfromstavechips.C+("list_Iter0.txt:Iter0","dca.pdf")'
    # from the repository root:
    root -l -b -q 'macros/plot_flagged_overlays.C+("runs/Iter0_vs_Iter1/flags_Iter0_vs_Iter1.csv","inputs/list_Iter0.txt:Iter0,inputs/list_Iter1.txt:Iter1")'
    root -l -b -q -e '.L makeDcaZShiftedAlignment.C' -e 'makeResponseAlignment(0, 0.020, "base.txt", "responseL0.txt")'

`makeDcaZShiftedAlignment()` with no arguments prints the available functions.
Each macro's header comment and the comment block above its entry-point function
document all arguments and outputs.

## Inputs

- `listfiles`: `"list.txt:label,list.txt:label"`. Each list file holds one ROOT file
  per line. Labels name the output CSVs and must be distinct when writing CSV.
  Inputs get colours and markers from `kInputColors`/`kInputMarkerStyles`
  (8 entries, then they repeat).
- ROOT trees written by MakeMilleFiles (all Float_t, lengths in mm):
  - `ntp`, one row per cluster of a track: `layer`, `stave`, `chip`,
    `residualX` (local r-phi), `residualY` (local z), plus `residual_vertexX/Y`,
    `track_pt`, `track_crossing` and optionally `clusphi`.
  - `track_ntp`, one row per track: `residual_vertexX/Y` (= -DCA_xy / -DCA_z),
    `track_phi`, `track_pt`.

## Data flow between macros

1. `plot_fits_overlaid_stavechip.C` fills and fits the per-chip/sensor residuals and
   writes `residual_fits_<label>.csv` (one row per layer, stave, chip and coord),
   `vertex_phi_fit_<label>.csv`, and a PDF plus PNGs.
   - VERTEX rows have layer = stave = -1. chip = -1 means all tracks, and
     chip = 0..13 are the track-phi bins over [-pi, pi).
   - INTT z uses a `box_gaus` fit; everything else uses a core Gaussian.
2. `stdiag.compare_runs` (Python, parent directory) compares those CSVs and writes
   `runs/<a>_vs_<b>/flags_<a>_vs_<b>.csv`.
3. `plot_flagged_overlays.C` reads that flags CSV and the same file lists, and
   re-fills and overlays only the flagged entries.
4. `plot_fits_overlaid_dcadistsfromstavechips.C` fills each chip with the DCA of the
   tracks crossing it and writes `dca_fits_<label>.csv` (same columns,
   coord `dca_xy`/`dca_z`).
5. `makeDcaZShiftedAlignment.C` reads `dca_fits_*.csv` and writes
   `localAlignmentParamsFile` variants. Only column 7 (dz, mm) of MVTX rows is changed.
   These files are used in a rerun of
   Fun4All_TrackReco_SiAlignment_ACTS_SiOnly_ReadPruned.C.

## Conventions that span files

- Each macro is self-contained (none includes another). Helpers (`trim`,
  `sanitize_for_filename`, `parse_input_specs`, `validate_csv_labels`, the fit code)
  and constants (histogram ranges, `kResidualBins`, `kVertexPhiBins = 14`, fit
  thresholds) are copied between files. When you change one copy, check the others:
  - plot_flagged_overlays.C must use the same histogram ranges as
    plot_fits_overlaid_stavechip.C.
  - dca_fits CSV columns must match residual_fits.
  - The CSV columns are read by `stdiag` and by makeDcaZShiftedAlignment.C.
- Signs: `residual_vertexX/Y` hold -DCA, so `mu_mm` of `dca_z` is <-DCA_z>.
- Fit status: if more than half the entries fall outside the histogram range,
  `fit_status` is `out_of_range`. With fewer than 10 entries or 4 populated bins,
  there is no fit.
- Style:
  - two-space indentation, braces on the next line
  - file-local constants (`k` prefix), structs and helpers in an anonymous namespace
  - the entry-point function has the same name as the file
  - errors go to `std::cerr` as `ERROR: ...`, followed by `return`
  - header comments end with "Runs locally only; nothing is sent to any external
    service."
