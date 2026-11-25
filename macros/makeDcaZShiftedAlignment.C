#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// Builds localAlignmentParamsFile.txt variants with per-chip global z shifts
// of the MVTX, to test whether moving chips along z reduces the per-chip mean
// DCA_z measured by plot_fits_overlaid_dcadistsfromstavechips.C. INTT is
// never modified: its z resolution (strip length / sqrt(12) ~ 0.5 cm) gives
// it a negligible weight in the track z at the vertex.
//
// Alignment file format (AlignmentTransformation::createMap):
//   hitsetkey alpha beta gamma dx dy dz [dgrx dgry dgrz]
// Only column 7 (dz, global z translation in mm) of MVTX rows is changed;
// every other line is copied unchanged. The base file must be complete
// (all MVTX, INTT, TPC and TPOT rows), since the file replaces the CDB
// alignment.
//
// Sign convention: the CSV column mu_mm of coord dca_z is <-DCA_z>
// (MakeMilleFiles residual_vertexY, DCA = track PCA - vertex). Moving a chip
// by +dz moves the z at the vertex of its tracks by w_layer * dz, so its mu
// changes by -w_layer * dz.
//
// Null modes: every selected track has 3 MVTX clusters and the vertex is
// fitted from the same tracks, so a shift common to all chips of a layer is
// absorbed by the vertex and does not change DCA_z.
//
// Functions:
//   makeResponseAlignment (test C): even staves of one layer +shiftMm, odd
//     staves -shiftMm. The vertex shift cancels, and the shifted chips
//     respond with dmu = -/+ w * shiftMm.
//   compareDcaZ: per-layer mean and RMS of mu before/after, and with
//     responseLayer >= 0 the weight w = -(dmu_even - dmu_odd) / (2 shiftMm).
//   makeLeverArmAlignment (config B): dz = damping * (mu - <mu>_layer) / w
//     for every chip passing the quality cuts (one displaced layer).
//   makeDcaZMeanAlignment: dz = -scale * (<DCA_z> - <DCA_z>_layer)
//     = scale * (mu - <mu>_layer) for every chip of L0-L2 passing the quality
//     cuts (whole lines from the vertex displaced in z).
//   applyDzSolution: adds damping * dz of the joint per-track solution of
//     solveDcaZAlignment.C (dz_solution_<label>.csv), which accounts for the
//     lever arms of all chips a track crosses.
// With kPrintDzPerChip, the functions that write an alignment file also print
// dz (mm) of every MVTX chip of the shifted layers, layer by layer, ordered by
// stave-chip ID = 9*stave + chip; unshifted chips are listed with dz = 0 and
// the reason. The _dz_summary.csv has the same order and a stavechip column.
//
// Workflow (all commands from the directory holding this macro, or give its
// path in the .L line):
//   1) Test C, once per layer L = 0, 1, 2:
//        root -l -b -q -e '.L makeDcaZShiftedAlignment.C' \
//          -e 'makeResponseAlignment(0, 0.020, "base.txt", "responseL0.txt")'
//      copy responseL0.txt as localAlignmentParamsFile.txt into the run
//      directory of Fun4All_TrackReco_SiAlignment_ACTS_SiOnly_ReadPruned.C,
//      rerun it and plot_fits_overlaid_dcadistsfromstavechips.C, then
//        root -l -b -q -e '.L makeDcaZShiftedAlignment.C' \
//          -e 'compareDcaZ("dca_fits_Base.csv", "dca_fits_RespL0.csv", 0, 0.020)'
//   2) Config B with the measured weights w0, w1, w2:
//        root -l -b -q -e '.L makeDcaZShiftedAlignment.C' \
//          -e 'makeLeverArmAlignment("dca_fits_Base.csv", "base.txt", "iterB1.txt", w0, w1, w2)'
//      For the next iteration use iterB1.txt as the base file and the CSV
//      of the run made with it.
//   3) Mean-DCA_z shift (no weights needed):
//        root -l -b -q -e '.L makeDcaZShiftedAlignment.C' \
//          -e 'makeDcaZMeanAlignment("dca_fits_Base.csv", "base.txt", "meanDcaZ1.txt")'
//   4) Joint per-track solution (base.txt must be the alignment file the
//      ntuples were made with):
//        root -l -b -q 'solveDcaZAlignment.C+("list_Base.txt:Base", "dca_fits_Base.csv")'
//        root -l -b -q -e '.L makeDcaZShiftedAlignment.C' \
//          -e 'applyDzSolution("dz_solution_Base.csv", "base.txt", "solvedDz1.txt", 0.5)'
// Runs locally only; nothing is sent to any external service.

namespace
{
  // ===========================================================================
  // USER SETTINGS: default paths, used when an argument is not given.
  // Edit them here or pass the paths as function arguments (see above).
  // Relative paths are resolved from the directory ROOT is started in.
  // ===========================================================================
  // Complete alignment file the shifts are added to (e.g. the current
  // iteration's localAlignmentParamsFile.txt or a copy of the CDB file).
  const char* const kBaseAlignmentFile = "localAlignmentParamsFile_base.txt";
  // dca_fits_<label>.csv of plot_fits_overlaid_dcadistsfromstavechips.C.
  const char* const kDcaCsvFile = "dca_fits_Iter1.csv";
  // Output alignment file; copy it as localAlignmentParamsFile.txt into the
  // run directory of the reconstruction macro.
  const char* const kOutputAlignmentFile = "localAlignmentParamsFile_dzshift.txt";
  // dca_fits CSVs of the runs before and after the shift, for compareDcaZ.
  const char* const kBeforeDcaCsvFile = "dca_fits_Base.csv";
  const char* const kAfterDcaCsvFile = "dca_fits_After.csv";
  // dz_solution_<label>.csv of solveDcaZAlignment.C, for applyDzSolution.
  const char* const kDzSolutionCsvFile = "dz_solution_Iter1.csv";
  // Print dz of every MVTX chip of the shifted layers versus stave-chip ID.
  constexpr bool kPrintDzPerChip = true;
  // ===========================================================================

  constexpr int kMvtxLayers = 3;
  constexpr int kExpectedMvtxRows = 432;  // 48 staves x 9 chips
  // Geometry lever arms of the MVTX layers in the track z at the vertex:
  // w_i = 1/3 - rbar (r_i - rbar) / sum_j (r_j - rbar)^2 of a straight-line
  // fit in (r, z) through equally precise points extrapolated to r = 0, at the
  // stave rMid of PHG4MvtxDefs.h (25.23, 33.36, 41.48 mm). Over the chip
  // radii rMin..rMax they span 2.39-2.56, 0.30-0.38 and -1.72 to -1.87.
  // Scattering and the vertex fit are not included; replace them with the
  // weights measured by makeResponseAlignment + compareDcaZ.
  constexpr double kEstimatedWeights[kMvtxLayers] = {2.39, 0.33, -1.72};

  // (layer, stave, chip)
  using ChipId = std::tuple<int, int, int>;

  struct AlignmentLine
  {
    std::string text;
    std::vector<std::string> tokens;
    bool isData = false;
  };

  struct DcaFit
  {
    int entries = 0;
    std::string status;
    double mu = std::numeric_limits<double>::quiet_NaN();
    double muErr = std::numeric_limits<double>::quiet_NaN();
  };

  // even staves +shift, odd staves -shift (makeResponseAlignment and
  // compareDcaZ must agree)
  int response_sign(int stave)
  {
    return stave % 2 == 0 ? 1 : -1;
  }

  std::string format_value(double value)
  {
    std::ostringstream stream;
    stream << std::setprecision(10) << value;
    return stream.str();
  }

  // Bit layout from TrkrDefs.cc (trkrId << 24, layer << 16) and MvtxDefs.cc
  // (stave << 9, chip << 5, strobe << 0). Returns false for non-MVTX keys.
  bool decode_mvtx_key(const std::string& token, ChipId& id)
  {
    char* end = nullptr;
    const unsigned long long value = std::strtoull(token.c_str(), &end, 10);
    if(end == token.c_str() || *end != '\0' || value > UINT32_MAX)
    {
      return false;
    }
    const uint32_t key = static_cast<uint32_t>(value);
    if((key >> 24) != 0)
    {
      return false;
    }
    id = ChipId((key >> 16) & 0xFF, (key >> 9) & 0x7F, (key >> 5) & 0xF);
    return true;
  }

  // Reads the alignment file with the parsing rules of
  // AlignmentTransformation::createMap (leading spaces trimmed, empty, "#"
  // and "//" lines skipped, 7 or >= 10 values per data line).
  bool read_alignment_file(const std::string& path, std::vector<AlignmentLine>& lines)
  {
    std::ifstream input(path);
    if(!input.is_open())
    {
      std::cerr << "ERROR: could not open alignment file " << path << std::endl;
      return false;
    }

    std::string text;
    int lineNumber = 0;
    while(std::getline(input, text))
    {
      ++lineNumber;
      if(!text.empty() && text.back() == '\r')
      {
        text.pop_back();
      }

      AlignmentLine line;
      line.text = text;
      const size_t first = text.find_first_not_of(" \t");
      const std::string trimmed = first == std::string::npos ? "" : text.substr(first);
      if(trimmed.empty() || trimmed.rfind("#", 0) == 0 || trimmed.rfind("//", 0) == 0)
      {
        lines.push_back(line);
        continue;
      }

      std::istringstream stream(trimmed);
      std::string token;
      while(stream >> token)
      {
        line.tokens.push_back(token);
      }
      if(line.tokens.size() < 7 || line.tokens.size() == 8 || line.tokens.size() == 9)
      {
        std::cerr << "ERROR: " << path << ":" << lineNumber << " has "
                  << line.tokens.size() << " values; expected 7 or 10:\n  "
                  << text << std::endl;
        return false;
      }
      line.isData = true;
      lines.push_back(line);
    }

    return true;
  }

  // Adds dzByChip to column 7 of the MVTX rows and writes outPath. Fails if
  // an MVTX chip appears twice or a chip of dzByChip is not in the file.
  bool shift_and_write(
    const std::vector<AlignmentLine>& lines,
    const std::map<ChipId, double>& dzByChip,
    const std::string& baseFile,
    const std::string& outPath)
  {
    if(outPath == baseFile)
    {
      std::cerr << "ERROR: output file equals the base file " << baseFile
                << "; choose another output name." << std::endl;
      return false;
    }

    std::set<ChipId> seen;
    std::vector<std::string> output;
    output.reserve(lines.size() + 1);
    int mvtxRows = 0;
    int shiftedRows = 0;
    for(const AlignmentLine& line : lines)
    {
      ChipId id;
      if(!line.isData || !decode_mvtx_key(line.tokens[0], id))
      {
        output.push_back(line.text);
        continue;
      }

      ++mvtxRows;
      if(!seen.insert(id).second)
      {
        std::cerr << "ERROR: MVTX layer " << std::get<0>(id) << " stave "
                  << std::get<1>(id) << " chip " << std::get<2>(id)
                  << " appears twice in " << baseFile << std::endl;
        return false;
      }

      const auto shift = dzByChip.find(id);
      if(shift == dzByChip.end())
      {
        output.push_back(line.text);
        continue;
      }

      std::vector<std::string> tokens = line.tokens;
      tokens[6] = format_value(std::stod(tokens[6]) + shift->second);
      std::string text = tokens[0];
      for(size_t i = 1; i < tokens.size(); ++i)
      {
        text += "  " + tokens[i];
      }
      output.push_back(text);
      ++shiftedRows;
    }

    for(const auto& [id, dz] : dzByChip)
    {
      if(!seen.count(id))
      {
        std::cerr << "ERROR: MVTX layer " << std::get<0>(id) << " stave "
                  << std::get<1>(id) << " chip " << std::get<2>(id)
                  << " (dz = " << dz << " mm) is not in " << baseFile
                  << std::endl;
        return false;
      }
    }

    if(mvtxRows != kExpectedMvtxRows)
    {
      std::cout << "WARNING: " << baseFile << " has " << mvtxRows
                << " MVTX rows, expected " << kExpectedMvtxRows
                << "; check that it is a complete alignment file." << std::endl;
    }

    std::ofstream out(outPath);
    if(!out.is_open())
    {
      std::cerr << "ERROR: could not write " << outPath << std::endl;
      return false;
    }
    for(const std::string& text : output)
    {
      out << text << '\n';
    }

    std::cout << "Wrote " << outPath << ": " << lines.size() << " lines, "
              << shiftedRows << " of " << mvtxRows << " MVTX rows shifted in z."
              << std::endl;
    return true;
  }

  std::vector<std::string> split_csv(const std::string& text)
  {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream stream(text);
    while(std::getline(stream, field, ','))
    {
      fields.push_back(field);
    }
    if(!text.empty() && text.back() == ',')
    {
      fields.emplace_back();
    }
    return fields;
  }

  // Reads the MVTX dca_z rows of a dca_fits_<label>.csv. With an empty
  // labelFilter the file must hold a single label.
  bool read_dca_csv(
    const std::string& path,
    const std::string& labelFilter,
    std::map<ChipId, DcaFit>& fits)
  {
    std::ifstream input(path);
    if(!input.is_open())
    {
      std::cerr << "ERROR: could not open DCA CSV " << path << std::endl;
      return false;
    }

    std::string text;
    if(!std::getline(input, text))
    {
      std::cerr << "ERROR: " << path << " is empty" << std::endl;
      return false;
    }
    const std::vector<std::string> header = split_csv(text);
    std::map<std::string, size_t> column;
    for(size_t i = 0; i < header.size(); ++i)
    {
      column[header[i]] = i;
    }
    for(const char* name : {"label", "detector", "layer", "stave", "chip", "coord",
                            "entries", "fit_status", "mu_mm", "mu_err_mm"})
    {
      if(!column.count(name))
      {
        std::cerr << "ERROR: column " << name << " missing in " << path << std::endl;
        return false;
      }
    }

    std::set<std::string> labels;
    while(std::getline(input, text))
    {
      if(!text.empty() && text.back() == '\r')
      {
        text.pop_back();
      }
      const std::vector<std::string> fields = split_csv(text);
      if(fields.size() != header.size())
      {
        continue;
      }
      if(fields[column["detector"]] != "MVTX" || fields[column["coord"]] != "dca_z")
      {
        continue;
      }
      const std::string& label = fields[column["label"]];
      if(!labelFilter.empty() && label != labelFilter)
      {
        continue;
      }
      labels.insert(label);

      const ChipId id(std::stoi(fields[column["layer"]]),
                      std::stoi(fields[column["stave"]]),
                      std::stoi(fields[column["chip"]]));
      DcaFit fit;
      fit.entries = static_cast<int>(std::stod(fields[column["entries"]]));
      fit.status = fields[column["fit_status"]];
      fit.mu = std::strtod(fields[column["mu_mm"]].c_str(), nullptr);
      fit.muErr = std::strtod(fields[column["mu_err_mm"]].c_str(), nullptr);
      fits[id] = fit;
    }

    if(labels.size() > 1)
    {
      std::cerr << "ERROR: " << path << " holds " << labels.size()
                << " labels; choose one with labelFilter." << std::endl;
      return false;
    }
    if(fits.empty())
    {
      std::cerr << "ERROR: no MVTX dca_z rows read from " << path << std::endl;
      return false;
    }
    std::cout << "Read " << fits.size() << " MVTX dca_z fits from " << path << std::endl;
    return true;
  }

  std::string summary_path(const std::string& outFile)
  {
    const size_t dot = outFile.rfind('.');
    const size_t slash = outFile.rfind('/');
    const bool hasExtension = dot != std::string::npos &&
                              (slash == std::string::npos || dot > slash);
    return (hasExtension ? outFile.substr(0, dot) : outFile) + "_dz_summary.csv";
  }

  bool good_fit(const DcaFit& fit, int minEntries)
  {
    return fit.status == "ok" && fit.entries >= minEntries &&
           std::isfinite(fit.mu) && std::isfinite(fit.muErr);
  }

  struct Stats
  {
    int n = 0;
    double mean = 0;
    double rms = 0;  // about the mean
    double meanError = 0;
  };

  Stats get_stats(const std::vector<double>& values)
  {
    Stats stats;
    stats.n = static_cast<int>(values.size());
    if(stats.n == 0)
    {
      return stats;
    }
    double sum = 0;
    for(double value : values)
    {
      sum += value;
    }
    stats.mean = sum / stats.n;
    double sum2 = 0;
    for(double value : values)
    {
      sum2 += (value - stats.mean) * (value - stats.mean);
    }
    stats.rms = std::sqrt(sum2 / stats.n);
    stats.meanError = stats.n > 1 ? std::sqrt(sum2 / (stats.n - 1) / stats.n) : 0;
    return stats;
  }

  // MVTX stave-chip ID, the x axis of the summary pages of
  // plot_fits_overlaid_dcadistsfromstavechips.C.
  int stavechip_id(const ChipId& id)
  {
    return 9 * std::get<1>(id) + std::get<2>(id);
  }

  // Prints the dz of every MVTX chip of the layers with printLayer, ordered
  // by stave-chip ID (stave 0 chip 0, stave 0 chip 1, ..., stave 1 chip 0,
  // ...). Chips without a shift are listed with dz = 0 and the reason
  // (not_in_dca_csv if they have none).
  void print_dz_by_stavechip(
    const std::vector<AlignmentLine>& lines,
    const std::map<ChipId, double>& dzByChip,
    const bool (&printLayer)[kMvtxLayers],
    const std::map<ChipId, std::string>& reasonByChip)
  {
    // std::map orders ChipId by layer, then stave, then chip.
    std::map<ChipId, double> dzOfAllChips;
    for(const AlignmentLine& line : lines)
    {
      ChipId id;
      if(line.isData && decode_mvtx_key(line.tokens[0], id))
      {
        const auto shift = dzByChip.find(id);
        dzOfAllChips[id] = shift == dzByChip.end() ? 0.0 : shift->second;
      }
    }

    const std::streamsize oldPrecision = std::cout.precision();
    for(int layer = 0; layer < kMvtxLayers; ++layer)
    {
      if(!printLayer[layer])
      {
        continue;
      }
      int chips = 0;
      int shifted = 0;
      for(const auto& [id, dz] : dzOfAllChips)
      {
        if(std::get<0>(id) == layer)
        {
          ++chips;
          shifted += dzByChip.count(id) ? 1 : 0;
        }
      }
      std::cout << "\nMVTX layer " << layer
                << ": dz vs stave-chip ID = 9*stave + chip (" << chips
                << " chips, " << shifted << " shifted)\n"
                << "  stavechip  stave  chip       dz_mm  note\n";
      for(const auto& [id, dz] : dzOfAllChips)
      {
        if(std::get<0>(id) != layer)
        {
          continue;
        }
        std::string note;
        if(!dzByChip.count(id))
        {
          const auto reason = reasonByChip.find(id);
          note = reason == reasonByChip.end() ? "not_in_dca_csv" : reason->second;
        }
        std::cout << std::setw(11) << stavechip_id(id) << std::setw(7)
                  << std::get<1>(id) << std::setw(6) << std::get<2>(id)
                  << std::fixed << std::setprecision(5) << std::showpos
                  << std::setw(12) << dz << std::noshowpos << "  " << note
                  << '\n';
      }
    }
    std::cout << std::defaultfloat << std::setprecision(oldPrecision) << std::flush;
  }
}

// Test C: even staves of `layer` +shiftMm, odd staves -shiftMm (global z,
// mm). Writes outFile and <outFile without extension>_dz_summary.csv.
void makeResponseAlignment(
  int layer = 0,
  double shiftMm = 0.020,
  const char* baseFile = kBaseAlignmentFile,
  const char* outFile = kOutputAlignmentFile)
{
  if(layer < 0 || layer >= kMvtxLayers)
  {
    std::cerr << "ERROR: layer must be 0, 1 or 2 (MVTX)" << std::endl;
    return;
  }

  std::vector<AlignmentLine> lines;
  if(!read_alignment_file(baseFile, lines))
  {
    return;
  }

  std::map<ChipId, double> dzByChip;
  for(const AlignmentLine& line : lines)
  {
    ChipId id;
    if(line.isData && decode_mvtx_key(line.tokens[0], id) && std::get<0>(id) == layer)
    {
      dzByChip[id] = response_sign(std::get<1>(id)) * shiftMm;
    }
  }
  if(dzByChip.empty())
  {
    std::cerr << "ERROR: no MVTX layer " << layer << " rows in " << baseFile << std::endl;
    return;
  }

  if(!shift_and_write(lines, dzByChip, baseFile, outFile))
  {
    return;
  }
  if(kPrintDzPerChip)
  {
    bool printLayer[kMvtxLayers] = {false, false, false};
    printLayer[layer] = true;
    print_dz_by_stavechip(lines, dzByChip, printLayer, {});
  }

  const std::string summaryFile = summary_path(outFile);
  std::ofstream summary(summaryFile);
  summary << std::setprecision(10) << "layer,stave,chip,stavechip,dz_mm\n";
  for(const auto& [id, dz] : dzByChip)
  {
    summary << std::get<0>(id) << ',' << std::get<1>(id) << ',' << std::get<2>(id)
            << ',' << stavechip_id(id) << ',' << dz << '\n';
  }
  std::cout << "Response test: MVTX layer " << layer << ", even staves "
            << shiftMm << " mm, odd staves " << -shiftMm << " mm ("
            << dzByChip.size() << " chips). Summary: " << summaryFile << std::endl;
}

// Per-layer mean and RMS (about the layer mean) of the MVTX mu of -DCA_z for
// the chips with fit_status ok and >= minEntries in both CSVs. With
// responseLayer >= 0, also the weight of that layer from a
// makeResponseAlignment run with the same shiftMm.
void compareDcaZ(
  const char* beforeCsv = kBeforeDcaCsvFile,
  const char* afterCsv = kAfterDcaCsvFile,
  int responseLayer = -1,
  double shiftMm = 0.020,
  int minEntries = 30,
  const char* beforeLabel = "",
  const char* afterLabel = "")
{
  std::map<ChipId, DcaFit> before;
  std::map<ChipId, DcaFit> after;
  if(!read_dca_csv(beforeCsv, beforeLabel, before) ||
     !read_dca_csv(afterCsv, afterLabel, after))
  {
    return;
  }

  std::vector<double> muBefore[kMvtxLayers];
  std::vector<double> muAfter[kMvtxLayers];
  std::vector<double> deltaEven;
  std::vector<double> deltaOdd;
  for(const auto& [id, fitBefore] : before)
  {
    const auto fitAfter = after.find(id);
    const int layer = std::get<0>(id);
    if(fitAfter == after.end() || layer < 0 || layer >= kMvtxLayers ||
       !good_fit(fitBefore, minEntries) || !good_fit(fitAfter->second, minEntries))
    {
      continue;
    }
    muBefore[layer].push_back(fitBefore.mu);
    muAfter[layer].push_back(fitAfter->second.mu);
    if(layer == responseLayer)
    {
      const double delta = fitAfter->second.mu - fitBefore.mu;
      (response_sign(std::get<1>(id)) > 0 ? deltaEven : deltaOdd).push_back(delta);
    }
  }

  std::cout << std::fixed << std::setprecision(2)
            << "\nmu of -DCA_z per MVTX chip, in um (chips good in both inputs)\n"
            << "layer  chips   mean_before  rms_before   mean_after  rms_after\n";
  for(int layer = 0; layer < kMvtxLayers; ++layer)
  {
    const Stats statsBefore = get_stats(muBefore[layer]);
    const Stats statsAfter = get_stats(muAfter[layer]);
    std::cout << std::setw(5) << layer << std::setw(7) << statsBefore.n
              << std::setw(14) << statsBefore.mean * 1e3
              << std::setw(12) << statsBefore.rms * 1e3
              << std::setw(13) << statsAfter.mean * 1e3
              << std::setw(11) << statsAfter.rms * 1e3 << '\n';
  }
  std::cout << "rms is about the layer mean; the layer mean itself is not "
               "constrained by DCA_z (absorbed by the vertex).\n";

  if(responseLayer >= 0)
  {
    const Stats even = get_stats(deltaEven);
    const Stats odd = get_stats(deltaOdd);
    if(even.n < 2 || odd.n < 2 || shiftMm == 0)
    {
      std::cerr << "ERROR: not enough good chips in layer " << responseLayer
                << " to measure the weight" << std::endl;
      return;
    }
    const double weight = -(even.mean - odd.mean) / (2 * shiftMm);
    const double weightError =
      std::hypot(even.meanError, odd.meanError) / (2 * std::fabs(shiftMm));
    std::cout << "\nResponse of layer " << responseLayer << " to +/-"
              << shiftMm * 1e3 << " um (even/odd staves):\n"
              << "  dmu even staves: " << even.mean * 1e3 << " +- "
              << even.meanError * 1e3 << " um (" << even.n << " chips)\n"
              << "  dmu odd staves:  " << odd.mean * 1e3 << " +- "
              << odd.meanError * 1e3 << " um (" << odd.n << " chips)\n"
              << std::setprecision(3)
              << "  weight w" << responseLayer << " = " << weight << " +- "
              << weightError << "   (estimate used so far: "
              << kEstimatedWeights[responseLayer] << ")\n";
  }
  std::cout << std::defaultfloat << std::flush;
}

namespace
{
  // Sets shiftLayer from a string of MVTX layer digits, e.g. "02".
  bool parse_layers(const char* layersToShift, bool (&shiftLayer)[kMvtxLayers])
  {
    for(const char* c = layersToShift; *c; ++c)
    {
      const int layer = *c - '0';
      if(layer < 0 || layer >= kMvtxLayers)
      {
        std::cerr << "ERROR: layersToShift must only contain 0, 1, 2" << std::endl;
        return false;
      }
      shiftLayer[layer] = true;
    }
    return true;
  }

  // Shared by makeLeverArmAlignment and makeDcaZMeanAlignment:
  // dz = scale * (mu - <mu>_layer) / weights[layer] for the MVTX chips of the
  // layers with shiftLayer passing the cuts; other chips keep dz = 0. Writes
  // outFile and <outFile without extension>_dz_summary.csv.
  void shift_by_dca_fits(
    const char* modeName,
    const char* dcaCsv,
    const char* baseFile,
    const char* outFile,
    const double (&weights)[kMvtxLayers],
    double scale,
    const bool (&shiftLayer)[kMvtxLayers],
    bool subtractLayerMean,
    int minEntries,
    double maxMuErrMm,
    double maxAbsMuMm,
    double maxAbsDzMm,
    const char* labelFilter)
  {
    std::map<ChipId, DcaFit> fits;
    std::vector<AlignmentLine> lines;
    if(!read_dca_csv(dcaCsv, labelFilter, fits) || !read_alignment_file(baseFile, lines))
    {
      return;
    }

    auto passes_cuts = [&](const DcaFit& fit)
    {
      return good_fit(fit, minEntries) && fit.muErr <= maxMuErrMm &&
             std::fabs(fit.mu) <= maxAbsMuMm;
    };

    double layerMean[kMvtxLayers] = {0, 0, 0};
    if(subtractLayerMean)
    {
      std::vector<double> values[kMvtxLayers];
      for(const auto& [id, fit] : fits)
      {
        const int layer = std::get<0>(id);
        if(layer >= 0 && layer < kMvtxLayers && passes_cuts(fit))
        {
          values[layer].push_back(fit.mu);
        }
      }
      for(int layer = 0; layer < kMvtxLayers; ++layer)
      {
        layerMean[layer] = get_stats(values[layer]).mean;
      }
    }

    std::map<ChipId, double> dzByChip;
    std::map<ChipId, std::string> reasonByChip;
    const std::string summaryFile = summary_path(outFile);
    std::ostringstream summary;
    summary << std::setprecision(10)
            << "layer,stave,chip,stavechip,entries,fit_status,mu_mm,mu_err_mm,"
               "layer_mean_mm,weight,dz_mm,applied,reason\n";
    int skipped = 0;
    for(const auto& [id, fit] : fits)
    {
      const int layer = std::get<0>(id);
      if(layer < 0 || layer >= kMvtxLayers)
      {
        continue;
      }

      double dz = scale * (fit.mu - layerMean[layer]) / weights[layer];
      std::string reason = "ok";
      if(!shiftLayer[layer])
      {
        reason = "layer_not_shifted";
      }
      else if(!good_fit(fit, minEntries))
      {
        reason = "bad_fit_or_low_stats";
      }
      else if(fit.muErr > maxMuErrMm)
      {
        reason = "mu_err_too_large";
      }
      else if(std::fabs(fit.mu) > maxAbsMuMm)
      {
        reason = "mu_too_large";
      }
      else if(std::fabs(dz) > maxAbsDzMm)
      {
        reason = "dz_too_large";
      }

      const bool applied = reason == "ok";
      reasonByChip[id] = reason;
      if(applied)
      {
        dzByChip[id] = dz;
      }
      else
      {
        dz = 0;
        skipped += shiftLayer[layer] ? 1 : 0;
      }
      summary << layer << ',' << std::get<1>(id) << ',' << std::get<2>(id) << ','
              << stavechip_id(id) << ',' << fit.entries << ',' << fit.status << ',' << fit.mu << ','
              << fit.muErr << ',' << layerMean[layer] << ',' << weights[layer]
              << ',' << dz << ',' << (applied ? 1 : 0) << ',' << reason << '\n';
    }

    if(!shift_and_write(lines, dzByChip, baseFile, outFile))
    {
      return;
    }
    if(kPrintDzPerChip)
    {
      print_dz_by_stavechip(lines, dzByChip, shiftLayer, reasonByChip);
    }

    std::ofstream summaryOut(summaryFile);
    summaryOut << summary.str();
    std::cout << std::setprecision(4) << modeName << ": scale " << scale
              << ", weights " << weights[0] << " " << weights[1] << " " << weights[2]
              << ", layer means of mu (um) " << layerMean[0] * 1e3 << " "
              << layerMean[1] * 1e3 << " " << layerMean[2] * 1e3 << "\n  "
              << dzByChip.size() << " chips shifted, " << skipped
              << " chips of shifted layers kept at dz = 0 (see " << summaryFile
              << ")" << std::defaultfloat << std::endl;
  }
}

// Config B: dz = damping * (mu - <mu>_layer) / w_layer for the MVTX chips of
// layersToShift (e.g. "02") passing the cuts; other chips keep dz = 0.
// Layers with |w| < minAbsWeight are skipped (noise would be amplified by
// 1/w). Writes outFile and <outFile without extension>_dz_summary.csv.
void makeLeverArmAlignment(
  const char* dcaCsv = kDcaCsvFile,
  const char* baseFile = kBaseAlignmentFile,
  const char* outFile = kOutputAlignmentFile,
  double w0 = kEstimatedWeights[0],
  double w1 = kEstimatedWeights[1],
  double w2 = kEstimatedWeights[2],
  double damping = 0.33,
  const char* layersToShift = "012",
  bool subtractLayerMean = true,
  double minAbsWeight = 0.5,
  int minEntries = 30,
  double maxMuErrMm = 0.005,
  double maxAbsMuMm = 0.1,
  double maxAbsDzMm = 0.1,
  const char* labelFilter = "")
{
  const double weights[kMvtxLayers] = {w0, w1, w2};
  if(w0 == kEstimatedWeights[0] && w1 == kEstimatedWeights[1] && w2 == kEstimatedWeights[2])
  {
    std::cout << "WARNING: using the estimated weights; replace them with the "
                 "ones measured by makeResponseAlignment + compareDcaZ." << std::endl;
  }

  bool shiftLayer[kMvtxLayers] = {false, false, false};
  if(!parse_layers(layersToShift, shiftLayer))
  {
    return;
  }
  for(int layer = 0; layer < kMvtxLayers; ++layer)
  {
    if(shiftLayer[layer] && std::fabs(weights[layer]) < minAbsWeight)
    {
      std::cout << "WARNING: layer " << layer << " skipped, |w" << layer << "| = "
                << std::fabs(weights[layer]) << " < minAbsWeight = "
                << minAbsWeight << std::endl;
      shiftLayer[layer] = false;
    }
  }

  shift_by_dca_fits(
    "Lever-arm shift", dcaCsv, baseFile, outFile, weights, damping, shiftLayer,
    subtractLayerMean, minEntries, maxMuErrMm, maxAbsMuMm, maxAbsDzMm,
    labelFilter);
}

// Mean-DCA_z shift: every MVTX chip of layersToShift passing the cuts is
// moved by dz = -scale * (<DCA_z> - <DCA_z>_layer) = scale * (mu - <mu>_layer)
// along global z, i.e. by minus the mean DCA_z of the tracks through it
// relative to its layer. This is the correction for a whole line from the
// vertex (one chip per layer) displaced in z: the lever arms sum to 1, so
// moving the three chips of the line by -DCA_z removes it once, with no
// damping needed (scale = 1). It is not the correction for a single
// displaced layer (use makeLeverArmAlignment); the z residual / DCA_z pages
// of plot_fits_overlaid_dcadistsfromstavechips.C tell the two apart. The
// layer mean is removed because the vertex absorbs a shift common to a
// layer. Writes outFile and <outFile without extension>_dz_summary.csv.
void makeDcaZMeanAlignment(
  const char* dcaCsv = kDcaCsvFile,
  const char* baseFile = kBaseAlignmentFile,
  const char* outFile = kOutputAlignmentFile,
  double scale = 1.0,
  const char* layersToShift = "012",
  bool subtractLayerMean = true,
  int minEntries = 30,
  double maxMuErrMm = 0.005,
  double maxAbsMuMm = 0.1,
  double maxAbsDzMm = 0.1,
  const char* labelFilter = "")
{
  const double weights[kMvtxLayers] = {1.0, 1.0, 1.0};
  bool shiftLayer[kMvtxLayers] = {false, false, false};
  if(!parse_layers(layersToShift, shiftLayer))
  {
    return;
  }

  shift_by_dca_fits(
    "Mean-DCA_z shift", dcaCsv, baseFile, outFile, weights, scale, shiftLayer,
    subtractLayerMean, minEntries, maxMuErrMm, maxAbsMuMm, maxAbsDzMm,
    labelFilter);
}

// Adds damping * dz_mm of every chip with free = 1 in solutionCsv
// (dz_solution_<label>.csv of solveDcaZAlignment.C) to column 7 of the MVTX
// rows of baseFile; chips with free = 0 keep dz = 0. The solution is joint
// (the dz of a chip depends on all others), so if any |damping * dz_mm| >
// maxAbsDzMm nothing is written. baseFile must be the alignment file the
// ntuples of the solution were made with. Writes outFile and <outFile
// without extension>_dz_summary.csv.
void applyDzSolution(
  const char* solutionCsv = kDzSolutionCsvFile,
  const char* baseFile = kBaseAlignmentFile,
  const char* outFile = kOutputAlignmentFile,
  double damping = 1.0,
  double maxAbsDzMm = 0.2)
{
  std::ifstream input(solutionCsv);
  std::string text;
  if(!input.is_open() || !std::getline(input, text))
  {
    std::cerr << "ERROR: could not read " << solutionCsv << std::endl;
    return;
  }
  const std::vector<std::string> header = split_csv(text);
  std::map<std::string, size_t> column;
  for(size_t i = 0; i < header.size(); ++i)
  {
    column[header[i]] = i;
  }
  for(const char* name : {"layer", "stave", "chip", "n_tracks", "free", "dz_mm", "dz_err_mm"})
  {
    if(!column.count(name))
    {
      std::cerr << "ERROR: column " << name << " missing in " << solutionCsv << std::endl;
      return;
    }
  }

  struct SolutionRow
  {
    int nTracks = 0;
    bool free = false;
    double dz = 0;
    double dzError = 0;
  };
  std::map<ChipId, SolutionRow> rows;
  while(std::getline(input, text))
  {
    if(!text.empty() && text.back() == '\r')
    {
      text.pop_back();
    }
    const std::vector<std::string> fields = split_csv(text);
    if(fields.size() != header.size())
    {
      continue;
    }
    const ChipId id(std::stoi(fields[column["layer"]]), std::stoi(fields[column["stave"]]),
                    std::stoi(fields[column["chip"]]));
    SolutionRow row;
    row.nTracks = std::stoi(fields[column["n_tracks"]]);
    row.free = std::stoi(fields[column["free"]]) == 1;
    row.dz = std::strtod(fields[column["dz_mm"]].c_str(), nullptr);
    row.dzError = std::strtod(fields[column["dz_err_mm"]].c_str(), nullptr);
    rows[id] = row;
  }
  if(rows.empty())
  {
    std::cerr << "ERROR: no chip rows read from " << solutionCsv << std::endl;
    return;
  }

  std::map<ChipId, double> dzByChip;
  std::map<ChipId, std::string> reasonByChip;
  int tooLarge = 0;
  for(const auto& [id, row] : rows)
  {
    if(!row.free)
    {
      reasonByChip[id] = "fixed_low_stats";
      continue;
    }
    const double dz = damping * row.dz;
    if(!std::isfinite(dz) || std::fabs(dz) > maxAbsDzMm)
    {
      if(tooLarge < 5)
      {
        std::cerr << "ERROR: MVTX layer " << std::get<0>(id) << " stave " << std::get<1>(id)
                  << " chip " << std::get<2>(id) << ": damping * dz = " << dz
                  << " mm exceeds maxAbsDzMm = " << maxAbsDzMm << std::endl;
      }
      ++tooLarge;
      continue;
    }
    dzByChip[id] = dz;
    reasonByChip[id] = "ok";
  }
  if(tooLarge > 0)
  {
    std::cerr << "ERROR: " << tooLarge << " chips exceed maxAbsDzMm; the solution is "
                 "joint, so nothing was written. Check the solution or lower damping."
              << std::endl;
    return;
  }

  std::vector<AlignmentLine> lines;
  if(!read_alignment_file(baseFile, lines) ||
     !shift_and_write(lines, dzByChip, baseFile, outFile))
  {
    return;
  }
  if(kPrintDzPerChip)
  {
    const bool printLayer[kMvtxLayers] = {true, true, true};
    print_dz_by_stavechip(lines, dzByChip, printLayer, reasonByChip);
  }

  const std::string summaryFile = summary_path(outFile);
  std::ofstream summary(summaryFile);
  summary << std::setprecision(10)
          << "layer,stave,chip,stavechip,n_tracks,free,dz_solution_mm,dz_err_mm,"
             "damping,dz_mm,applied,reason\n";
  for(const auto& [id, row] : rows)
  {
    const auto shift = dzByChip.find(id);
    const bool applied = shift != dzByChip.end();
    summary << std::get<0>(id) << ',' << std::get<1>(id) << ',' << std::get<2>(id) << ','
            << stavechip_id(id) << ',' << row.nTracks << ',' << (row.free ? 1 : 0) << ','
            << row.dz << ',' << row.dzError << ',' << damping << ','
            << (applied ? shift->second : 0.0) << ',' << (applied ? 1 : 0) << ','
            << reasonByChip[id] << '\n';
  }
  std::cout << "Applied " << solutionCsv << " with damping " << damping << ": "
            << dzByChip.size() << " chips shifted (summary: " << summaryFile
            << "). " << baseFile << " must be the alignment file the ntuples of "
               "the solution were made with." << std::endl;
}

void makeDcaZShiftedAlignment()
{
  std::cout << "Load this file and call one of:\n"
               "  makeResponseAlignment(layer, shiftMm, baseFile, outFile)\n"
               "  compareDcaZ(beforeCsv, afterCsv, responseLayer, shiftMm)\n"
               "  makeLeverArmAlignment(dcaCsv, baseFile, outFile, w0, w1, w2, damping, layersToShift)\n"
               "  makeDcaZMeanAlignment(dcaCsv, baseFile, outFile, scale, layersToShift)\n"
               "  applyDzSolution(solutionCsv, baseFile, outFile, damping, maxAbsDzMm)\n"
               "e.g. root -l -b -q -e '.L makeDcaZShiftedAlignment.C' -e 'makeResponseAlignment(0, 0.020, \"base.txt\", \"responseL0.txt\")'\n"
               "Default paths: USER SETTINGS block at the top of this file."
            << std::endl;
}
