#include <TAxis.h>
#include <TCanvas.h>
#include <TChain.h>
#include <TColor.h>
#include <TF1.h>
#include <TGraphErrors.h>
#include <TH1D.h>
#include <TLegend.h>
#include <TLatex.h>
#include <TLine.h>
#include <TMath.h>
#include <TROOT.h>
#include <TString.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TPad.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// Track-to-vertex residuals per MVTX chip and INTT sensor: every ntp row
// (one cluster of one track) fills the -DCA_xy and -DCA_z of its track into
// the histograms of the chip the cluster is on, so each chip shows the DCA
// distribution of the tracks passing through it. Companion of
// plot_fits_overlaid_stavechip.C (same inputs, fits, plot layout and CSV
// columns); the DCA is the one MakeMilleFiles stores in track_ntp
// (residual_vertexX/Y = -DCA_xy/-DCA_z in mm, computed with the vertex of the
// track's own crossing), copied onto every ntp row of the track. Writes
//   - one CSV table per input file list (writeToCsv);
//   - a PDF (plus PNGs with savePngs) with two summary pages per layer (Gaussian mu and
//     sigma of every chip versus its stave-chip ID and versus its cluster
//     phi, drawStaveChipSummary), the MVTX z residual / DCA_z pages
//     (drawResidualRatio) and one page per stave and coordinate (makePlots).
// The MVTX z residual (ntp residualY) of the same rows is fitted per chip as
// in plot_fits_overlaid_stavechip.C; it is only plotted, not written to CSV.
// Runs locally only; nothing is sent to any external service.

namespace
{
  constexpr int kResidualBins = 25;
  constexpr double kMinimumEntriesForFit = 10.0;
  constexpr int kMinimumPopulatedBinsForFit = 4;
  constexpr int kCoreFitIterations = 2;
  constexpr double kCoreFitSigmaWindow = 2.0;
  // More than this fraction of the entries in the under/overflow gives
  // fit_status out_of_range instead of a fit.
  constexpr double kMaximumOutOfRangeFraction = 0.5;
  constexpr int kInputColors[] = {
    kBlack, kRed + 1, kBlue + 1, kGreen + 2,
    kMagenta + 1, kOrange + 7, kCyan + 2, kViolet + 1};
  constexpr int kInputMarkerStyles[] = {20, 24, 21, 25, 22, 26, 23, 32};

  // -DCA ranges as on the track-to-vertex page of
  // plot_fits_overlaid_stavechip.C: displayed over +-kDcaDisplayRangeMm,
  // filled over +-kDcaHistogramRangeMm with kResidualBins bins across the
  // display range; kDcaFitRangeMm is the half-width of the Gaussian seed fit.
  constexpr double kDcaDisplayRangeMm = 0.15;
  constexpr double kDcaHistogramRangeMm = 2.0;
  constexpr double kDcaFitRangeMm = 0.1;
  // SvtxTrack crossing of tracks whose bunch crossing is unknown
  // (SHRT_MAX); PHSimpleVertexFinder vertexes them as one group.
  constexpr double kUnknownCrossing = std::numeric_limits<short>::max();

  // MVTX z residuals (ntp residualY: cluster - smoothed track state in the
  // local frame, mm) binned and fitted as the MVTX z residuals of
  // plot_fits_overlaid_stavechip.C: kResidualZBins bins across +-0.04 mm,
  // filled over +-0.5 mm, Gaussian seed half-width 0.015 mm.
  constexpr int kResidualZBins = 200;
  constexpr double kResidualZDisplayRangeMm = 0.04;
  constexpr double kResidualZHistogramRangeMm = 0.5;
  constexpr double kResidualZFitRangeMm = 0.015;
  // The z residual / DCA_z ratio pads cover the ratios with |ratio| <=
  // kRatioDisplayRange and the expected ratios with |value| <=
  // kRatioReferenceRange; larger expected ratios are drawn only if inside.
  constexpr double kRatioDisplayRange = 3.0;
  constexpr double kRatioReferenceRange = 0.5;
  // MVTX stave mid radii (rMid of PHG4MvtxDefs.h) in cm, for the expected
  // ratios of expected_residual_ratio.
  constexpr double kMvtxRadiiCm[] = {2.523, 3.336, 4.148};

  struct DetectorConfig
  {
    const char* detectorName;
    const char* outputPrefix;
    const char* panelLabel;
    const char* combinedTitle;
    int firstLayer;
    int numberOfLayers;
    int chipsPerStave;
    int canvasColumns;
    int canvasRows;
  };

  constexpr DetectorConfig kMvtxConfig = {
    "MVTX", "mvtx", "Chip", "All chips combined", 0, 3, 9, 5, 2};
  constexpr DetectorConfig kInttConfig = {
    "INTT", "intt", "Sensor", "All sensors combined", 3, 4, 4, 3, 2};
  // MVTX staves (L0-L2) and INTT ladders (L3-L6) per layer; sets the
  // stave-chip ID axis of the summary pages, so missing staves show as gaps.
  constexpr int kStavesPerLayer[] = {12, 16, 20, 12, 12, 16, 16};

  // Index 0: -DCA_xy (residual_vertexX), 1: -DCA_z (residual_vertexY).
  constexpr int kNumberOfCoordinates = 2;
  constexpr const char* kCoordinateNames[kNumberOfCoordinates] = {
    "dca_xy", "dca_z"};
  constexpr const char* kCoordinateLabels[kNumberOfCoordinates] = {
    "#minusDCA_{xy}", "#minusDCA_{z}"};

  struct InputSpec
  {
    std::string listfile;
    std::string label;
  };

  struct ChipFitResult
  {
    std::string status = "low_stats";
    double entriesInWindow = 0.0;
    int populatedBins = 0;
    double amplitude = std::numeric_limits<double>::quiet_NaN();
    double mean = std::numeric_limits<double>::quiet_NaN();
    double meanError = std::numeric_limits<double>::quiet_NaN();
    double sigma = std::numeric_limits<double>::quiet_NaN();
    double sigmaError = std::numeric_limits<double>::quiet_NaN();
    double chi2 = std::numeric_limits<double>::quiet_NaN();
    int ndf = 0;
    double fitLowMm = std::numeric_limits<double>::quiet_NaN();
    double fitHighMm = std::numeric_limits<double>::quiet_NaN();
  };

  struct StaveHistograms
  {
    int layer = -1;
    int stave = -1;
    // Indexed as [coordinate][input][chip].
    std::vector<std::vector<std::vector<TH1D*>>> histograms;
    std::vector<std::vector<std::vector<ChipFitResult>>> fits;
    // Indexed as [chip]: sums of cos and sin of the cluster phi (clusphi) of
    // the filled rows of every input, giving the chip's circular-mean cluster
    // phi (no wrap-around problem for staves at phi = +-pi), and the number
    // of rows in the sums (for the circular RMS of the cluster phi).
    std::vector<double> clusterPhiCosSums;
    std::vector<double> clusterPhiSinSums;
    std::vector<double> clusterPhiCounts;
    // MVTX only (empty for INTT), indexed as [input][chip]: z residual
    // (residualY) of the same rows as the -DCA histograms.
    std::vector<std::vector<TH1D*>> residualZHistograms;
    std::vector<std::vector<ChipFitResult>> residualZFits;
  };

  using StaveHistogramMap =
    std::map<std::pair<int, int>, StaveHistograms>;

  // Per-input ntp row counts of the selection in fill_dca_histograms.
  struct SelectionCounts
  {
    long rows = 0;
    long notSilicon = 0;
    long nonFiniteDca = 0;
    long failedPt = 0;
    long unknownCrossing = 0;
    long filled = 0;
  };

  // ---------------------------------------------------------------------------
  // Inputs
  // ---------------------------------------------------------------------------

  std::string trim(const std::string& text)
  {
    const std::string::size_type first = text.find_first_not_of(" \t");
    if(first == std::string::npos)
    {
      return "";
    }

    const std::string::size_type last = text.find_last_not_of(" \t");
    return text.substr(first, last - first + 1);
  }

  std::string default_input_label(const std::string& listfile)
  {
    std::string label = listfile;
    const std::string::size_type slash = label.find_last_of('/');
    if(slash != std::string::npos)
    {
      label = label.substr(slash + 1);
    }

    const std::string::size_type dot = label.find_last_of('.');
    if(dot != std::string::npos && dot > 0)
    {
      label = label.substr(0, dot);
    }

    return label;
  }

  // Parses "list1.txt:Label1,list2.txt:Label2"; a missing label defaults to
  // the list filename without directory and extension.
  std::vector<InputSpec> parse_input_specs(const char* listfiles)
  {
    std::vector<InputSpec> inputs;
    std::stringstream stream(listfiles);
    std::string token;
    while(std::getline(stream, token, ','))
    {
      token = trim(token);
      if(token.empty())
      {
        continue;
      }

      InputSpec input;
      const std::string::size_type colon = token.find(':');
      input.listfile = trim(token.substr(0, colon));
      input.label =
        colon == std::string::npos ? "" : trim(token.substr(colon + 1));
      if(input.listfile.empty())
      {
        continue;
      }
      if(input.label.empty())
      {
        input.label = default_input_label(input.listfile);
      }
      inputs.push_back(input);
    }

    return inputs;
  }

  std::string sanitize_for_filename(const std::string& text)
  {
    std::string sanitized;
    for(const char character : text)
    {
      const bool keep =
        std::isalnum(static_cast<unsigned char>(character)) ||
        character == '-' || character == '_' || character == '.';
      sanitized += keep ? character : '_';
    }

    return sanitized.empty() ? "input" : sanitized;
  }

  // "<outputPdf without .pdf>_<label1>_<label2>....pdf" (as
  // multipage_pdf_name of plot_fits_overlaid.C), or
  // "<outputPdf without .pdf>_longfilelist.pdf" when that name is longer
  // than maxLength characters (macOS allows 255).
  std::string labelled_pdf_name(
    const std::string& outputPdf,
    const std::vector<InputSpec>& inputs,
    size_t maxLength = 200)
  {
    const std::string extension = ".pdf";
    std::string prefix = outputPdf;
    if(prefix.size() >= extension.size() &&
       prefix.compare(prefix.size() - extension.size(), extension.size(),
                      extension) == 0)
    {
      prefix.erase(prefix.size() - extension.size());
    }
    std::string name = prefix;
    for(const InputSpec& input : inputs)
    {
      name += "_" + sanitize_for_filename(input.label);
    }
    name += extension;
    return name.size() > maxLength ? prefix + "_longfilelist" + extension : name;
  }

  // Labels name the CSV files and fill a CSV column, so they must be distinct
  // after sanitizing and must not contain commas.
  bool validate_csv_labels(const std::vector<InputSpec>& inputs)
  {
    std::set<std::string> outputNames;
    for(const InputSpec& input : inputs)
    {
      if(input.label.find(',') != std::string::npos ||
         input.listfile.find(',') != std::string::npos)
      {
        std::cerr << "ERROR: labels and list files must not contain commas."
                  << std::endl;
        return false;
      }
      if(!outputNames.insert(sanitize_for_filename(input.label)).second)
      {
        std::cerr << "ERROR: duplicate input label '" << input.label
                  << "'; every input needs a distinct label." << std::endl;
        return false;
      }
    }

    return true;
  }

  bool load_ntuple_chain(TChain* chain, const char* listfile)
  {
    std::ifstream infile(listfile);
    if(!infile.is_open())
    {
      std::cerr << "ERROR: could not open " << listfile << std::endl;
      return false;
    }

    std::string filename;
    int filesAdded = 0;
    while(infile >> filename)
    {
      std::cout << "Adding " << filename << std::endl;
      const int added = chain->Add(filename.c_str());
      if(added > 0)
      {
        filesAdded += added;
      }
      else
      {
        std::cerr << "WARNING: failed to add " << filename << std::endl;
      }
    }

    if(filesAdded == 0 || chain->GetEntries() == 0)
    {
      std::cerr << "ERROR: no usable " << chain->GetName()
                << " entries were loaded from " << listfile << std::endl;
      return false;
    }

    return true;
  }

  // The track columns of ntp exist only in files written by MakeMilleFiles
  // with the per-cluster track columns (event:track_id:track_crossing:...).
  bool has_required_branches(TChain* chain)
  {
    const std::vector<std::string> requiredBranches = {
      "layer", "stave", "chip", "residual_vertexX", "residual_vertexY",
      "residualY", "track_pt", "track_crossing"};

    chain->LoadTree(0);
    for(const auto& branch : requiredBranches)
    {
      if(!chain->GetBranch(branch.c_str()))
      {
        std::cerr << "ERROR: required branch '" << branch
                  << "' is missing from tree ntp; regenerate the ntuples "
                     "with a MakeMilleFiles that writes the track columns "
                     "to ntp." << std::endl;
        return false;
      }
    }

    return true;
  }

  const DetectorConfig* get_detector_config(int layer)
  {
    for(const DetectorConfig* config : {&kMvtxConfig, &kInttConfig})
    {
      if(layer >= config->firstLayer &&
         layer < config->firstLayer + config->numberOfLayers)
      {
        return config;
      }
    }

    return nullptr;
  }

  bool get_integer_identifier(Float_t value, int& identifier)
  {
    const double doubleValue = static_cast<double>(value);
    if(!std::isfinite(value) ||
       doubleValue < static_cast<double>(std::numeric_limits<int>::min()) ||
       doubleValue > static_cast<double>(std::numeric_limits<int>::max()))
    {
      return false;
    }

    const int integerValue = static_cast<int>(value);
    if(std::fabs(value - integerValue) > 1e-5)
    {
      return false;
    }

    identifier = integerValue;
    return true;
  }

  // ---------------------------------------------------------------------------
  // Histograms
  // ---------------------------------------------------------------------------

  int input_color(int inputIndex)
  {
    return kInputColors[
      inputIndex % (sizeof(kInputColors) / sizeof(kInputColors[0]))];
  }

  int input_marker_style(int inputIndex)
  {
    return kInputMarkerStyles[
      inputIndex %
      (sizeof(kInputMarkerStyles) / sizeof(kInputMarkerStyles[0]))];
  }

  void style_input_histogram(TH1* histogram, int inputIndex)
  {
    histogram->GetXaxis()->SetTitleSize(0.055);
    histogram->GetXaxis()->CenterTitle();
    histogram->GetYaxis()->SetTitleSize(0.055);
    histogram->GetXaxis()->SetLabelSize(0.045);
    histogram->GetYaxis()->SetLabelSize(0.045);
    histogram->SetLineWidth(2);
    histogram->SetLineColor(input_color(inputIndex));
    histogram->SetMarkerColor(input_color(inputIndex));
    histogram->SetMarkerStyle(input_marker_style(inputIndex));
    histogram->SetMarkerSize(0.8);
  }

  TH1D* make_chip_histogram(
    const DetectorConfig& config,
    int layer,
    int stave,
    int chip,
    int inputIndex,
    int coordinate)
  {
    const TString histogramName = Form(
      "h_dca_%s_L%d_stave%d_chip%d_in%d_%s",
      config.outputPrefix, layer, stave, chip, inputIndex,
      kCoordinateNames[coordinate]);
    // Same bin width as kResidualBins over the display range.
    const int bins = static_cast<int>(std::lround(
      kResidualBins * kDcaHistogramRangeMm / kDcaDisplayRangeMm));
    TH1D* histogram = new TH1D(
      histogramName.Data(),
      Form("%s %d", config.panelLabel, chip),
      bins,
      -kDcaHistogramRangeMm,
      kDcaHistogramRangeMm);
    histogram->SetDirectory(nullptr);
    histogram->GetXaxis()->SetTitle(
      Form("%s [mm]", kCoordinateLabels[coordinate]));
    histogram->GetYaxis()->SetTitle("Entries");
    style_input_histogram(histogram, inputIndex);
    return histogram;
  }

  TH1D* make_residual_z_histogram(
    int layer,
    int stave,
    int chip,
    int inputIndex)
  {
    const int bins = static_cast<int>(std::lround(
      kResidualZBins * kResidualZHistogramRangeMm / kResidualZDisplayRangeMm));
    TH1D* histogram = new TH1D(
      Form("h_resz_mvtx_L%d_stave%d_chip%d_in%d", layer, stave, chip,
           inputIndex),
      Form("Chip %d", chip),
      bins,
      -kResidualZHistogramRangeMm,
      kResidualZHistogramRangeMm);
    histogram->SetDirectory(nullptr);
    histogram->GetXaxis()->SetTitle("local z residual [mm]");
    histogram->GetYaxis()->SetTitle("Entries");
    style_input_histogram(histogram, inputIndex);
    return histogram;
  }

  StaveHistograms make_stave_histograms(
    const DetectorConfig& config,
    int layer,
    int stave,
    int numberOfInputs)
  {
    StaveHistograms histograms;
    histograms.layer = layer;
    histograms.stave = stave;
    histograms.histograms.resize(kNumberOfCoordinates);
    histograms.fits.resize(kNumberOfCoordinates);
    histograms.clusterPhiCosSums.assign(config.chipsPerStave, 0.0);
    histograms.clusterPhiSinSums.assign(config.chipsPerStave, 0.0);
    histograms.clusterPhiCounts.assign(config.chipsPerStave, 0.0);
    for(int coordinate = 0; coordinate < kNumberOfCoordinates; ++coordinate)
    {
      histograms.histograms[coordinate].resize(numberOfInputs);
      histograms.fits[coordinate].assign(
        numberOfInputs, std::vector<ChipFitResult>(config.chipsPerStave));
      for(int input = 0; input < numberOfInputs; ++input)
      {
        histograms.histograms[coordinate][input].reserve(config.chipsPerStave);
        for(int chip = 0; chip < config.chipsPerStave; ++chip)
        {
          histograms.histograms[coordinate][input].push_back(
            make_chip_histogram(config, layer, stave, chip, input, coordinate));
        }
      }
    }

    if(&config == &kMvtxConfig)
    {
      histograms.residualZHistograms.resize(numberOfInputs);
      histograms.residualZFits.assign(
        numberOfInputs, std::vector<ChipFitResult>(config.chipsPerStave));
      for(int input = 0; input < numberOfInputs; ++input)
      {
        for(int chip = 0; chip < config.chipsPerStave; ++chip)
        {
          histograms.residualZHistograms[input].push_back(
            make_residual_z_histogram(layer, stave, chip, input));
        }
      }
    }

    return histograms;
  }

  // Fills the -DCA of the track of every MVTX/INTT ntp row into the
  // histograms of the row's chip and, for MVTX with a finite residualY, the
  // cluster z residual into its z residual histogram (same rows, so the two
  // come from the same tracks). Rows need both -DCA components finite
  // (NaN when MakeMilleFiles ran without the event vertex), pT at or above
  // the cut (a non-finite pT fails the comparison) and, with
  // excludeUnknownCrossing, a known track crossing. With hasClusterPhi, the
  // cluster phi (clusphi) of every filled row is added to its chip's
  // circular-mean sums; a missing clusphi branch must not be read, since
  // TTreeReader stops at a reader value without branch.
  SelectionCounts fill_dca_histograms(
    TChain* ntp,
    int inputIndex,
    int numberOfInputs,
    double minimumPtGeV,
    bool excludeUnknownCrossing,
    bool hasClusterPhi,
    StaveHistogramMap& histograms)
  {
    SelectionCounts counts;
    TTreeReader reader(ntp);
    TTreeReaderValue<Float_t> layerValue(reader, "layer");
    TTreeReaderValue<Float_t> staveValue(reader, "stave");
    TTreeReaderValue<Float_t> chipValue(reader, "chip");
    TTreeReaderValue<Float_t> transverseValue(reader, "residual_vertexX");
    TTreeReaderValue<Float_t> longitudinalValue(reader, "residual_vertexY");
    TTreeReaderValue<Float_t> residualZValue(reader, "residualY");
    TTreeReaderValue<Float_t> ptValue(reader, "track_pt");
    TTreeReaderValue<Float_t> crossingValue(reader, "track_crossing");
    std::unique_ptr<TTreeReaderValue<Float_t>> clusterPhiValue;
    if(hasClusterPhi)
    {
      clusterPhiValue =
        std::make_unique<TTreeReaderValue<Float_t>>(reader, "clusphi");
    }

    while(reader.Next())
    {
      ++counts.rows;
      int layer = -1;
      int stave = -1;
      int chip = -1;
      if(!get_integer_identifier(*layerValue, layer) ||
         !get_integer_identifier(*staveValue, stave) ||
         !get_integer_identifier(*chipValue, chip))
      {
        ++counts.notSilicon;
        continue;
      }

      const DetectorConfig* config = get_detector_config(layer);
      if(!config || stave < 0 || chip < 0 || chip >= config->chipsPerStave)
      {
        ++counts.notSilicon;
        continue;
      }
      if(!std::isfinite(*transverseValue) ||
         !std::isfinite(*longitudinalValue))
      {
        ++counts.nonFiniteDca;
        continue;
      }
      if(!(*ptValue >= minimumPtGeV))
      {
        ++counts.failedPt;
        continue;
      }
      if(excludeUnknownCrossing &&
         (!std::isfinite(*crossingValue) || *crossingValue == kUnknownCrossing))
      {
        ++counts.unknownCrossing;
        continue;
      }

      const std::pair<int, int> key(layer, stave);
      auto iterator = histograms.find(key);
      if(iterator == histograms.end())
      {
        iterator = histograms.emplace(
          key, make_stave_histograms(
            *config, layer, stave, numberOfInputs)).first;
      }

      iterator->second.histograms[0][inputIndex][chip]->Fill(*transverseValue);
      iterator->second.histograms[1][inputIndex][chip]->Fill(
        *longitudinalValue);
      if(!iterator->second.residualZHistograms.empty() &&
         std::isfinite(*residualZValue))
      {
        iterator->second.residualZHistograms[inputIndex][chip]->Fill(
          *residualZValue);
      }
      if(clusterPhiValue && std::isfinite(**clusterPhiValue))
      {
        iterator->second.clusterPhiCosSums[chip] += std::cos(**clusterPhiValue);
        iterator->second.clusterPhiSinSums[chip] += std::sin(**clusterPhiValue);
        iterator->second.clusterPhiCounts[chip] += 1.0;
      }
      ++counts.filled;
    }

    return counts;
  }

  void delete_all_histograms(StaveHistogramMap& staveHistograms)
  {
    for(auto& entry : staveHistograms)
    {
      for(auto& inputHistograms : entry.second.histograms)
      {
        for(auto& chipHistograms : inputHistograms)
        {
          for(TH1D* histogram : chipHistograms)
          {
            delete histogram;
          }
          chipHistograms.clear();
        }
      }
      for(auto& chipHistograms : entry.second.residualZHistograms)
      {
        for(TH1D* histogram : chipHistograms)
        {
          delete histogram;
        }
        chipHistograms.clear();
      }
    }
    staveHistograms.clear();
  }

  // ---------------------------------------------------------------------------
  // Fits and summary statistics (shared by the CSV tables and the plots)
  // ---------------------------------------------------------------------------

  struct WindowSummary
  {
    double entries = 0.0;
    int populatedBins = 0;
  };

  WindowSummary summarize_window(
    const TH1D* histogram,
    double lowMm,
    double highMm)
  {
    WindowSummary summary;
    const TAxis* axis = histogram->GetXaxis();
    const int firstBin = std::max(1, axis->FindFixBin(lowMm));
    const int lastBin = std::min(axis->GetNbins(), axis->FindFixBin(highMm));
    for(int bin = firstBin; bin <= lastBin; ++bin)
    {
      const double binContent = histogram->GetBinContent(bin);
      summary.entries += binContent;
      if(binContent > 0.0)
      {
        ++summary.populatedBins;
      }
    }

    return summary;
  }

  double out_of_range_entries(const TH1D* histogram)
  {
    return histogram->GetBinContent(0) +
           histogram->GetBinContent(histogram->GetNbinsX() + 1);
  }

  // A distribution mostly outside the histogram range is reported as
  // out_of_range, not as low_stats, so it is never mistaken for a chip
  // without data.
  bool mark_out_of_range(const TH1D* histogram, ChipFitResult& result)
  {
    const double entries = histogram->GetEntries();
    if(entries < kMinimumEntriesForFit ||
       out_of_range_entries(histogram) <=
         kMaximumOutOfRangeFraction * entries)
    {
      return false;
    }

    result.status = "out_of_range";
    return true;
  }

  // Seeds a Gaussian at the peak bin with the configured half-width, then
  // refits within mean +/- kCoreFitSigmaWindow * sigma, so a shifted
  // distribution is not truncated by a window fixed at zero. Binned Poisson
  // likelihood ("L") keeps low-count bins unbiased. The function is not
  // stored in the histogram ("N"), so later clones carry no fit.
  ChipFitResult fit_core_gaussian(TH1D* histogram, double seedHalfWidthMm)
  {
    ChipFitResult result;
    if(mark_out_of_range(histogram, result))
    {
      return result;
    }

    const TAxis* axis = histogram->GetXaxis();
    const double axisLow = axis->GetXmin();
    const double axisHigh = axis->GetXmax();
    const double peak = axis->GetBinCenter(histogram->GetMaximumBin());
    double low = std::max(axisLow, peak - seedHalfWidthMm);
    double high = std::min(axisHigh, peak + seedHalfWidthMm);

    TF1 fit(Form("f_%s", histogram->GetName()), "gaus", low, high);
    fit.SetParameters(histogram->GetMaximum(), peak, 0.5 * seedHalfWidthMm);
    for(int iteration = 0; iteration <= kCoreFitIterations; ++iteration)
    {
      const WindowSummary windowSummary = summarize_window(
        histogram, low, high);
      result.entriesInWindow = windowSummary.entries;
      result.populatedBins = windowSummary.populatedBins;
      result.fitLowMm = low;
      result.fitHighMm = high;
      if(windowSummary.entries < kMinimumEntriesForFit ||
         windowSummary.populatedBins < kMinimumPopulatedBinsForFit)
      {
        // Enough data for the seed fit but not for the narrowed core window
        // means the fit did not converge to a usable width.
        result.status = iteration == 0 ? "low_stats" : "fit_failed";
        return result;
      }

      fit.SetRange(low, high);
      const int fitStatus = histogram->Fit(&fit, "RLQN0");
      const double mean = fit.GetParameter(1);
      const double sigma = std::fabs(fit.GetParameter(2));
      if(fitStatus != 0 || !std::isfinite(mean) || !std::isfinite(sigma) ||
         sigma <= 0.0)
      {
        result.status = "fit_failed";
        return result;
      }

      result.status = "ok";
      result.amplitude = fit.GetParameter(0);
      result.mean = mean;
      result.meanError = fit.GetParError(1);
      result.sigma = sigma;
      result.sigmaError = fit.GetParError(2);
      result.chi2 = fit.GetChisquare();
      result.ndf = fit.GetNDF();
      low = std::max(axisLow, mean - kCoreFitSigmaWindow * sigma);
      high = std::min(axisHigh, mean + kCoreFitSigmaWindow * sigma);
    }

    return result;
  }

  void fit_all_histograms(
    StaveHistogramMap& staveHistograms,
    int numberOfInputs)
  {
    for(auto& entry : staveHistograms)
    {
      StaveHistograms& histograms = entry.second;
      const DetectorConfig* config = get_detector_config(histograms.layer);
      for(int coordinate = 0; coordinate < kNumberOfCoordinates; ++coordinate)
      {
        for(int input = 0; input < numberOfInputs; ++input)
        {
          for(int chip = 0; chip < config->chipsPerStave; ++chip)
          {
            histograms.fits[coordinate][input][chip] = fit_core_gaussian(
              histograms.histograms[coordinate][input][chip], kDcaFitRangeMm);
          }
        }
      }
      for(size_t input = 0; input < histograms.residualZHistograms.size();
          ++input)
      {
        for(int chip = 0; chip < config->chipsPerStave; ++chip)
        {
          histograms.residualZFits[input][chip] = fit_core_gaussian(
            histograms.residualZHistograms[input][chip], kResidualZFitRangeMm);
        }
      }
    }
  }

  struct TruncatedMean
  {
    double entries = 0.0;
    double mean = std::numeric_limits<double>::quiet_NaN();
    double meanError = std::numeric_limits<double>::quiet_NaN();
  };

  // Unbinned mean and its uncertainty (RMS / sqrt(N)) of the fills inside the
  // histogram axis range (the histogram range, not the display range),
  // taken from the histogram statistics.
  TruncatedMean truncated_mean(const TH1D* histogram)
  {
    TruncatedMean result;
    double stats[4] = {0.0, 0.0, 0.0, 0.0};
    histogram->GetStats(stats);
    const double sumWeights = stats[0];
    result.entries = sumWeights;
    if(sumWeights < 2.0)
    {
      return result;
    }

    result.mean = stats[2] / sumWeights;
    const double variance =
      std::max(0.0, stats[3] / sumWeights - result.mean * result.mean);
    result.meanError = std::sqrt(variance / sumWeights);
    return result;
  }

  // ---------------------------------------------------------------------------
  // CSV tables
  // ---------------------------------------------------------------------------

  // Same columns as residual_fits_<label>.csv of
  // plot_fits_overlaid_stavechip.C; fit_model is always "gaus", so the box
  // columns are NaN.
  void write_fit_row(
    std::ofstream& output,
    const InputSpec& input,
    const char* detectorName,
    int layer,
    int stave,
    int chip,
    const char* coordinateName,
    const TH1D* histogram,
    const ChipFitResult& fitResult)
  {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const TruncatedMean truncatedMean = truncated_mean(histogram);
    output << input.label << ',' << input.listfile << ','
           << detectorName << ',' << layer << ',' << stave << ','
           << chip << ',' << coordinateName << ','
           << histogram->GetEntries() << ','
           << fitResult.entriesInWindow << ',' << fitResult.populatedBins
           << ',' << fitResult.status << ','
           << fitResult.mean << ',' << fitResult.meanError << ','
           << fitResult.sigma << ',' << fitResult.sigmaError << ','
           << fitResult.chi2 << ',' << fitResult.ndf << ','
           << fitResult.fitLowMm << ',' << fitResult.fitHighMm << ','
           << truncatedMean.mean << ',' << truncatedMean.meanError << ','
           << "gaus" << ','
           << histogram->GetBinContent(0) << ','
           << histogram->GetBinContent(histogram->GetNbinsX() + 1) << ','
           << nan << ',' << nan << ',' << nan << ',' << nan << '\n';
  }

  // Writes dca_fits_<label>.csv per input, one row per
  // (layer, stave, chip, coordinate). Returns false if a file could not be
  // written.
  bool write_csv_tables(
    const std::vector<InputSpec>& inputs,
    const StaveHistogramMap& staveHistograms,
    const char* outputDirectory)
  {
    gSystem->mkdir(outputDirectory, kTRUE);
    const int numberOfInputs = static_cast<int>(inputs.size());
    for(int input = 0; input < numberOfInputs; ++input)
    {
      const std::string outputPath =
        std::string(outputDirectory) + "/dca_fits_" +
        sanitize_for_filename(inputs[input].label) + ".csv";
      std::ofstream output(outputPath);
      if(!output.is_open())
      {
        std::cerr << "ERROR: could not write " << outputPath << std::endl;
        return false;
      }

      output << std::setprecision(8)
             << "label,listfile,detector,layer,stave,chip,coord,entries,"
                "entries_in_fit_window,populated_bins,fit_status,mu_mm,"
                "mu_err_mm,sigma_mm,sigma_err_mm,chi2,ndf,fit_low_mm,"
                "fit_high_mm,trunc_mean_mm,trunc_mean_err_mm,fit_model,"
                "n_underflow,n_overflow,box_length_mm,box_length_err_mm,"
                "edge_sigma_mm,edge_sigma_err_mm\n";

      int rows = 0;
      int goodFits = 0;
      for(const auto& entry : staveHistograms)
      {
        const StaveHistograms& histograms = entry.second;
        const DetectorConfig* config = get_detector_config(histograms.layer);
        for(int chip = 0; chip < config->chipsPerStave; ++chip)
        {
          for(int coordinate = 0; coordinate < kNumberOfCoordinates;
              ++coordinate)
          {
            const ChipFitResult& fit = histograms.fits[coordinate][input][chip];
            write_fit_row(
              output, inputs[input], config->detectorName, histograms.layer,
              histograms.stave, chip, kCoordinateNames[coordinate],
              histograms.histograms[coordinate][input][chip], fit);
            ++rows;
            goodFits += fit.status == "ok";
          }
        }
      }

      std::cout << "Wrote " << rows << " rows (" << goodFits
                << " successful fits) for " << inputs[input].label << " to "
                << outputPath << std::endl;
    }

    return true;
  }

  // ---------------------------------------------------------------------------
  // Plots
  // ---------------------------------------------------------------------------

  // First and last bin of the displayed window [-displayRangeMm,
  // displayRangeMm] of a histogram filled over a wider range.
  std::pair<int, int> display_bin_range(
    const TH1D* histogram,
    double displayRangeMm)
  {
    const TAxis* axis = histogram->GetXaxis();
    const double halfBin = 0.5 * axis->GetBinWidth(1);
    return {axis->FindFixBin(-displayRangeMm + halfBin),
            axis->FindFixBin(displayRangeMm - halfBin)};
  }

  // Draws the Gaussian written to the CSV over its final fit window clipped
  // to the displayed range, with the amplitude scaled like the displayed
  // (possibly unit-area normalized) histogram. Returns the legend text.
  TString draw_core_fit(
    const TH1D* rawHistogram,
    const TH1D* displayHistogram,
    const ChipFitResult& fitResult,
    double scale,
    double displayRangeMm,
    std::vector<TObject*>& ownedObjects)
  {
    if(fitResult.status == "low_stats")
    {
      return Form("too few in fit range (%.0f, %d bins)",
                  fitResult.entriesInWindow, fitResult.populatedBins);
    }
    if(fitResult.status == "out_of_range")
    {
      return Form("beyond #pm%g mm, no fit",
                  rawHistogram->GetXaxis()->GetXmax());
    }
    if(fitResult.status != "ok")
    {
      return "Gaussian fit failed";
    }

    const TString legendText =
      Form("#mu=%.4f, #sigma=%.4f", fitResult.mean, fitResult.sigma);
    const double drawLow = std::max(fitResult.fitLowMm, -displayRangeMm);
    const double drawHigh = std::min(fitResult.fitHighMm, displayRangeMm);
    if(!(drawHigh > drawLow))
    {
      return legendText;
    }

    TF1* fit = new TF1(
      Form("f_%s", displayHistogram->GetName()), "gaus", drawLow, drawHigh);
    ownedObjects.push_back(fit);
    fit->SetParameters(
      fitResult.amplitude * scale, fitResult.mean, fitResult.sigma);
    fit->SetLineColor(displayHistogram->GetLineColor());
    fit->SetLineWidth(2);
    fit->Draw("same");
    return legendText;
  }

  // Draws one curve per input in the current pad, zoomed to
  // +-displayRangeMm, with a legend carrying the input label, entry count,
  // the share of entries outside the view and (optionally) the fit result.
  // fitResults is indexed like rawHistograms and is used only if
  // includeGaussianFit is set. Unit-area normalization uses every entry in
  // the histogram range, not only the displayed ones.
  void draw_overlay_panel(
    const std::vector<TH1D*>& rawHistograms,
    const std::vector<ChipFitResult>& fitResults,
    const std::vector<InputSpec>& inputs,
    const TString& panelText,
    double displayRangeMm,
    bool includeGaussianFit,
    bool drawHistogramsAsMarkers,
    bool normalizeToUnitArea,
    std::vector<TObject*>& ownedObjects)
  {
    constexpr double leftMargin = 0.20;
    constexpr double bottomMargin = 0.17;
    constexpr double topMargin = 0.11;
    gPad->SetGrid(0, 0);
    gPad->SetLeftMargin(leftMargin);
    gPad->SetRightMargin(0.05);
    gPad->SetBottomMargin(bottomMargin);
    gPad->SetTopMargin(topMargin);

    const int numberOfInputs = static_cast<int>(rawHistograms.size());
    std::vector<TH1D*> displayHistograms;
    std::vector<double> scales;
    displayHistograms.reserve(numberOfInputs);
    scales.reserve(numberOfInputs);
    double maximum = 0.0;
    for(TH1D* rawHistogram : rawHistograms)
    {
      TH1D* displayHistogram = static_cast<TH1D*>(rawHistogram->Clone(
        Form("%s_display", rawHistogram->GetName())));
      displayHistogram->SetDirectory(nullptr);
      ownedObjects.push_back(displayHistogram);
      // Integral over the full histogram range: no axis range is set yet.
      const double integral = displayHistogram->Integral();
      const double scale =
        normalizeToUnitArea && integral > 0.0 ? 1.0 / integral : 1.0;
      if(scale != 1.0)
      {
        displayHistogram->Scale(scale);
      }
      const std::pair<int, int> displayBins =
        display_bin_range(displayHistogram, displayRangeMm);
      displayHistogram->GetXaxis()->SetRange(
        displayBins.first, displayBins.second);
      displayHistogram->GetYaxis()->SetTitle(
        normalizeToUnitArea ? "Normalized entries" : "Entries");
      // GetMaximum is taken over the displayed bins only.
      maximum = std::max(maximum, displayHistogram->GetMaximum());
      displayHistograms.push_back(displayHistogram);
      scales.push_back(scale);
    }

    // Leave headroom so the tallest curve stays below the legend.
    constexpr double legendTop = 0.88;
    constexpr double legendRowHeight = 0.065;
    const double legendBottom = legendTop - legendRowHeight * numberOfInputs;
    const double frameHeight = 1.0 - topMargin - bottomMargin;
    const double usableFraction = std::max(
      0.3, (legendBottom - 0.02 - bottomMargin) / frameHeight);
    displayHistograms.front()->SetMaximum(
      maximum > 0.0 ? maximum / usableFraction : 1.0);
    displayHistograms.front()->GetYaxis()->SetTitleOffset(1.6);

    const TString drawOption = drawHistogramsAsMarkers ? "P" : "HIST";
    for(int input = 0; input < numberOfInputs; ++input)
    {
      displayHistograms[input]->Draw(
        input == 0 ? drawOption : drawOption + " same");
    }

    TLegend* legend = new TLegend(
      leftMargin + 0.02, legendBottom, 0.94, legendTop);
    ownedObjects.push_back(legend);
    legend->SetBorderSize(0);
    legend->SetFillStyle(0);
    legend->SetTextFont(42);
    legend->SetTextSize(0.036);
    legend->SetMargin(0.10);
    for(int input = 0; input < numberOfInputs; ++input)
    {
      const TH1D* rawHistogram = rawHistograms[input];
      const double entries = rawHistogram->GetEntries();
      TString entryText = Form(
        "%s (%.0f)", inputs[input].label.c_str(), entries);
      // Entries outside the view, including the under/overflow.
      const std::pair<int, int> displayBins =
        display_bin_range(rawHistogram, displayRangeMm);
      const double outsideView = entries -
        rawHistogram->Integral(displayBins.first, displayBins.second);
      if(entries > 0.0 && outsideView > 0.01 * entries)
      {
        entryText += Form(" [%.0f%% off-scale]",
                          100.0 * outsideView / entries);
      }
      if(includeGaussianFit)
      {
        entryText += ": ";
        entryText += draw_core_fit(
          rawHistogram, displayHistograms[input], fitResults[input],
          scales[input], displayRangeMm, ownedObjects);
      }
      legend->AddEntry(
        displayHistograms[input], entryText.Data(),
        drawHistogramsAsMarkers ? "p" : "l");
    }
    legend->Draw();

    TLatex label;
    label.SetNDC();
    label.SetTextFont(42);
    label.SetTextSize(0.052);
    label.DrawLatex(leftMargin + 0.03, 0.90, panelText.Data());
  }

  // Shrinks every pad so that the top pageTitleHeight of the canvas stays free
  // for the page title.
  void reserve_page_title_space(
    TCanvas* canvas,
    int numberOfPads,
    double pageTitleHeight)
  {
    for(int padIndex = 1; padIndex <= numberOfPads; ++padIndex)
    {
      TVirtualPad* pad = canvas->GetPad(padIndex);
      if(!pad)
      {
        continue;
      }

      const double xLow = pad->GetXlowNDC();
      const double yLow = pad->GetYlowNDC();
      const double width = pad->GetWNDC();
      const double height = pad->GetHNDC();
      pad->SetPad(
        xLow, yLow * (1.0 - pageTitleHeight), xLow + width,
        (yLow + height) * (1.0 - pageTitleHeight));
    }
  }

  void draw_page_title(
    TCanvas* canvas,
    double pageTitleHeight,
    const TString& title)
  {
    canvas->cd();
    TLatex pageTitle;
    pageTitle.SetNDC();
    pageTitle.SetTextFont(62);
    pageTitle.SetTextSize(0.030);
    pageTitle.SetTextAlign(22);
    pageTitle.DrawLatex(0.5, 1.0 - 0.5 * pageTitleHeight, title.Data());
  }

  // ROOT takes the orientation of the whole PDF and the geometry of its first
  // page from the canvas that opens it, so the first page opens the file
  // ("(") instead of a dummy 1x1 canvas, which scrambled the first page. Every
  // page is then landscape like the (wide) first one; all canvases here are
  // wider than tall, so none is clipped.
  struct PdfOutput
  {
    std::string path;
    bool isOpen = false;
    bool savePngs = false;
  };

  void print_and_delete_page(
    TCanvas* canvas,
    PdfOutput& outputPdf,
    const char* outputPng,
    std::vector<TObject*>& ownedObjects)
  {
    canvas->Print((outputPdf.path + (outputPdf.isOpen ? "" : "(")).c_str());
    outputPdf.isOpen = true;
    if(outputPdf.savePngs)
    {
      canvas->SaveAs(outputPng);
    }
    delete canvas;
    // Delete in reverse so legends and fits go before the histograms they use.
    for(auto object = ownedObjects.rbegin(); object != ownedObjects.rend();
        ++object)
    {
      delete *object;
    }
    ownedObjects.clear();
  }

  // One page: one panel per chip plus the stave sum, for one coordinate.
  void draw_chip_dca_page(
    const DetectorConfig& config,
    int layer,
    int stave,
    int coordinate,
    const std::vector<std::vector<TH1D*>>& inputChipHistograms,
    const std::vector<std::vector<ChipFitResult>>& inputChipFits,
    const std::vector<InputSpec>& inputs,
    const TString& selectionText,
    bool includeGaussianFit,
    bool drawHistogramsAsMarkers,
    bool normalizeToUnitArea,
    PdfOutput& outputPdf,
    const char* outputPng)
  {
    const char* coordinateName = kCoordinateNames[coordinate];
    const char* coordinateLabel = kCoordinateLabels[coordinate];
    const TString canvasName = Form(
      "c_dca_%s_L%d_stave%d_%s",
      config.outputPrefix, layer, stave, coordinateName);
    const TString canvasTitle = Form(
      "%s layer %d, stave %d: %s", config.detectorName, layer, stave,
      coordinateLabel);
    const int numberOfInputs = static_cast<int>(inputs.size());
    std::vector<TObject*> ownedObjects;

    // Per-input stave sums. They are not in the CSV, so they are fitted here
    // when the fit is shown. A track crossing two chips of the stave enters
    // twice.
    std::vector<TH1D*> combinedHistograms;
    std::vector<ChipFitResult> combinedFits(numberOfInputs);
    combinedHistograms.reserve(numberOfInputs);
    for(int input = 0; input < numberOfInputs; ++input)
    {
      const std::vector<TH1D*>& chipHistograms = inputChipHistograms[input];
      TH1D* combinedHistogram = static_cast<TH1D*>(
        chipHistograms.front()->Clone(Form(
          "h_dca_%s_L%d_stave%d_allchips_in%d_%s",
          config.outputPrefix, layer, stave, input, coordinateName)));
      combinedHistogram->SetDirectory(nullptr);
      for(int chip = 1; chip < config.chipsPerStave; ++chip)
      {
        combinedHistogram->Add(chipHistograms[chip]);
      }
      combinedHistogram->SetTitle(config.combinedTitle);
      ownedObjects.push_back(combinedHistogram);
      combinedHistograms.push_back(combinedHistogram);
      if(includeGaussianFit)
      {
        combinedFits[input] =
          fit_core_gaussian(combinedHistogram, kDcaFitRangeMm);
      }
    }

    TCanvas* canvas = new TCanvas(
      canvasName.Data(), canvasTitle.Data(),
      450 * config.canvasColumns, 500 * config.canvasRows);
    canvas->Divide(config.canvasColumns, config.canvasRows, 0.001, 0.001);
    constexpr double pageTitleHeight = 0.055;
    reserve_page_title_space(
      canvas, config.canvasColumns * config.canvasRows, pageTitleHeight);
    for(int chip = 0; chip < config.chipsPerStave; ++chip)
    {
      canvas->cd(chip + 1);
      std::vector<TH1D*> chipHistograms;
      std::vector<ChipFitResult> chipFits;
      chipHistograms.reserve(numberOfInputs);
      chipFits.reserve(numberOfInputs);
      for(int input = 0; input < numberOfInputs; ++input)
      {
        chipHistograms.push_back(inputChipHistograms[input][chip]);
        chipFits.push_back(inputChipFits[input][chip]);
      }
      draw_overlay_panel(
        chipHistograms, chipFits, inputs,
        Form("%s %d", config.panelLabel, chip), kDcaDisplayRangeMm,
        includeGaussianFit, drawHistogramsAsMarkers, normalizeToUnitArea,
        ownedObjects);
    }

    canvas->cd(config.chipsPerStave + 1);
    draw_overlay_panel(
      combinedHistograms, combinedFits, inputs,
      Form("%ss 0-%d combined", config.panelLabel, config.chipsPerStave - 1),
      kDcaDisplayRangeMm, includeGaussianFit, drawHistogramsAsMarkers,
      normalizeToUnitArea, ownedObjects);

    draw_page_title(
      canvas, pageTitleHeight,
      Form("%s layer %d, stave %d: %s of the tracks through each %s%s",
           config.detectorName, layer, stave, coordinateLabel,
           TString(config.panelLabel).Data(), selectionText.Data()));
    print_and_delete_page(canvas, outputPdf, outputPng, ownedObjects);
  }

  int draw_detector_pages(
    const DetectorConfig& config,
    const StaveHistogramMap& histogramMap,
    const std::vector<InputSpec>& inputs,
    const TString& selectionText,
    PdfOutput& outputPdf,
    bool includeGaussianFit,
    bool drawHistogramsAsMarkers,
    bool normalizeToUnitArea)
  {
    int populatedStaves = 0;

    for(const auto& entry : histogramMap)
    {
      const StaveHistograms& histograms = entry.second;
      if(histograms.layer < config.firstLayer ||
         histograms.layer >= config.firstLayer + config.numberOfLayers)
      {
        continue;
      }

      ++populatedStaves;
      const TString outputBase = Form(
        "%s_dca_fits_L%d_stave%d", config.outputPrefix,
        histograms.layer, histograms.stave);
      for(int coordinate = 0; coordinate < kNumberOfCoordinates; ++coordinate)
      {
        draw_chip_dca_page(
          config, histograms.layer, histograms.stave, coordinate,
          histograms.histograms[coordinate], histograms.fits[coordinate],
          inputs, selectionText, includeGaussianFit, drawHistogramsAsMarkers,
          normalizeToUnitArea, outputPdf,
          (outputBase + "_" + kCoordinateNames[coordinate] + ".png").Data());
      }
    }

    return populatedStaves;
  }

  // Per-chip fit results of one layer, coordinate and input versus the
  // stave-chip ID (chipsPerStave * stave + chip, as in plot_fits_overlaid.C)
  // or, with versusClusterPhi, versus the chip's circular-mean cluster phi.
  // Only chips with fit_status ok (and, versus phi, a cluster phi) get a
  // point; populatedChips counts the chips with any entry.
  struct StaveChipGraphs
  {
    TGraphErrors* mean = nullptr;
    TGraphErrors* sigma = nullptr;
    int populatedChips = 0;
  };

  StaveChipGraphs make_stavechip_graphs(
    const DetectorConfig& config,
    int layer,
    int coordinate,
    int inputIndex,
    bool versusClusterPhi,
    const StaveHistogramMap& histogramMap)
  {
    const char* xName = versusClusterPhi ? "clusphi" : "stavechip";
    StaveChipGraphs graphs;
    graphs.mean = new TGraphErrors();
    graphs.sigma = new TGraphErrors();
    graphs.mean->SetName(Form(
      "g_dca_%s_L%d_in%d_%s_vs_%s_mean", config.outputPrefix, layer,
      inputIndex, kCoordinateNames[coordinate], xName));
    graphs.sigma->SetName(Form(
      "g_dca_%s_L%d_in%d_%s_vs_%s_sigma", config.outputPrefix, layer,
      inputIndex, kCoordinateNames[coordinate], xName));
    for(const auto& entry : histogramMap)
    {
      const StaveHistograms& histograms = entry.second;
      if(histograms.layer != layer)
      {
        continue;
      }

      for(int chip = 0; chip < config.chipsPerStave; ++chip)
      {
        if(histograms.histograms[coordinate][inputIndex][chip]
             ->GetEntries() <= 0.0)
        {
          continue;
        }

        ++graphs.populatedChips;
        const ChipFitResult& fit = histograms.fits[coordinate][inputIndex][chip];
        if(fit.status != "ok")
        {
          continue;
        }

        double x = config.chipsPerStave * histograms.stave + chip;
        if(versusClusterPhi)
        {
          const double cosSum = histograms.clusterPhiCosSums[chip];
          const double sinSum = histograms.clusterPhiSinSums[chip];
          if(cosSum == 0.0 && sinSum == 0.0)
          {
            continue;
          }
          x = std::atan2(sinSum, cosSum);
        }

        const int point = graphs.mean->GetN();
        graphs.mean->SetPoint(point, x, fit.mean);
        graphs.mean->SetPointError(point, 0.0, fit.meanError);
        graphs.sigma->SetPoint(point, x, fit.sigma);
        graphs.sigma->SetPointError(point, 0.0, fit.sigmaError);
      }
    }

    for(TGraphErrors* graph : {graphs.mean, graphs.sigma})
    {
      graph->SetLineColor(input_color(inputIndex));
      graph->SetMarkerColor(input_color(inputIndex));
      graph->SetMarkerStyle(input_marker_style(inputIndex));
      graph->SetMarkerSize(0.8);
    }

    return graphs;
  }

  // Fit drawn over every input graph of a summary pad: a constant (pol0) or
  // a straight line (pol1) versus stave-chip ID, or
  // p0 + p1 sin(phi) + p2 cos(phi) versus chip phi (first harmonic of
  // -DCA_xy; the DCA is to the event vertex, so a common translation of the
  // silicon or of the beam spot is absorbed and does not produce it).
  enum class SummaryFit
  {
    none,
    constant,
    line,
    sinusoid
  };

  // Draws the per-input graphs of one statistic (Gaussian mean or sigma) in
  // the current pad over [xLow, xHigh], with dotted vertical lines at
  // boundaryLines (the stave boundaries on the stave-chip ID axis). The y
  // range covers the points with |value| <= kDcaDisplayRangeMm
  // (sigma: 0 < value) and error below that, so one bad chip cannot squash
  // the others; points left outside the range are counted in the legend.
  // With summaryFit, the same in-range points are fitted (weighted by the
  // Gaussian fit errors) and the fit is drawn and listed in the legend.
  // Chips share tracks, so the fit errors and chi2 are optimistic.
  void draw_stavechip_summary_pad(
    const std::vector<TGraphErrors*>& graphs,
    const std::vector<int>& populatedChips,
    const std::vector<InputSpec>& inputs,
    bool isSigma,
    SummaryFit summaryFit,
    double xLow,
    double xHigh,
    const std::vector<double>& boundaryLines,
    const TString& chipWord,
    const TString& frameName,
    const TString& xTitle,
    const TString& yTitle,
    std::vector<TObject*>& ownedObjects)
  {
    constexpr double leftMargin = 0.14;
    constexpr double bottomMargin = 0.17;
    constexpr double topMargin = 0.04;
    gPad->SetLeftMargin(leftMargin);
    gPad->SetRightMargin(0.03);
    gPad->SetBottomMargin(bottomMargin);
    gPad->SetTopMargin(topMargin);

    auto in_display_range = [isSigma](double value, double error)
    {
      const bool valueInRange = isSigma
        ? value > 0.0 && value <= kDcaDisplayRangeMm
        : std::fabs(value) <= kDcaDisplayRangeMm;
      return valueInRange && std::isfinite(error) && error > 0.0 &&
             error <= kDcaDisplayRangeMm;
    };

    double yLow = std::numeric_limits<double>::max();
    double yHigh = std::numeric_limits<double>::lowest();
    for(const TGraphErrors* graph : graphs)
    {
      for(int point = 0; point < graph->GetN(); ++point)
      {
        const double value = graph->GetY()[point];
        const double error = graph->GetEY()[point];
        if(!in_display_range(value, error))
        {
          continue;
        }

        yLow = std::min(yLow, value - error);
        yHigh = std::max(yHigh, value + error);
      }
    }

    // Leave headroom so the points stay below the legend (one row per input,
    // plus one per fit, plus the fit-window header).
    const int numberOfInputs = static_cast<int>(inputs.size());
    const bool hasFit = summaryFit != SummaryFit::none;
    const double legendRowHeight = hasFit ? 0.05 : 0.06;
    const int legendRows = numberOfInputs * (hasFit ? 2 : 1) + 1;
    const double legendTop = 1.0 - topMargin - 0.02;
    const double legendBottom = legendTop - legendRowHeight * legendRows;
    double rangeLow = isSigma ? 0.0 : -kDcaDisplayRangeMm;
    double rangeHigh = kDcaDisplayRangeMm;
    if(yHigh >= yLow)
    {
      const double span = std::max(yHigh - yLow, 0.002);
      rangeLow = yLow - 0.1 * span;
      if(isSigma)
      {
        rangeLow = std::max(0.0, rangeLow);
      }
      const double frameHeight = 1.0 - topMargin - bottomMargin;
      const double usableFraction = std::max(
        0.3, (legendBottom - 0.02 - bottomMargin) / frameHeight);
      rangeHigh = rangeLow + (yHigh - rangeLow) / usableFraction;
    }

    TH1D* frame = new TH1D(frameName.Data(), "", 100, xLow, xHigh);
    frame->SetDirectory(nullptr);
    ownedObjects.push_back(frame);
    frame->SetMinimum(rangeLow);
    frame->SetMaximum(rangeHigh);
    frame->GetXaxis()->SetTitle(xTitle.Data());
    frame->GetYaxis()->SetTitle(yTitle.Data());
    frame->GetXaxis()->SetTitleSize(0.050);
    frame->GetXaxis()->CenterTitle();
    frame->GetYaxis()->SetTitleSize(0.050);
    frame->GetXaxis()->SetLabelSize(0.045);
    frame->GetYaxis()->SetLabelSize(0.045);
    frame->GetYaxis()->SetTitleOffset(1.35);
    frame->Draw("AXIS");

    for(const double boundary : boundaryLines)
    {
      TLine* line = new TLine(boundary, rangeLow, boundary, rangeHigh);
      ownedObjects.push_back(line);
      line->SetLineColor(kGray);
      line->SetLineStyle(3);
      line->Draw();
    }
    if(!isSigma && rangeLow < 0.0 && rangeHigh > 0.0)
    {
      TLine* zeroLine = new TLine(xLow, 0.0, xHigh, 0.0);
      ownedObjects.push_back(zeroLine);
      zeroLine->SetLineColor(kGray + 2);
      zeroLine->SetLineStyle(2);
      zeroLine->Draw();
    }

    std::vector<TF1*> fits(numberOfInputs, nullptr);
    std::vector<TString> fitTexts(numberOfInputs);
    for(int input = 0; input < numberOfInputs; ++input)
    {
      TGraphErrors* graph = graphs[input];
      if(graph->GetN() > 0)
      {
        graph->Draw("P");
      }
      if(!hasFit)
      {
        continue;
      }

      // Off-scale chips are left out so they cannot pull the fit.
      TGraphErrors* fitted = new TGraphErrors();
      ownedObjects.push_back(fitted);
      for(int point = 0; point < graph->GetN(); ++point)
      {
        const double value = graph->GetY()[point];
        const double error = graph->GetEY()[point];
        if(in_display_range(value, error))
        {
          const int fittedPoint = fitted->GetN();
          fitted->SetPoint(fittedPoint, graph->GetX()[point], value);
          fitted->SetPointError(fittedPoint, 0.0, error);
        }
      }
      // One more point than parameters, so ndf >= 1.
      const char* formula = "pol0";
      int minimumPoints = 2;
      if(summaryFit == SummaryFit::line)
      {
        formula = "pol1";
        minimumPoints = 3;
      }
      else if(summaryFit == SummaryFit::sinusoid)
      {
        formula = "1 ++ sin(x) ++ cos(x)";
        minimumPoints = 4;
      }
      if(fitted->GetN() < minimumPoints)
      {
        fitTexts[input] = Form("fit: too few %s", chipWord.Data());
        continue;
      }

      // All forms are linear in the parameters, so ROOT fits them exactly.
      TF1* fit = new TF1(
        Form("f_%s_in%d", frameName.Data(), input), formula, xLow, xHigh);
      ownedObjects.push_back(fit);
      fitted->Fit(fit, "QN0");
      fit->SetLineColor(input_color(input));
      fit->SetLineWidth(2);
      fit->Draw("same");
      fits[input] = fit;

      // Parameters in um (the graphs are in mm); pol0 has only p0.
      const double p0 = 1e3 * fit->GetParameter(0);
      const double p0Error = 1e3 * fit->GetParError(0);
      if(summaryFit == SummaryFit::constant)
      {
        fitTexts[input] = Form(
          "const %.2f#pm%.2f #mum, #chi^{2}/ndf %.0f/%d",
          p0, p0Error, fit->GetChisquare(), fit->GetNDF());
      }
      else if(summaryFit == SummaryFit::line)
      {
        fitTexts[input] = Form(
          "%.2f#pm%.2f %+.3g(#pm%.2g) #times ID #mum, #chi^{2}/ndf %.0f/%d",
          p0, p0Error, 1e3 * fit->GetParameter(1), 1e3 * fit->GetParError(1),
          fit->GetChisquare(), fit->GetNDF());
      }
      else
      {
        fitTexts[input] = Form(
          "%.2f#pm%.2f %+.2f(#pm%.2f) sin#phi %+.2f(#pm%.2f) cos#phi #mum, "
          "#chi^{2}/ndf %.0f/%d",
          p0, p0Error, 1e3 * fit->GetParameter(1), 1e3 * fit->GetParError(1),
          1e3 * fit->GetParameter(2), 1e3 * fit->GetParError(2),
          fit->GetChisquare(), fit->GetNDF());
      }
    }

    TLegend* legend = new TLegend(
      leftMargin + 0.02, legendBottom, 0.95, legendTop);
    ownedObjects.push_back(legend);
    legend->SetBorderSize(0);
    legend->SetFillStyle(0);
    legend->SetTextFont(42);
    legend->SetTextSize(hasFit ? 0.034 : 0.042);
    // Fit window of every point (fit_core_gaussian).
    legend->SetHeader(Form(
      "Core Gaussian: seed #pm%g mm, refit #mu #pm %g#sigma",
      kDcaFitRangeMm, kCoreFitSigmaWindow));
    for(int input = 0; input < numberOfInputs; ++input)
    {
      const TGraphErrors* graph = graphs[input];
      int offScale = 0;
      for(int point = 0; point < graph->GetN(); ++point)
      {
        const double value = graph->GetY()[point];
        offScale += value < rangeLow || value > rangeHigh;
      }

      TString entryText = Form(
        "%s: %d of %d %s fitted", inputs[input].label.c_str(),
        graph->GetN(), populatedChips[input], chipWord.Data());
      if(offScale > 0)
      {
        entryText += Form(", %d off-scale", offScale);
      }
      legend->AddEntry(graph, entryText.Data(), "p");
      if(hasFit)
      {
        legend->AddEntry(fits[input], fitTexts[input].Data(),
                         fits[input] ? "l" : "");
      }
    }
    legend->Draw();
  }

  // One page per layer: Gaussian mean (top) and sigma (bottom) of -DCA_xy
  // (left) and -DCA_z (right) of every chip versus its stave-chip ID or, with
  // versusClusterPhi, its mean cluster phi, one graph per input. The values
  // are the per-chip fits of the CSV table. Versus cluster phi, the chips of
  // one stave (along z) sit at nearly the same phi and stack vertically.
  // Unless staveChipFit is none, every graph versus stave-chip ID gets the
  // staveChipFit fit (constant or straight line) and, versus cluster phi, the
  // mean of -DCA_xy gets a sinusoid fit.
  // Returns false (no page) if the layer has no populated stave.
  bool draw_stavechip_summary_page(
    const DetectorConfig& config,
    int layer,
    bool versusClusterPhi,
    SummaryFit staveChipFit,
    const StaveHistogramMap& histogramMap,
    const std::vector<InputSpec>& inputs,
    const TString& selectionText,
    PdfOutput& outputPdf)
  {
    int numberOfStaves = kStavesPerLayer[layer];
    bool hasLayer = false;
    for(const auto& entry : histogramMap)
    {
      if(entry.second.layer == layer)
      {
        hasLayer = true;
        numberOfStaves = std::max(numberOfStaves, entry.second.stave + 1);
      }
    }
    if(!hasLayer)
    {
      return false;
    }

    TString chipWord(config.panelLabel);
    chipWord.ToLower();
    const char* xName = versusClusterPhi ? "clusphi" : "stavechip";
    double xLow = -TMath::Pi();
    double xHigh = TMath::Pi();
    std::vector<double> boundaryLines;
    TString xTitle = "chip #phi [rad]";
    if(!versusClusterPhi)
    {
      const int numberOfIds = config.chipsPerStave * numberOfStaves;
      const char* staveWord = &config == &kInttConfig ? "ladder" : "stave";
      xLow = -0.5;
      xHigh = numberOfIds - 0.5;
      for(int boundary = config.chipsPerStave; boundary < numberOfIds;
          boundary += config.chipsPerStave)
      {
        boundaryLines.push_back(boundary - 0.5);
      }
      xTitle = Form(
        "%d #times %s + %s", config.chipsPerStave, staveWord, chipWord.Data());
    }
    const int numberOfInputs = static_cast<int>(inputs.size());
    std::vector<TObject*> ownedObjects;

    TCanvas* canvas = new TCanvas(
      Form("c_dca_%s_L%d_vs_%s", config.outputPrefix, layer, xName),
      Form("%s layer %d: -DCA fits vs %s", config.detectorName, layer,
           versusClusterPhi ? "chip phi" : "stave-chip ID"),
      1600, 1000);
    canvas->Divide(2, 2, 0.001, 0.001);
    constexpr double pageTitleHeight = 0.055;
    reserve_page_title_space(canvas, 4, pageTitleHeight);
    for(int coordinate = 0; coordinate < kNumberOfCoordinates; ++coordinate)
    {
      std::vector<TGraphErrors*> meanGraphs;
      std::vector<TGraphErrors*> sigmaGraphs;
      std::vector<int> populatedChips;
      for(int input = 0; input < numberOfInputs; ++input)
      {
        const StaveChipGraphs graphs = make_stavechip_graphs(
          config, layer, coordinate, input, versusClusterPhi, histogramMap);
        ownedObjects.push_back(graphs.mean);
        ownedObjects.push_back(graphs.sigma);
        meanGraphs.push_back(graphs.mean);
        sigmaGraphs.push_back(graphs.sigma);
        populatedChips.push_back(graphs.populatedChips);
      }

      const char* coordinateName = kCoordinateNames[coordinate];
      const char* coordinateLabel = kCoordinateLabels[coordinate];
      SummaryFit meanFit = SummaryFit::none;
      SummaryFit sigmaFit = SummaryFit::none;
      if(!versusClusterPhi)
      {
        meanFit = staveChipFit;
        sigmaFit = staveChipFit;
      }
      else if(staveChipFit != SummaryFit::none && coordinate == 0)
      {
        meanFit = SummaryFit::sinusoid;
      }
      canvas->cd(coordinate + 1);
      draw_stavechip_summary_pad(
        meanGraphs, populatedChips, inputs, false, meanFit, xLow, xHigh,
        boundaryLines, chipWord + "s",
        Form("h_frame_%s_L%d_%s_vs_%s_mean", config.outputPrefix, layer,
             coordinateName, xName),
        xTitle, Form("Gaussian #mu of %s [mm]", coordinateLabel),
        ownedObjects);
      canvas->cd(coordinate + 3);
      draw_stavechip_summary_pad(
        sigmaGraphs, populatedChips, inputs, true, sigmaFit, xLow, xHigh,
        boundaryLines, chipWord + "s",
        Form("h_frame_%s_L%d_%s_vs_%s_sigma", config.outputPrefix, layer,
             coordinateName, xName),
        xTitle, Form("Gaussian #sigma of %s [mm]", coordinateLabel),
        ownedObjects);
    }

    draw_page_title(
      canvas, pageTitleHeight,
      Form("%s layer %d: Gaussian #mu and #sigma of the track #minusDCA "
           "per %s%s%s",
           config.detectorName, layer, chipWord.Data(),
           versusClusterPhi ? " vs chip #phi" : "", selectionText.Data()));
    print_and_delete_page(
      canvas, outputPdf,
      Form("%s_dca_vs_%s_L%d.png", config.outputPrefix, xName, layer),
      ownedObjects);
    return true;
  }

  // One page per populated layer of the detector, all versus stave-chip ID
  // or, with versusClusterPhi, all versus chip phi.
  int draw_stavechip_summary_pages(
    const DetectorConfig& config,
    const StaveHistogramMap& histogramMap,
    const std::vector<InputSpec>& inputs,
    const TString& selectionText,
    bool versusClusterPhi,
    SummaryFit staveChipFit,
    PdfOutput& outputPdf)
  {
    int pages = 0;
    for(int layer = config.firstLayer;
        layer < config.firstLayer + config.numberOfLayers; ++layer)
    {
      pages += draw_stavechip_summary_page(
        config, layer, versusClusterPhi, staveChipFit, histogramMap, inputs,
        selectionText, outputPdf);
    }

    return pages;
  }

  // ---------------------------------------------------------------------------
  // MVTX z residual / DCA_z pages
  // ---------------------------------------------------------------------------

  // Expected mu(z residual) / DCA_z of the chips of plottedLayer when only the
  // chips of displacedLayer are displaced along z, for a straight-line fit in
  // (r, z) through equally precise MVTX clusters at kMvtxRadiiCm, residual =
  // cluster - fitted line. A cluster z error dz of the displaced layer moves
  // the track z at the vertex by w * dz, w = 1/3 - rbar (r_d - rbar) / S, and
  // leaves the residual (delta_pd - H_pd) * dz on the plotted layer, with the
  // hat matrix H_pd = 1/3 + (r_p - rbar)(r_d - rbar) / S of the fit
  // (S = sum (r - rbar)^2). A z shift common to the three layers gives 0.
  // Multiple scattering, INTT and the vertex fit (which absorbs part of the
  // DCA) are not included, and the local y axis is assumed along +z.
  double expected_residual_ratio(int plottedLayer, int displacedLayer)
  {
    const double meanRadius =
      (kMvtxRadiiCm[0] + kMvtxRadiiCm[1] + kMvtxRadiiCm[2]) / 3.0;
    double spread = 0.0;
    for(const double radius : kMvtxRadiiCm)
    {
      spread += (radius - meanRadius) * (radius - meanRadius);
    }
    const double plottedOffset = kMvtxRadiiCm[plottedLayer] - meanRadius;
    const double displacedOffset = kMvtxRadiiCm[displacedLayer] - meanRadius;
    const double hat = 1.0 / 3.0 + plottedOffset * displacedOffset / spread;
    const double residual =
      (plottedLayer == displacedLayer ? 1.0 : 0.0) - hat;
    const double weight = 1.0 / 3.0 - meanRadius * displacedOffset / spread;
    return residual / weight;
  }

  struct ReferenceLine
  {
    double value = 0.0;
    TString text;
    int color = kGray + 2;
    int style = 1;
  };

  // Ratio of a whole line from the vertex shifted (0) and of only one MVTX
  // layer displaced, for the chips of `layer`.
  std::vector<ReferenceLine> residual_ratio_references(int layer)
  {
    std::vector<ReferenceLine> references;
    references.push_back({0.0, "whole line shifted: 0", kGray + 2, 1});
    const int colors[] = {kAzure + 2, kPink + 7, kSpring - 6};
    const int styles[] = {2, 10, 3};
    for(int displaced = 0; displaced < kMvtxConfig.numberOfLayers; ++displaced)
    {
      const double value = expected_residual_ratio(layer, displaced);
      references.push_back(
        {value, Form("only L%d displaced: %+.2f", displaced, value),
         colors[displaced], styles[displaced]});
    }
    return references;
  }

  // Per-chip graphs of one MVTX layer and input: mu of the z residual, DCA_z
  // (= -mu of -DCA_z) and their ratio versus stave-chip ID, and the z
  // residual versus DCA_z. Chips need fit_status ok for the values involved;
  // the ratio also needs |DCA_z| >= minimumDcaSignificance * its error, and
  // the scatter both values inside the display ranges.
  struct ResidualRatioGraphs
  {
    TGraphErrors* residual = nullptr;
    TGraphErrors* dca = nullptr;
    TGraphErrors* ratio = nullptr;
    TGraphErrors* scatter = nullptr;
    int residualPopulated = 0;
    int dcaPopulated = 0;
    int bothFitted = 0;
    int outsideScatterRange = 0;
  };

  ResidualRatioGraphs make_residual_ratio_graphs(
    int layer,
    int inputIndex,
    double minimumDcaSignificance,
    const StaveHistogramMap& histogramMap)
  {
    ResidualRatioGraphs graphs;
    graphs.residual = new TGraphErrors();
    graphs.dca = new TGraphErrors();
    graphs.ratio = new TGraphErrors();
    graphs.scatter = new TGraphErrors();
    graphs.residual->SetName(Form("g_resz_mvtx_L%d_in%d", layer, inputIndex));
    graphs.dca->SetName(Form("g_dcaz_mvtx_L%d_in%d", layer, inputIndex));
    graphs.ratio->SetName(Form("g_resz_over_dcaz_mvtx_L%d_in%d", layer, inputIndex));
    graphs.scatter->SetName(Form("g_resz_vs_dcaz_mvtx_L%d_in%d", layer, inputIndex));
    for(const auto& entry : histogramMap)
    {
      const StaveHistograms& histograms = entry.second;
      if(histograms.layer != layer || histograms.residualZFits.empty())
      {
        continue;
      }

      for(int chip = 0; chip < kMvtxConfig.chipsPerStave; ++chip)
      {
        const double x = kMvtxConfig.chipsPerStave * histograms.stave + chip;
        const ChipFitResult& residualFit =
          histograms.residualZFits[inputIndex][chip];
        const ChipFitResult& dcaFit = histograms.fits[1][inputIndex][chip];
        graphs.residualPopulated +=
          histograms.residualZHistograms[inputIndex][chip]->GetEntries() > 0.0;
        graphs.dcaPopulated +=
          histograms.histograms[1][inputIndex][chip]->GetEntries() > 0.0;

        const bool residualOk = residualFit.status == "ok";
        const bool dcaOk = dcaFit.status == "ok";
        const double dcaZ = -dcaFit.mean;
        if(residualOk)
        {
          const int point = graphs.residual->GetN();
          graphs.residual->SetPoint(point, x, residualFit.mean);
          graphs.residual->SetPointError(point, 0.0, residualFit.meanError);
        }
        if(dcaOk)
        {
          const int point = graphs.dca->GetN();
          graphs.dca->SetPoint(point, x, dcaZ);
          graphs.dca->SetPointError(point, 0.0, dcaFit.meanError);
        }
        if(!residualOk || !dcaOk)
        {
          continue;
        }

        ++graphs.bothFitted;
        if(dcaZ != 0.0 &&
           std::fabs(dcaZ) >= minimumDcaSignificance * dcaFit.meanError)
        {
          const double ratio = residualFit.mean / dcaZ;
          const double ratioError =
            std::hypot(residualFit.meanError, ratio * dcaFit.meanError) /
            std::fabs(dcaZ);
          const int point = graphs.ratio->GetN();
          graphs.ratio->SetPoint(point, x, ratio);
          graphs.ratio->SetPointError(point, 0.0, ratioError);
        }
        if(std::fabs(dcaZ) <= kDcaDisplayRangeMm &&
           std::fabs(residualFit.mean) <= kResidualZDisplayRangeMm)
        {
          const int point = graphs.scatter->GetN();
          graphs.scatter->SetPoint(point, dcaZ, residualFit.mean);
          graphs.scatter->SetPointError(
            point, dcaFit.meanError, residualFit.meanError);
        }
        else
        {
          ++graphs.outsideScatterRange;
        }
      }
    }

    for(TGraphErrors* graph :
        {graphs.residual, graphs.dca, graphs.ratio, graphs.scatter})
    {
      graph->SetLineColor(input_color(inputIndex));
      graph->SetMarkerColor(input_color(inputIndex));
      graph->SetMarkerStyle(input_marker_style(inputIndex));
      graph->SetMarkerSize(0.8);
    }

    return graphs;
  }

  void style_frame(TH1D* frame, const TString& xTitle, const TString& yTitle)
  {
    frame->SetDirectory(nullptr);
    frame->GetXaxis()->SetTitle(xTitle.Data());
    frame->GetYaxis()->SetTitle(yTitle.Data());
    frame->GetXaxis()->SetTitleSize(0.055);
    frame->GetXaxis()->CenterTitle();
    frame->GetYaxis()->SetTitleSize(0.055);
    frame->GetXaxis()->SetLabelSize(0.050);
    frame->GetYaxis()->SetLabelSize(0.050);
    frame->GetYaxis()->SetTitleOffset(0.8);
  }

  // Adds the input entries (with the number of points outside [yLow, yHigh]
  // appended) and the reference lines to a two-column legend.
  void fill_legend(
    TLegend* legend,
    const std::vector<TGraphErrors*>& graphs,
    const std::vector<TString>& entryTexts,
    const std::vector<TLine*>& referenceLines,
    const std::vector<ReferenceLine>& references,
    double yLow,
    double yHigh)
  {
    legend->SetBorderSize(0);
    legend->SetFillStyle(0);
    legend->SetTextFont(42);
    legend->SetTextSize(0.050);
    legend->SetNColumns(2);
    for(size_t input = 0; input < graphs.size(); ++input)
    {
      int offScale = 0;
      for(int point = 0; point < graphs[input]->GetN(); ++point)
      {
        const double value = graphs[input]->GetY()[point];
        offScale += value < yLow || value > yHigh;
      }
      TString text = entryTexts[input];
      if(offScale > 0)
      {
        text += Form(", %d off-scale", offScale);
      }
      legend->AddEntry(graphs[input], text.Data(), "p");
    }
    for(size_t index = 0; index < referenceLines.size(); ++index)
    {
      legend->AddEntry(referenceLines[index], references[index].text.Data(), "l");
    }
  }

  // Draws the per-input graphs versus stave-chip ID in the current pad, with
  // dotted stave boundaries, a dashed zero line (no references) or the
  // reference lines. The y range covers 0, the points with |value| <= yClip
  // and the references with |value| <= referenceRange; references outside
  // the range are not drawn and marked off-scale in the legend. The top part
  // of the pad is kept for the legend.
  void draw_stavechip_values_pad(
    const std::vector<TGraphErrors*>& graphs,
    const std::vector<TString>& entryTexts,
    std::vector<ReferenceLine> references,
    double xLow,
    double xHigh,
    const std::vector<double>& boundaryLines,
    double yClip,
    double referenceRange,
    const TString& header,
    const TString& frameName,
    const TString& xTitle,
    const TString& yTitle,
    std::vector<TObject*>& ownedObjects)
  {
    constexpr double leftMargin = 0.10;
    constexpr double bottomMargin = 0.16;
    constexpr double topMargin = 0.03;
    gPad->SetLeftMargin(leftMargin);
    gPad->SetRightMargin(0.02);
    gPad->SetBottomMargin(bottomMargin);
    gPad->SetTopMargin(topMargin);

    double low = 0.0;
    double high = 0.0;
    for(const TGraphErrors* graph : graphs)
    {
      for(int point = 0; point < graph->GetN(); ++point)
      {
        const double value = graph->GetY()[point];
        const double error = graph->GetEY()[point];
        if(std::fabs(value) > yClip || !std::isfinite(error) || error > yClip)
        {
          continue;
        }
        low = std::min(low, value - error);
        high = std::max(high, value + error);
      }
    }
    for(const ReferenceLine& reference : references)
    {
      if(std::fabs(reference.value) <= referenceRange)
      {
        low = std::min(low, reference.value);
        high = std::max(high, reference.value);
      }
    }
    const double span = std::max(high - low, 1e-4);
    const double yLow = low - 0.1 * span;
    const double yHigh = high + 0.1 * span;
    // Keep the top ~40% of the frame free for the legend.
    const double rangeHigh = yLow + (yHigh - yLow) / 0.6;
    for(ReferenceLine& reference : references)
    {
      if(reference.value < yLow || reference.value > rangeHigh)
      {
        reference.text += " (off-scale)";
      }
    }

    TH1D* frame = new TH1D(frameName.Data(), "", 100, xLow, xHigh);
    ownedObjects.push_back(frame);
    style_frame(frame, xTitle, yTitle);
    frame->SetMinimum(yLow);
    frame->SetMaximum(rangeHigh);
    frame->Draw("AXIS");

    for(const double boundary : boundaryLines)
    {
      TLine* line = new TLine(boundary, yLow, boundary, rangeHigh);
      ownedObjects.push_back(line);
      line->SetLineColor(kGray);
      line->SetLineStyle(3);
      line->Draw();
    }
    if(references.empty() && yLow < 0.0 && rangeHigh > 0.0)
    {
      TLine* zeroLine = new TLine(xLow, 0.0, xHigh, 0.0);
      ownedObjects.push_back(zeroLine);
      zeroLine->SetLineColor(kGray + 2);
      zeroLine->SetLineStyle(2);
      zeroLine->Draw();
    }
    std::vector<TLine*> referenceLines;
    for(const ReferenceLine& reference : references)
    {
      TLine* line = new TLine(xLow, reference.value, xHigh, reference.value);
      ownedObjects.push_back(line);
      line->SetLineColor(reference.color);
      line->SetLineStyle(reference.style);
      line->SetLineWidth(2);
      if(reference.value >= yLow && reference.value <= rangeHigh)
      {
        line->Draw();
      }
      referenceLines.push_back(line);
    }

    for(TGraphErrors* graph : graphs)
    {
      if(graph->GetN() > 0)
      {
        graph->Draw("P");
      }
    }

    const int rows = 1 + static_cast<int>(
      (graphs.size() + references.size() + 1) / 2);
    const double legendTop = 1.0 - topMargin - 0.01;
    TLegend* legend = new TLegend(
      leftMargin + 0.01, legendTop - 0.075 * rows, 0.97, legendTop);
    ownedObjects.push_back(legend);
    legend->SetHeader(header.Data());
    fill_legend(legend, graphs, entryTexts, referenceLines, references,
                yLow, rangeHigh);
    legend->Draw();
  }

  // One page per MVTX layer, three pads versus stave-chip ID: mu of the z
  // residual, DCA_z, and the ratio z residual / DCA_z with the expected
  // values of residual_ratio_references. Returns false (no page) if the
  // layer has no populated stave.
  bool draw_residual_ratio_page(
    int layer,
    const StaveHistogramMap& histogramMap,
    const std::vector<InputSpec>& inputs,
    const TString& selectionText,
    double minimumDcaSignificance,
    PdfOutput& outputPdf)
  {
    int numberOfStaves = kStavesPerLayer[layer];
    bool hasLayer = false;
    for(const auto& entry : histogramMap)
    {
      if(entry.second.layer == layer)
      {
        hasLayer = true;
        numberOfStaves = std::max(numberOfStaves, entry.second.stave + 1);
      }
    }
    if(!hasLayer)
    {
      return false;
    }

    const int numberOfIds = kMvtxConfig.chipsPerStave * numberOfStaves;
    const double xLow = -0.5;
    const double xHigh = numberOfIds - 0.5;
    std::vector<double> boundaryLines;
    for(int boundary = kMvtxConfig.chipsPerStave; boundary < numberOfIds;
        boundary += kMvtxConfig.chipsPerStave)
    {
      boundaryLines.push_back(boundary - 0.5);
    }
    const TString xTitle = "9 #times stave + chip";

    std::vector<TObject*> ownedObjects;
    std::vector<TGraphErrors*> residualGraphs;
    std::vector<TGraphErrors*> dcaGraphs;
    std::vector<TGraphErrors*> ratioGraphs;
    std::vector<TString> residualTexts;
    std::vector<TString> dcaTexts;
    std::vector<TString> ratioTexts;
    for(size_t input = 0; input < inputs.size(); ++input)
    {
      const ResidualRatioGraphs graphs = make_residual_ratio_graphs(
        layer, static_cast<int>(input), minimumDcaSignificance, histogramMap);
      for(TObject* graph :
          {graphs.residual, graphs.dca, graphs.ratio, graphs.scatter})
      {
        ownedObjects.push_back(graph);
      }
      residualGraphs.push_back(graphs.residual);
      dcaGraphs.push_back(graphs.dca);
      ratioGraphs.push_back(graphs.ratio);
      const char* label = inputs[input].label.c_str();
      residualTexts.push_back(Form(
        "%s: %d of %d chips fitted", label, graphs.residual->GetN(),
        graphs.residualPopulated));
      dcaTexts.push_back(Form(
        "%s: %d of %d chips fitted", label, graphs.dca->GetN(),
        graphs.dcaPopulated));
      ratioTexts.push_back(Form(
        "%s: %d of %d chips with |DCA_{z}| #geq %g#sigma", label,
        graphs.ratio->GetN(), graphs.bothFitted, minimumDcaSignificance));
    }

    TCanvas* canvas = new TCanvas(
      Form("c_resz_dcaz_mvtx_L%d", layer),
      Form("MVTX layer %d: z residual / DCA_z", layer), 1600, 1500);
    canvas->Divide(1, 3, 0.001, 0.001);
    constexpr double pageTitleHeight = 0.04;
    reserve_page_title_space(canvas, 3, pageTitleHeight);

    canvas->cd(1);
    draw_stavechip_values_pad(
      residualGraphs, residualTexts, {}, xLow, xHigh, boundaryLines,
      kResidualZDisplayRangeMm, 0.0,
      Form("z residual: core Gaussian, seed #pm%g mm", kResidualZFitRangeMm),
      Form("h_frame_resz_mvtx_L%d", layer), xTitle,
      "#mu of z residual [mm]", ownedObjects);
    canvas->cd(2);
    draw_stavechip_values_pad(
      dcaGraphs, dcaTexts, {}, xLow, xHigh, boundaryLines,
      kDcaDisplayRangeMm, 0.0,
      Form("DCA_{z} = #minus#mu of #minusDCA_{z}: core Gaussian, seed #pm%g mm",
           kDcaFitRangeMm),
      Form("h_frame_dcaz_mvtx_L%d", layer), xTitle, "DCA_{z} [mm]",
      ownedObjects);
    canvas->cd(3);
    draw_stavechip_values_pad(
      ratioGraphs, ratioTexts, residual_ratio_references(layer), xLow, xHigh,
      boundaryLines, kRatioDisplayRange, kRatioReferenceRange,
      "Expected (straight line in r-z, local y along +z, no vertex absorption):",
      Form("h_frame_ratio_mvtx_L%d", layer), xTitle,
      "z residual / DCA_{z}", ownedObjects);

    draw_page_title(
      canvas, pageTitleHeight,
      Form("MVTX layer %d: z residual, DCA_{z} and their ratio per chip%s",
           layer, selectionText.Data()));
    print_and_delete_page(
      canvas, outputPdf, Form("mvtx_resz_dcaz_ratio_L%d.png", layer),
      ownedObjects);
    return true;
  }

  // Endpoints of the line y = slope * x inside [-xMax, xMax] x [yLow, yHigh].
  TLine* make_clipped_slope_line(
    double slope, double xMax, double yLow, double yHigh)
  {
    double x1 = -xMax;
    double x2 = xMax;
    if(slope != 0.0)
    {
      const double xAtLow = yLow / slope;
      const double xAtHigh = yHigh / slope;
      x1 = std::max(x1, std::min(xAtLow, xAtHigh));
      x2 = std::min(x2, std::max(xAtLow, xAtHigh));
    }
    return new TLine(x1, slope * x1, x2, slope * x2);
  }

  // One page, one pad per MVTX layer: mu of the z residual versus DCA_z of
  // every chip with both fits ok and both values in the display ranges, with
  // a straight-line fit per input (slope = average ratio of the layer, no
  // division by small DCA_z) and the expected ratios as lines through 0.
  // Returns false (no page) if no MVTX layer is populated.
  bool draw_residual_scatter_page(
    const StaveHistogramMap& histogramMap,
    const std::vector<InputSpec>& inputs,
    const TString& selectionText,
    PdfOutput& outputPdf)
  {
    bool anyMvtx = false;
    for(const auto& entry : histogramMap)
    {
      anyMvtx = anyMvtx || !entry.second.residualZFits.empty();
    }
    if(!anyMvtx)
    {
      return false;
    }

    std::vector<TObject*> ownedObjects;
    TCanvas* canvas = new TCanvas(
      "c_resz_vs_dcaz_mvtx", "MVTX: z residual vs DCA_z", 1800, 950);
    canvas->Divide(kMvtxConfig.numberOfLayers, 1, 0.001, 0.001);
    constexpr double pageTitleHeight = 0.05;
    reserve_page_title_space(canvas, kMvtxConfig.numberOfLayers, pageTitleHeight);

    for(int layer = 0; layer < kMvtxConfig.numberOfLayers; ++layer)
    {
      canvas->cd(layer + 1);
      gPad->SetLeftMargin(0.21);
      gPad->SetRightMargin(0.03);
      gPad->SetBottomMargin(0.13);
      gPad->SetTopMargin(0.03);

      std::vector<TGraphErrors*> graphs;
      std::vector<int> outside;
      double xMax = 0.0;
      double yMax = 0.0;
      for(size_t input = 0; input < inputs.size(); ++input)
      {
        const ResidualRatioGraphs ratioGraphs = make_residual_ratio_graphs(
          layer, static_cast<int>(input), 0.0, histogramMap);
        for(TObject* graph : {ratioGraphs.residual, ratioGraphs.dca,
                              ratioGraphs.ratio, ratioGraphs.scatter})
        {
          ownedObjects.push_back(graph);
        }
        TGraphErrors* graph = ratioGraphs.scatter;
        graphs.push_back(graph);
        outside.push_back(ratioGraphs.outsideScatterRange);
        for(int point = 0; point < graph->GetN(); ++point)
        {
          xMax = std::max(xMax, std::fabs(graph->GetX()[point]) + graph->GetEX()[point]);
          yMax = std::max(yMax, std::fabs(graph->GetY()[point]) + graph->GetEY()[point]);
        }
      }
      xMax = 1.15 * std::max(xMax, 0.002);
      yMax = 1.15 * std::max(yMax, 0.0005);
      const double yLow = -yMax;
      const double yHigh = yLow + 2.0 * yMax / 0.5;  // top half for the legend

      TH1D* frame = new TH1D(
        Form("h_frame_resz_vs_dcaz_mvtx_L%d", layer), "", 100, -xMax, xMax);
      ownedObjects.push_back(frame);
      style_frame(frame, "DCA_{z} [mm]", "#mu of z residual [mm]");
      frame->GetXaxis()->SetTitleSize(0.045);
      frame->GetYaxis()->SetTitleSize(0.045);
      frame->GetXaxis()->SetLabelSize(0.038);
      frame->GetYaxis()->SetLabelSize(0.038);
      frame->GetYaxis()->SetTitleOffset(2.1);
      frame->SetMinimum(yLow);
      frame->SetMaximum(yHigh);
      frame->Draw("AXIS");

      const std::vector<ReferenceLine> references =
        residual_ratio_references(layer);
      std::vector<TLine*> referenceLines;
      for(const ReferenceLine& reference : references)
      {
        TLine* line =
          make_clipped_slope_line(reference.value, xMax, yLow, yHigh);
        ownedObjects.push_back(line);
        line->SetLineColor(reference.color);
        line->SetLineStyle(reference.style);
        line->SetLineWidth(2);
        line->Draw();
        referenceLines.push_back(line);
      }

      std::vector<TString> entryTexts;
      for(size_t input = 0; input < graphs.size(); ++input)
      {
        TGraphErrors* graph = graphs[input];
        TString text = Form("%s: ", inputs[input].label.c_str());
        if(graph->GetN() >= 3)
        {
          TF1* fit = new TF1(
            Form("f_resz_vs_dcaz_mvtx_L%d_in%zu", layer, input), "pol1",
            -xMax, xMax);
          ownedObjects.push_back(fit);
          graph->Fit(fit, "QN0");
          fit->SetLineColor(input_color(static_cast<int>(input)));
          fit->SetLineWidth(2);
          fit->Draw("same");
          text += Form("slope %.3f #pm %.2g", fit->GetParameter(1),
                       fit->GetParError(1));
        }
        else
        {
          text += "too few chips to fit";
        }
        text += Form(" (%d chips", graph->GetN());
        if(outside[input] > 0)
        {
          text += Form(", %d outside", outside[input]);
        }
        text += ")";
        entryTexts.push_back(text);
        if(graph->GetN() > 0)
        {
          graph->Draw("P");
        }
      }

      TLegend* legend = new TLegend(0.22, 0.58, 0.97, 0.96);
      ownedObjects.push_back(legend);
      legend->SetHeader(Form("MVTX layer %d", layer));
      fill_legend(legend, graphs, entryTexts, referenceLines, references,
                  yLow, yHigh);
      legend->SetNColumns(1);
      legend->SetTextSize(0.030);
      legend->Draw();
    }

    draw_page_title(
      canvas, pageTitleHeight,
      Form("MVTX: #mu of z residual vs DCA_{z} per chip, line fit per input "
           "(lines through 0: expected ratios)%s",
           selectionText.Data()));
    print_and_delete_page(
      canvas, outputPdf, "mvtx_resz_vs_dcaz.png", ownedObjects);
    return true;
  }

  // One page per detector, one pad per layer: chip phi (circular mean of the
  // cluster phi of the selected rows of every input) versus stave-chip ID,
  // with the circular RMS sqrt(-2 ln R) of the cluster phi as error bar
  // (R = |sum of exp(i phi)| / N), i.e. the phi range each chip covers. The
  // chips of one stave share a phi, so the graph is a staircase that jumps
  // at phi = +-pi. Returns false (no page) if no chip has a cluster phi.
  bool draw_chip_phi_page(
    const DetectorConfig& config,
    const StaveHistogramMap& histogramMap,
    const TString& selectionText,
    PdfOutput& outputPdf)
  {
    const int columns = config.numberOfLayers <= 3 ? config.numberOfLayers : 2;
    const int rows = (config.numberOfLayers + columns - 1) / columns;
    std::vector<TObject*> ownedObjects;
    // MVTX 1800x1000, INTT 1600x1000: wider than tall for the landscape PDF,
    // and 1000 high so the page title has the size of the other pages.
    TCanvas* canvas = new TCanvas(
      Form("c_%s_chipphi_vs_stavechip", config.outputPrefix),
      Form("%s: chip phi vs stave-chip ID", config.detectorName),
      rows == 1 ? 600 * columns : 800 * columns, 1000);
    canvas->Divide(columns, rows, 0.001, 0.001);
    constexpr double pageTitleHeight = 0.055;
    reserve_page_title_space(canvas, columns * rows, pageTitleHeight);

    const char* staveWord = &config == &kInttConfig ? "ladder" : "stave";
    TString chipWord(config.panelLabel);
    chipWord.ToLower();
    int chipsWithPhi = 0;
    for(int layerIndex = 0; layerIndex < config.numberOfLayers; ++layerIndex)
    {
      const int layer = config.firstLayer + layerIndex;
      int numberOfStaves = kStavesPerLayer[layer];
      TGraphErrors* graph = new TGraphErrors();
      graph->SetName(Form("g_%s_L%d_chipphi", config.outputPrefix, layer));
      ownedObjects.push_back(graph);
      for(const auto& entry : histogramMap)
      {
        const StaveHistograms& histograms = entry.second;
        if(histograms.layer != layer)
        {
          continue;
        }

        numberOfStaves = std::max(numberOfStaves, histograms.stave + 1);
        for(int chip = 0; chip < config.chipsPerStave; ++chip)
        {
          const double count = histograms.clusterPhiCounts[chip];
          if(count <= 0.0)
          {
            continue;
          }

          const double cosSum = histograms.clusterPhiCosSums[chip];
          const double sinSum = histograms.clusterPhiSinSums[chip];
          const double resultantLength =
            std::min(1.0, std::hypot(cosSum, sinSum) / count);
          const int point = graph->GetN();
          graph->SetPoint(
            point, config.chipsPerStave * histograms.stave + chip,
            std::atan2(sinSum, cosSum));
          graph->SetPointError(
            point, 0.0,
            resultantLength > 0.0
              ? std::sqrt(-2.0 * std::log(resultantLength))
              : TMath::Pi());
        }
      }
      chipsWithPhi += graph->GetN();
      graph->SetMarkerStyle(20);
      graph->SetMarkerSize(0.6);
      graph->SetMarkerColor(kBlack);
      graph->SetLineColor(kBlack);

      canvas->cd(layerIndex + 1);
      constexpr double leftMargin = 0.14;
      gPad->SetLeftMargin(leftMargin);
      gPad->SetRightMargin(0.03);
      gPad->SetBottomMargin(0.17);
      gPad->SetTopMargin(0.09);
      gPad->SetGridy(1);
      const int numberOfIds = config.chipsPerStave * numberOfStaves;
      constexpr double phiAxisMax = 3.5;
      TH1D* frame = new TH1D(
        Form("h_frame_%s_L%d_chipphi", config.outputPrefix, layer), "",
        100, -0.5, numberOfIds - 0.5);
      frame->SetDirectory(nullptr);
      ownedObjects.push_back(frame);
      frame->SetMinimum(-phiAxisMax);
      frame->SetMaximum(phiAxisMax);
      frame->GetXaxis()->SetTitle(Form(
        "%d #times %s + %s", config.chipsPerStave, staveWord,
        chipWord.Data()));
      frame->GetYaxis()->SetTitle("chip #phi [rad]");
      frame->GetXaxis()->SetTitleSize(0.050);
      frame->GetXaxis()->CenterTitle();
      frame->GetYaxis()->SetTitleSize(0.050);
      frame->GetXaxis()->SetLabelSize(0.045);
      frame->GetYaxis()->SetLabelSize(0.045);
      frame->GetYaxis()->SetTitleOffset(1.2);
      frame->Draw("AXIS");

      for(int boundary = config.chipsPerStave; boundary < numberOfIds;
          boundary += config.chipsPerStave)
      {
        TLine* line = new TLine(
          boundary - 0.5, -phiAxisMax, boundary - 0.5, phiAxisMax);
        ownedObjects.push_back(line);
        line->SetLineColor(kGray);
        line->SetLineStyle(3);
        line->Draw();
      }
      if(graph->GetN() > 0)
      {
        graph->Draw("P");
      }

      TLatex label;
      label.SetNDC();
      label.SetTextFont(42);
      label.SetTextSize(0.050);
      label.DrawLatex(
        leftMargin + 0.01, 0.93,
        Form("%s layer %d: %d %ss", config.detectorName, layer,
             graph->GetN(), chipWord.Data()));
    }

    if(chipsWithPhi == 0)
    {
      delete canvas;
      for(auto object = ownedObjects.rbegin(); object != ownedObjects.rend();
          ++object)
      {
        delete *object;
      }
      return false;
    }

    draw_page_title(
      canvas, pageTitleHeight,
      Form("%s: chip #phi vs stave-chip ID (mean cluster #phi per %s, bar: "
           "its spread)%s",
           config.detectorName, chipWord.Data(), selectionText.Data()));
    print_and_delete_page(
      canvas, outputPdf,
      Form("%s_chipphi_vs_stavechip.png", config.outputPrefix),
      ownedObjects);
    return true;
  }
}

// Fills, fits and writes the track-to-vertex residuals (-DCA_xy, -DCA_z) of
// the tracks through every populated MVTX chip and INTT sensor.
//
// Inputs: comma-separated file lists with optional legend labels, e.g.
//   "ntuple_list1.txt:Iter0,ntuple_list2.txt:Iter1"
// Each file list must contain one ROOT filename per line. The ntp tree must
// have the per-cluster track columns written by MakeMilleFiles
// (residual_vertexX/Y = -DCA_xy/-DCA_z in mm, track_pt, track_crossing);
// residual_vertexX/Y are NaN (and the rows skipped) when MakeMilleFiles ran
// without the event vertex. The cluster phi (clusphi) is optional.
//
// Selection: both -DCA components finite, track pT >= minimumPtGeV and, with
//   excludeUnknownCrossing, track_crossing != SHRT_MAX (tracks without a
//   known bunch crossing pass the vertex crossing match trivially).
// Each ntp row fills its track's -DCA into the histograms of its chip, so a
//   track enters once per chip it crosses (about five chips per track) and
//   the chips' distributions are correlated.
// writeToCsv: writes dca_fits_<label>.csv per input into csvOutputDirectory,
//   one row per (layer, stave, chip, coord) with coord dca_xy or dca_z and
//   the columns of residual_fits_<label>.csv (fit_model gaus; box columns
//   NaN). Histograms are filled over +-2 mm; n_underflow and n_overflow
//   count the entries outside, and a distribution with more than half of
//   its entries outside gets fit_status out_of_range. trunc_mean_mm is the
//   unbinned mean inside +-2 mm.
// makePlots: writes <outputPdf without .pdf>_<label1>_<label2>....pdf (or
//   <outputPdf without .pdf>_longfilelist.pdf for names over 200
//   characters; PNG names carry no labels) with two pages per populated stave
//   (-DCA_xy and -DCA_z), displayed over +-0.15 mm. Each panel overlays one
//   curve per input. With includeGaussianFit the legend shows the same fit
//   that is written to the CSV.
// savePngs: with makePlots, also writes every PDF page as a PNG into the
//   working directory (names carry no labels, so a later run overwrites
//   them). Off by default: only the PDF is written.
// drawStaveChipSummary: with makePlots, starts the PDF with summary pages of
//   the Gaussian mu and sigma (the CSV fit) of -DCA_xy and -DCA_z of every
//   chip, one graph per input; chips without fit_status ok have no point.
//   In order:
//   1. versus stave-chip ID (9*stave + chip for MVTX, 4*ladder + sensor for
//      INTT), one page per populated layer;
//   2. chip phi (circular mean of the cluster phi, clusphi) versus stave-chip
//      ID, one page per detector (bars: circular RMS of the cluster phi);
//   3. versus chip phi, one page per populated layer.
//   2 and 3 need the optional clusphi branch.
//   With fitSummary, every mu and sigma graph of 1 gets a staveChipFit fit,
//   "pol0" (constant: the weighted mean of the chips, chi2/ndf measures their
//   scatter) or "pol1" (straight line), one per layer and input over all
//   staves, and the mu of -DCA_xy of 3 gets p0 + p1 sin(phi) + p2 cos(phi),
//   all weighted by the Gaussian fit errors of the chips inside the display
//   range. The parameters (um) and chi2/ndf are listed in the legend. Chips
//   share tracks, so the fit errors and chi2 are optimistic; the DCA is to
//   the event vertex, so a common translation of the silicon or of the beam
//   spot gives no sinusoid.
// drawResidualRatio: with makePlots, adds after the summary pages, per MVTX
//   layer, the Gaussian mu of the cluster z residual (residualY, fitted as in
//   plot_fits_overlaid_stavechip.C), DCA_z (= -mu of -DCA_z) and their ratio
//   versus stave-chip ID, then one page of mu(z residual) versus DCA_z with
//   a straight-line fit per layer and input. The ratio is drawn only for
//   chips with |DCA_z| >= minimumDcaSignificance times its fit error. Both
//   show the expected ratios: 0 for a whole line from the vertex shifted in
//   z, and the values for only one layer displaced (expected_residual_ratio).
//   z residual and DCA_z come from the same rows, i.e. the same tracks.
void plot_fits_overlaid_dcadistsfromstavechips(
  const char* listfiles = "filelist1.txt:Iter1, filelist2.txt:Iter2, filelist3.txt:Iter3",
  const char* outputPdf = "silicon_stavechip_dca_fits.pdf",
  bool includeGaussianFit = false,
  bool drawHistogramsAsMarkers = true,
  bool normalizeToUnitArea = true,
  double minimumPtGeV = 0.0,
  bool excludeUnknownCrossing = true,
  bool writeToCsv = true,
  const char* csvOutputDirectory = ".",
  bool makePlots = true,
  bool drawStaveChipSummary = true,
  bool drawResidualRatio = true,
  double minimumDcaSignificance = 3.0,
  bool fitSummary = true,
  const char* staveChipFit = "pol0",
  bool savePngs = false)
{
  gROOT->SetBatch(kTRUE);
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(0);

  if(!writeToCsv && !makePlots)
  {
    std::cerr << "ERROR: writeToCsv and makePlots are both false; nothing "
                 "to do." << std::endl;
    return;
  }

  SummaryFit staveChipSummaryFit = SummaryFit::none;
  if(fitSummary)
  {
    const std::string fitName = staveChipFit ? staveChipFit : "";
    if(fitName == "pol0")
    {
      staveChipSummaryFit = SummaryFit::constant;
    }
    else if(fitName == "pol1")
    {
      staveChipSummaryFit = SummaryFit::line;
    }
    else
    {
      std::cerr << "ERROR: staveChipFit must be \"pol0\" (constant) or "
                   "\"pol1\" (straight line), not '" << fitName << "'"
                << std::endl;
      return;
    }
  }

  const std::vector<InputSpec> inputs = parse_input_specs(listfiles);
  if(inputs.empty())
  {
    std::cerr << "ERROR: no file lists given in '" << listfiles << "'"
              << std::endl;
    return;
  }
  if(writeToCsv && !validate_csv_labels(inputs))
  {
    return;
  }

  const int numberOfInputs = static_cast<int>(inputs.size());
  StaveHistogramMap histogramMap;
  long totalFilled = 0;
  bool anyClusterPhi = false;
  for(int input = 0; input < numberOfInputs; ++input)
  {
    std::cout << "Input " << input << " (" << inputs[input].label << "): "
              << inputs[input].listfile << std::endl;
    TChain ntp("ntp");
    if(!load_ntuple_chain(&ntp, inputs[input].listfile.c_str()) ||
       !has_required_branches(&ntp))
    {
      delete_all_histograms(histogramMap);
      return;
    }

    // clusphi is optional: it only places the chips on the cluster-phi
    // summary pages.
    const bool hasClusterPhi = ntp.GetBranch("clusphi") != nullptr;
    anyClusterPhi = anyClusterPhi || hasClusterPhi;
    if(!hasClusterPhi && drawStaveChipSummary)
    {
      std::cerr << "WARNING: no clusphi branch in ntp of '"
                << inputs[input].label << "'; its chips get no cluster phi."
                << std::endl;
    }

    const SelectionCounts counts = fill_dca_histograms(
      &ntp, input, numberOfInputs, minimumPtGeV, excludeUnknownCrossing,
      hasClusterPhi, histogramMap);
    totalFilled += counts.filled;
    std::cout << inputs[input].label << ": " << counts.rows
              << " ntp rows, " << counts.notSilicon
              << " not MVTX/INTT, " << counts.nonFiniteDca
              << " without finite -DCA, " << counts.failedPt
              << " below pT " << minimumPtGeV << " GeV/c, "
              << counts.unknownCrossing << " with unknown crossing, "
              << counts.filled << " filled" << std::endl;
    if(counts.filled == 0)
    {
      std::cerr << "WARNING: no ntp row of '" << inputs[input].label
                << "' passed the selection (was MakeMilleFiles run with the "
                   "event vertex?)." << std::endl;
    }
  }

  if(writeToCsv ||
     (makePlots &&
      (includeGaussianFit || drawStaveChipSummary || drawResidualRatio)))
  {
    fit_all_histograms(histogramMap, numberOfInputs);
  }

  if(writeToCsv &&
     !write_csv_tables(inputs, histogramMap, csvOutputDirectory))
  {
    delete_all_histograms(histogramMap);
    return;
  }

  if(makePlots && totalFilled > 0)
  {
    TString selectionText;
    if(minimumPtGeV > 0.0)
    {
      selectionText += Form("p_{T} #geq %g GeV/c", minimumPtGeV);
    }
    if(excludeUnknownCrossing)
    {
      selectionText += selectionText.IsNull() ? "" : ", ";
      selectionText += "known crossing";
    }
    if(!selectionText.IsNull())
    {
      selectionText = " (" + selectionText + ")";
    }

    const std::string pdfPath = labelled_pdf_name(outputPdf, inputs);
    PdfOutput pdfOutput{pdfPath, false, savePngs};

    int summaryPages = 0;
    if(drawStaveChipSummary)
    {
      // Order: mu/sigma vs stave-chip ID for every layer, then chip phi vs
      // stave-chip ID (one page per detector), then mu/sigma vs chip phi for
      // every layer.
      summaryPages += draw_stavechip_summary_pages(
        kMvtxConfig, histogramMap, inputs, selectionText, false,
        staveChipSummaryFit, pdfOutput);
      summaryPages += draw_stavechip_summary_pages(
        kInttConfig, histogramMap, inputs, selectionText, false,
        staveChipSummaryFit, pdfOutput);
      if(anyClusterPhi)
      {
        summaryPages += draw_chip_phi_page(
          kMvtxConfig, histogramMap, selectionText, pdfOutput);
        summaryPages += draw_chip_phi_page(
          kInttConfig, histogramMap, selectionText, pdfOutput);
        summaryPages += draw_stavechip_summary_pages(
          kMvtxConfig, histogramMap, inputs, selectionText, true,
          staveChipSummaryFit, pdfOutput);
        summaryPages += draw_stavechip_summary_pages(
          kInttConfig, histogramMap, inputs, selectionText, true,
          staveChipSummaryFit, pdfOutput);
      }
    }

    int ratioPages = 0;
    if(drawResidualRatio)
    {
      for(int layer = kMvtxConfig.firstLayer;
          layer < kMvtxConfig.firstLayer + kMvtxConfig.numberOfLayers; ++layer)
      {
        ratioPages += draw_residual_ratio_page(
          layer, histogramMap, inputs, selectionText, minimumDcaSignificance,
          pdfOutput);
      }
      ratioPages += draw_residual_scatter_page(
        histogramMap, inputs, selectionText, pdfOutput);
    }

    const int mvtxStaves = draw_detector_pages(
      kMvtxConfig, histogramMap, inputs, selectionText, pdfOutput,
      includeGaussianFit, drawHistogramsAsMarkers, normalizeToUnitArea);
    const int inttStaves = draw_detector_pages(
      kInttConfig, histogramMap, inputs, selectionText, pdfOutput,
      includeGaussianFit, drawHistogramsAsMarkers, normalizeToUnitArea);

    if(pdfOutput.isOpen)
    {
      TCanvas pdfBookend("c_silicon_stavechip_dca_bookend", "", 1, 1);
      pdfBookend.Print((pdfOutput.path + "]").c_str());
    }
    std::cout << "Wrote "
              << summaryPages + ratioPages +
                   kNumberOfCoordinates * (mvtxStaves + inttStaves)
              << " pages (" << summaryPages << " stave-chip summary pages, "
              << ratioPages << " z residual / DCA_z pages) for " << mvtxStaves
              << " populated MVTX and "
              << inttStaves << " populated INTT staves from "
              << numberOfInputs << " input list(s) to " << pdfPath
              << std::endl;
  }
  else if(makePlots)
  {
    std::cerr << "WARNING: no ntp row passed the selection; no plots were "
                 "written." << std::endl;
  }

  delete_all_histograms(histogramMap);
}
