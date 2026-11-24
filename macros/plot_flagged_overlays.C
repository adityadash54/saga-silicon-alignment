#include <TCanvas.h>
#include <TChain.h>
#include <TH1D.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TLine.h>
#include <TMath.h>
#include <TROOT.h>
#include <TString.h>
#include <TStyle.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// Overlaid residual histograms of the flagged entries of one comparison, to
// see the shift between the reference and the comparison input directly.
// Reads the flags table that stdiag.compare_runs wrote
// (flags_<reference>_vs_<comparison>.csv) and the same ROOT files as
// plot_fits_overlaid_stavechip.C, and fills histograms only for the flagged
// entries, with the same histogram ranges and bin width as that macro:
//   - MVTX chips / INTT sensors from ntp (residualX = local r-phi,
//     residualY = local z, in mm);
//   - whole-stave rows (chip = -1) as the sum of the stave's chips;
//   - VERTEX rows from track_ntp (residual_vertexX/Y = -DCA_xy/-DCA_z, mm):
//     chip = -1 all tracks, chip = 0..13 the track-phi bins over [-pi, pi).
// Each panel overlays the two inputs normalised to unit area, with the
// fitted means from the flags table as dashed lines and Delta mu, z and the
// flag reason in the legend.
//
// Usage (from the repository root):
//   root -l -b -q 'macros/plot_flagged_overlays.C+("runs/Iter0_vs_Iter1/flags_Iter0_vs_Iter1.csv","inputs/list_Iter0.txt:Iter0,inputs/list_Iter1.txt:Iter1")'
// listfiles are the list:label pairs given to the fit macro; only the two
// labels of the flags table are read, so a list with more labels is fine.
// outputPdf defaults to flagged_overlays_<reference>_vs_<comparison>.pdf
// next to the flags table; one PNG per page is written next to the PDF.
// Runs locally only; nothing is sent to any external service.

namespace
{
  constexpr int kResidualBins = 200;
  constexpr int kVertexPhiBins = 14;
  constexpr int kPadsPerPage = 6;
  constexpr int kPadColumns = 3;
  constexpr int kPadRows = 2;
  // Histogram ranges as in plot_fits_overlaid_stavechip.C; the bin width is
  // the display range divided by kResidualBins.
  constexpr double kMvtxDisplayMm = 0.05;
  constexpr double kMvtxHistogramMm = 0.5;
  constexpr double kInttRphiDisplayMm = 0.6;
  constexpr double kInttRphiHistogramMm = 3.0;
  constexpr double kInttZDisplayMm = 20.0;
  constexpr double kInttZHistogramMm = 40.0;
  constexpr double kVertexDisplayMm = 2.0;

  struct FlaggedEntry
  {
    std::string detector;
    int layer = 0;
    int stave = 0;
    int chip = 0;
    std::string coord;
    double mu1 = std::numeric_limits<double>::quiet_NaN();
    double mu2 = std::numeric_limits<double>::quiet_NaN();
    double sigma1 = std::numeric_limits<double>::quiet_NaN();
    double sigma2 = std::numeric_limits<double>::quiet_NaN();
    double deltaMu = std::numeric_limits<double>::quiet_NaN();
    double muZ = std::numeric_limits<double>::quiet_NaN();
    double sigmaRel = std::numeric_limits<double>::quiet_NaN();
    double sigmaRelSignificance = std::numeric_limits<double>::quiet_NaN();
    bool flagMean = false;
    bool flagWidth = false;
    TH1D* histograms[2] = {nullptr, nullptr};
  };

  // Splits one CSV line; handles double-quoted fields (pandas quotes fields
  // that contain commas).
  std::vector<std::string> split_csv_line(const std::string& line)
  {
    std::vector<std::string> fields(1);
    bool quoted = false;
    for(std::string::size_type index = 0; index < line.size(); ++index)
    {
      const char character = line[index];
      if(character == '"')
      {
        if(quoted && index + 1 < line.size() && line[index + 1] == '"')
        {
          fields.back() += '"';
          ++index;
        }
        else
        {
          quoted = !quoted;
        }
      }
      else if(character == ',' && !quoted)
      {
        fields.emplace_back();
      }
      else if(character != '\r')
      {
        fields.back() += character;
      }
    }
    return fields;
  }

  double to_double(const std::string& text)
  {
    try
    {
      return text.empty() ? std::numeric_limits<double>::quiet_NaN()
                          : std::stod(text);
    }
    catch(const std::exception&)
    {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }

  bool read_flags(const char* path, std::vector<FlaggedEntry>& entries,
                  std::string& referenceLabel, std::string& comparisonLabel)
  {
    std::ifstream input(path);
    if(!input)
    {
      std::cerr << "ERROR: cannot open " << path << std::endl;
      return false;
    }
    std::string line;
    std::getline(input, line);
    const std::vector<std::string> header = split_csv_line(line);
    std::map<std::string, std::size_t> column;
    for(std::size_t index = 0; index < header.size(); ++index)
    {
      column[header[index]] = index;
    }
    for(const char* name : {"reference_label", "comparison_label", "detector",
                            "layer", "stave", "chip", "coord", "mu_1_mm",
                            "mu_2_mm", "sigma_1_mm", "sigma_2_mm",
                            "delta_mu_mm", "mu_z", "sigma_rel",
                            "sigma_rel_significance", "flag_mean", "flag_width",
                            "comparison_status"})
    {
      if(!column.count(name))
      {
        std::cerr << "ERROR: " << path << " has no column '" << name
                  << "'; give a flags table from stdiag.compare_runs"
                  << std::endl;
        return false;
      }
    }

    while(std::getline(input, line))
    {
      const std::vector<std::string> fields = split_csv_line(line);
      if(fields.size() < header.size())
      {
        continue;
      }
      referenceLabel = fields[column["reference_label"]];
      comparisonLabel = fields[column["comparison_label"]];
      if(fields[column["comparison_status"]] != "flagged")
      {
        continue;
      }
      FlaggedEntry entry;
      entry.detector = fields[column["detector"]];
      entry.layer = static_cast<int>(to_double(fields[column["layer"]]));
      entry.stave = static_cast<int>(to_double(fields[column["stave"]]));
      entry.chip = static_cast<int>(to_double(fields[column["chip"]]));
      entry.coord = fields[column["coord"]];
      entry.mu1 = to_double(fields[column["mu_1_mm"]]);
      entry.mu2 = to_double(fields[column["mu_2_mm"]]);
      entry.sigma1 = to_double(fields[column["sigma_1_mm"]]);
      entry.sigma2 = to_double(fields[column["sigma_2_mm"]]);
      entry.deltaMu = to_double(fields[column["delta_mu_mm"]]);
      entry.muZ = to_double(fields[column["mu_z"]]);
      entry.sigmaRel = to_double(fields[column["sigma_rel"]]);
      entry.sigmaRelSignificance =
        to_double(fields[column["sigma_rel_significance"]]);
      entry.flagMean = fields[column["flag_mean"]] == "True";
      entry.flagWidth = fields[column["flag_width"]] == "True";
      entries.push_back(entry);
    }
    return true;
  }

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

  // "list:label,list:label" -> label -> list file.
  std::map<std::string, std::string> parse_listfiles(const char* listfiles)
  {
    std::map<std::string, std::string> lists;
    std::stringstream stream(listfiles);
    std::string token;
    while(std::getline(stream, token, ','))
    {
      token = trim(token);
      const std::string::size_type colon = token.find(':');
      if(!token.empty() && colon != std::string::npos)
      {
        lists[trim(token.substr(colon + 1))] = trim(token.substr(0, colon));
      }
    }
    return lists;
  }

  // Adds every ROOT file named in a list file (one per line) to the chain.
  int add_files(TChain& chain, const std::string& listfile)
  {
    std::ifstream input(listfile);
    if(!input)
    {
      std::cerr << "ERROR: cannot open list file " << listfile << std::endl;
      return 0;
    }
    int added = 0;
    std::string line;
    while(std::getline(input, line))
    {
      line = trim(line);
      if(line.empty() || line[0] == '#')
      {
        continue;
      }
      added += chain.Add(line.c_str());
    }
    return added;
  }

  void histogram_range(const FlaggedEntry& entry, double& halfRange,
                       int& bins)
  {
    double display = kMvtxDisplayMm;
    halfRange = kMvtxHistogramMm;
    if(entry.detector == "INTT" && entry.coord == "rphi")
    {
      display = kInttRphiDisplayMm;
      halfRange = kInttRphiHistogramMm;
    }
    else if(entry.detector == "INTT")
    {
      display = kInttZDisplayMm;
      halfRange = kInttZHistogramMm;
    }
    else if(entry.detector == "VERTEX")
    {
      display = kVertexDisplayMm;
      halfRange = kVertexDisplayMm;
    }
    bins = static_cast<int>(std::lround(kResidualBins * halfRange / display));
  }

  std::string entry_title(const FlaggedEntry& entry)
  {
    if(entry.detector == "VERTEX")
    {
      const std::string coord = entry.coord == "dca_xy" ? "#minusDCA_{xy}"
                                                        : "#minusDCA_{z}";
      return entry.chip < 0 ? Form("VERTEX %s, all tracks", coord.c_str())
                            : Form("VERTEX %s, #phi bin %d", coord.c_str(),
                                   entry.chip);
    }
    const std::string unit = entry.detector == "MVTX" ? "stave" : "ladder";
    const std::string part = entry.detector == "MVTX" ? "chip" : "sensor";
    const std::string coord = entry.coord == "rphi" ? "r#phi" : "z";
    if(entry.chip < 0)
    {
      return Form("%s L%d %s %d, whole %s, %s", entry.detector.c_str(),
                  entry.layer, unit.c_str(), entry.stave, unit.c_str(),
                  coord.c_str());
    }
    return Form("%s L%d %s %d %s %d, %s", entry.detector.c_str(), entry.layer,
                unit.c_str(), entry.stave, part.c_str(), entry.chip,
                coord.c_str());
  }

  bool get_integer(Float_t value, int& identifier)
  {
    if(!std::isfinite(value) || std::fabs(value - std::round(value)) > 1e-5)
    {
      return false;
    }
    identifier = static_cast<int>(std::round(value));
    return true;
  }

  using EntryKey = std::tuple<int, int, int, int>;  // layer, stave, chip, coord

  void fill_silicon(const std::string& listfile, int input,
                    std::vector<FlaggedEntry>& entries)
  {
    std::map<EntryKey, std::vector<std::size_t>> lookup;
    for(std::size_t index = 0; index < entries.size(); ++index)
    {
      const FlaggedEntry& entry = entries[index];
      if(entry.detector != "VERTEX")
      {
        lookup[EntryKey(entry.layer, entry.stave, entry.chip,
                        entry.coord == "rphi" ? 0 : 1)].push_back(index);
      }
    }
    if(lookup.empty())
    {
      return;
    }
    TChain chain("ntp");
    if(add_files(chain, listfile) == 0)
    {
      return;
    }
    TTreeReader reader(&chain);
    TTreeReaderValue<Float_t> layerValue(reader, "layer");
    TTreeReaderValue<Float_t> staveValue(reader, "stave");
    TTreeReaderValue<Float_t> chipValue(reader, "chip");
    TTreeReaderValue<Float_t> residualX(reader, "residualX");
    TTreeReaderValue<Float_t> residualY(reader, "residualY");
    while(reader.Next())
    {
      int layer = 0;
      int stave = 0;
      int chip = 0;
      if(!get_integer(*layerValue, layer) || !get_integer(*staveValue, stave)
         || !get_integer(*chipValue, chip))
      {
        continue;
      }
      const Float_t residuals[2] = {*residualX, *residualY};
      for(int coord = 0; coord < 2; ++coord)
      {
        if(!std::isfinite(residuals[coord]))
        {
          continue;
        }
        for(const int key : {chip, -1})
        {
          const auto found = lookup.find(EntryKey(layer, stave, key, coord));
          if(found == lookup.end())
          {
            continue;
          }
          for(const std::size_t index : found->second)
          {
            entries[index].histograms[input]->Fill(residuals[coord]);
          }
        }
      }
    }
  }

  void fill_vertex(const std::string& listfile, int input,
                   std::vector<FlaggedEntry>& entries)
  {
    std::vector<std::size_t> vertexEntries;
    for(std::size_t index = 0; index < entries.size(); ++index)
    {
      if(entries[index].detector == "VERTEX")
      {
        vertexEntries.push_back(index);
      }
    }
    if(vertexEntries.empty())
    {
      return;
    }
    TChain chain("track_ntp");
    if(add_files(chain, listfile) == 0 || chain.GetEntries() == 0)
    {
      std::cerr << "WARNING: no track_ntp in " << listfile
                << "; VERTEX panels stay empty" << std::endl;
      return;
    }
    TTreeReader reader(&chain);
    TTreeReaderValue<Float_t> transverse(reader, "residual_vertexX");
    TTreeReaderValue<Float_t> longitudinal(reader, "residual_vertexY");
    TTreeReaderValue<Float_t> phi(reader, "track_phi");
    while(reader.Next())
    {
      int phiBin = -2;
      if(std::isfinite(*phi))
      {
        phiBin = static_cast<int>(std::floor(
          (*phi + TMath::Pi()) / (2.0 * TMath::Pi()) * kVertexPhiBins));
        if(phiBin < 0 || phiBin >= kVertexPhiBins)
        {
          phiBin = -2;
        }
      }
      for(const std::size_t index : vertexEntries)
      {
        FlaggedEntry& entry = entries[index];
        const Float_t value =
          entry.coord == "dca_xy" ? *transverse : *longitudinal;
        if(std::isfinite(value) && (entry.chip == -1 || entry.chip == phiBin))
        {
          entry.histograms[input]->Fill(value);
        }
      }
    }
  }

  // Shifts and widths in um, except INTT z in mm.
  std::string format_length(const FlaggedEntry& entry, double valueMm)
  {
    if(entry.detector == "INTT" && entry.coord == "z")
    {
      return Form("%.2f mm", valueMm);
    }
    return Form("%.1f #mum", 1000.0 * valueMm);
  }

  void draw_panel(FlaggedEntry& entry, const std::string labels[2])
  {
    const int colors[2] = {kBlack, kRed + 1};
    double halfRange = 0.0;
    int bins = 0;
    histogram_range(entry, halfRange, bins);
    const double width = std::max(std::isfinite(entry.sigma1) ? entry.sigma1
                                                             : 0.0,
                                  std::isfinite(entry.sigma2) ? entry.sigma2
                                                             : 0.0);
    double centre = 0.5 * (entry.mu1 + entry.mu2);
    double half = std::max(4.0 * width, 0.75 * std::fabs(entry.mu2 - entry.mu1)
                                          + 3.0 * width);
    if(!std::isfinite(centre) || !(half > 0.0))
    {
      centre = 0.0;
      half = halfRange;
    }
    half = std::min(half, halfRange);

    double maximum = 0.0;
    for(int input = 0; input < 2; ++input)
    {
      TH1D* histogram = entry.histograms[input];
      if(histogram->Integral() > 0.0)
      {
        histogram->Scale(1.0 / histogram->Integral());
      }
      histogram->GetXaxis()->SetRangeUser(centre - half, centre + half);
      maximum = std::max(maximum, histogram->GetMaximum());
    }

    for(int input = 0; input < 2; ++input)
    {
      TH1D* histogram = entry.histograms[input];
      histogram->SetLineColor(colors[input]);
      histogram->SetLineWidth(2);
      // Headroom above the histograms for the legend and the text.
      histogram->SetMaximum(2.0 * maximum);
      histogram->SetMinimum(0.0);
      histogram->SetTitle(entry_title(entry).c_str());
      histogram->GetXaxis()->SetTitle("residual [mm]");
      histogram->GetYaxis()->SetTitle("fraction of entries");
      histogram->Draw(input == 0 ? "hist" : "hist same");
    }

    const double means[2] = {entry.mu1, entry.mu2};
    for(int input = 0; input < 2; ++input)
    {
      if(std::isfinite(means[input]))
      {
        TLine* line = new TLine(means[input], 0.0, means[input],
                                1.05 * maximum);
        line->SetLineColor(colors[input]);
        line->SetLineStyle(2);
        line->Draw();
      }
    }

    TLegend* legend = new TLegend(0.15, 0.75, 0.88, 0.89);
    legend->SetBorderSize(0);
    legend->SetFillStyle(0);
    legend->SetTextSize(0.040);
    const double sigmas[2] = {entry.sigma1, entry.sigma2};
    for(int input = 0; input < 2; ++input)
    {
      legend->AddEntry(entry.histograms[input],
                       Form("%s: #mu = %s, #sigma = %s",
                            labels[input].c_str(),
                            format_length(entry, means[input]).c_str(),
                            format_length(entry, sigmas[input]).c_str()),
                       "l");
    }
    legend->Draw();

    TLatex text;
    text.SetNDC();
    text.SetTextSize(0.038);
    const std::string test = entry.flagMean && entry.flagWidth
                               ? "mean and width"
                               : entry.flagWidth ? "width" : "mean";
    text.DrawLatex(0.16, 0.70, Form("#Delta#mu = %s (z = %.1f),  "
                                    "#sigma_{2}/#sigma_{1} #minus 1 = %+.0f%% "
                                    "(%.1f#sigma)",
                                    format_length(entry, entry.deltaMu).c_str(),
                                    entry.muZ, 100.0 * entry.sigmaRel,
                                    entry.sigmaRelSignificance));
    text.DrawLatex(0.16, 0.65, Form("flagged by the %s test", test.c_str()));
  }
}  // namespace

void plot_flagged_overlays(
  const char* flagsCsv,
  const char* listfiles,
  const char* outputPdf = "")
{
  gROOT->SetBatch(kTRUE);
  gStyle->SetOptStat(0);

  std::vector<FlaggedEntry> entries;
  std::string labels[2];
  if(!read_flags(flagsCsv, entries, labels[0], labels[1]))
  {
    return;
  }
  if(entries.empty())
  {
    std::cout << "No flagged entries in " << flagsCsv << "; nothing to plot."
              << std::endl;
    return;
  }

  const std::map<std::string, std::string> lists = parse_listfiles(listfiles);
  for(const std::string& label : labels)
  {
    if(!lists.count(label))
    {
      std::cerr << "ERROR: no list file for label '" << label
                << "' in \"" << listfiles << "\"" << std::endl;
      return;
    }
  }

  for(std::size_t index = 0; index < entries.size(); ++index)
  {
    double halfRange = 0.0;
    int bins = 0;
    histogram_range(entries[index], halfRange, bins);
    for(int input = 0; input < 2; ++input)
    {
      entries[index].histograms[input] = new TH1D(
        Form("h_flagged_%zu_in%d", index, input), "", bins, -halfRange,
        halfRange);
      entries[index].histograms[input]->SetDirectory(nullptr);
    }
  }
  for(int input = 0; input < 2; ++input)
  {
    fill_silicon(lists.at(labels[input]), input, entries);
    fill_vertex(lists.at(labels[input]), input, entries);
  }

  std::string pdf = outputPdf;
  if(pdf.empty())
  {
    std::string folder = flagsCsv;
    const std::string::size_type slash = folder.find_last_of('/');
    folder = slash == std::string::npos ? "." : folder.substr(0, slash);
    pdf = folder + "/flagged_overlays_" + labels[0] + "_vs_" + labels[1] +
          ".pdf";
  }
  const std::string pngBase = pdf.substr(0, pdf.size() - 4);

  TCanvas canvas("c_flagged", "", 1500, 900);
  canvas.Print((pdf + "[").c_str());
  const int pages = (static_cast<int>(entries.size()) + kPadsPerPage - 1) /
                    kPadsPerPage;
  for(int page = 0; page < pages; ++page)
  {
    canvas.Clear();
    canvas.Divide(kPadColumns, kPadRows);
    for(int pad = 0; pad < kPadsPerPage; ++pad)
    {
      const std::size_t index =
        static_cast<std::size_t>(page * kPadsPerPage + pad);
      if(index >= entries.size())
      {
        break;
      }
      canvas.cd(pad + 1);
      gPad->SetLeftMargin(0.13);
      gPad->SetBottomMargin(0.13);
      draw_panel(entries[index], labels);
    }
    canvas.Print(pdf.c_str());
    canvas.Print(Form("%s_page%d.png", pngBase.c_str(), page + 1));
  }
  canvas.Print((pdf + "]").c_str());
  std::cout << "Wrote " << entries.size() << " flagged entries ("
            << labels[1] << " vs " << labels[0] << ") on " << pages
            << " page(s) to " << pdf << std::endl;
}
