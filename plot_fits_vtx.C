#include <TArrow.h>
#include <TFile.h>
#include <TNtuple.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TCanvas.h>
#include <TCut.h>
#include <TF1.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TChain.h>
#include <TMath.h>
#include <TStyle.h>
#include <TPaveText.h>
#include <TString.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <fstream>
#include <string>
#include <vector>

bool fit_gaussian_if_possible(TH1D *hist, TF1 *fit_function)
{
  if(!hist || !fit_function || hist->GetEntries() < 10)
    {
      std::cerr << "WARNING: insufficient entries for Gaussian fit of "
                << (hist ? hist->GetName() : "<null histogram>") << std::endl;
      return false;
    }

  const int fit_status = hist->Fit(fit_function, "RQ");
  if(fit_status != 0)
    {
      std::cerr << "WARNING: Gaussian fit failed for " << hist->GetName()
                << " with status " << fit_status << std::endl;
      return false;
    }

  return true;
}

// Left edge (NDC) of the text boxes: just inside the left margin of the
// current pad, 0.12 for the default margin.
double box_left_edge()
{
  return std::max(0.12, gPad->GetLeftMargin() + 0.02);
}

void draw_fit_range_box(const TF1 *fit_function)
{
  if(!fit_function)
    {
      return;
    }

  const double left = box_left_edge();
  TPaveText *fit_range_box =
      new TPaveText(left, 0.80, left + 0.40, 0.88, "NDC");
  fit_range_box->SetFillColor(0);
  fit_range_box->SetFillStyle(0);
  fit_range_box->SetBorderSize(0);
  fit_range_box->SetTextAlign(12);
  fit_range_box->SetTextSize(0.04);
  fit_range_box->AddText(
      Form("Fit range: [%g, %g] mm",
           fit_function->GetXmin(),
           fit_function->GetXmax()));
  fit_range_box->Draw();
}

// Box with the track selection (e.g. a pT cut) below the fit-range box;
// nothing is drawn for an empty selection.
void draw_selection_box(const TString &selection_text)
{
  if(selection_text.IsNull())
    {
      return;
    }

  const double left = box_left_edge();
  TPaveText *selection_box =
      new TPaveText(left, 0.72, left + 0.40, 0.80, "NDC");
  selection_box->SetFillColor(0);
  selection_box->SetFillStyle(0);
  selection_box->SetBorderSize(0);
  selection_box->SetTextAlign(12);
  selection_box->SetTextSize(0.04);
  selection_box->AddText(selection_text.Data());
  selection_box->Draw();
}

// All canvases are written as pages of this PDF, in the order they are
// drawn: add_to_combined_pdf() collects them and write_combined_pdf()
// writes the file at the end of the macro.
const char *const kCombinedPdf = "plot_fits_vtx_all.pdf";

// Canvases for the combined PDF, in drawing order.
std::vector<TCanvas*> &combined_pdf_pages()
{
  static std::vector<TCanvas*> pages;
  return pages;
}

// Adds canvas as a page of the combined PDF. With save_individual_files,
// the canvas is also saved as <page_title>.pdf and <page_title>.png; this
// is safe here because the combined PDF is only opened at the end.
void add_to_combined_pdf(TCanvas *canvas, const TString &page_title,
                         bool save_individual_files)
{
  canvas->Update();
  combined_pdf_pages().push_back(canvas);
  if(save_individual_files)
    {
      canvas->SaveAs((page_title + ".pdf").Data());
      canvas->SaveAs((page_title + ".png").Data());
    }
}

// Writes the collected canvases into kCombinedPdf in one pass, as
// write_multipage_pdf in plot_fits_overlaid.C does: ROOT keeps a single
// open PDF (gVirtualPS), so no other PDF may be written while it is open.
// ROOT also takes the page layout from the canvas that opened the file, so
// canvases of other sizes would be shifted off the page. Each canvas is
// therefore copied (DrawClonePad) into a pad of the same aspect ratio on one
// fixed-size page canvas, at the top and centred, and that page is printed.
// Print gets no option (and so no "Title:" bookmark): ROOT 6.32 misread
// "Title:vertex_..." as the TeX format ("tex") and crashed.
void write_combined_pdf()
{
  const std::vector<TCanvas*> &pages = combined_pdf_pages();
  if(pages.empty())
    {
      std::cout << "No pages for " << kCombinedPdf << "; not written."
                << std::endl;
      return;
    }

  const double page_width = 1600.0;
  const double page_height = 2263.0;  // A4 portrait aspect ratio
  TCanvas *page = new TCanvas("c_combined_pdf_page", "", 0, 0,
                              (int)page_width, (int)page_height);
  page->Print(Form("%s[", kCombinedPdf));
  for(TCanvas *canvas : pages)
    {
      page->Clear();
      page->cd();
      const double canvas_width = canvas->GetWw();
      const double canvas_height = canvas->GetWh();
      const double scale = std::min(page_width / canvas_width,
                                    page_height / canvas_height);
      const double width = scale * canvas_width / page_width;
      const double height = scale * canvas_height / page_height;
      TPad *frame = new TPad(Form("p_pdf_%s", canvas->GetName()), "",
                             0.5 * (1.0 - width), 1.0 - height,
                             0.5 * (1.0 + width), 1.0);
      frame->Draw();
      frame->cd();
      canvas->DrawClonePad();
      page->Update();
      page->Print(kCombinedPdf);
    }
  page->Print(Form("%s]", kCombinedPdf));
  std::cout << "Wrote " << pages.size() << " pages to " << kCombinedPdf
            << std::endl;
}

// Marks the slices of hist whose value lies outside [y_min, y_max], the
// fixed y range of the current pad, with an arrow at the pad edge pointing
// off-scale, and prints them. Slices without a FitSlicesY result (content
// and error 0) are skipped. Returns the number of off-scale slices.
int mark_off_scale_slices(const TH1D *hist, double y_min, double y_max)
{
  const double arrow_length = 0.08 * (y_max - y_min);
  int n_off_scale = 0;
  for(int bin = 1; bin <= hist->GetNbinsX(); ++bin)
    {
      const double content = hist->GetBinContent(bin);
      const double error = hist->GetBinError(bin);
      if(!std::isfinite(content) || (content == 0.0 && error == 0.0))
        {
          continue;
        }
      if(content >= y_min && content <= y_max)
        {
          continue;
        }

      const double x = hist->GetXaxis()->GetBinCenter(bin);
      const double edge = content > y_max ? y_max : y_min;
      const double tail = content > y_max ? y_max - arrow_length
                                          : y_min + arrow_length;
      TArrow *arrow = new TArrow(x, tail, x, edge, 0.01, "|>");
      arrow->SetLineColor(kMagenta + 1);
      arrow->SetFillColor(kMagenta + 1);
      arrow->SetLineWidth(2);
      arrow->Draw();

      std::cerr << "WARNING: " << hist->GetName() << " slice at "
                << hist->GetXaxis()->GetTitle() << " = " << x << " is "
                << 1000.0 * content << " um, outside the plotted range ["
                << 1000.0 * y_min << ", " << 1000.0 * y_max << "] um."
                << std::endl;
      ++n_off_scale;
    }
  return n_off_scale;
}

// Fits A*sin(phi - phi0) + c to the <-DCA_xy> vs track phi points of
// mean_hist through a*sin(phi) + b*cos(phi) + c (linear in the parameters,
// so the minimum is unique) with A = sqrt(a^2 + b^2), phi0 = atan2(-b, a),
// and draws the curve and the result in the current pad. Slices without a
// FitSlicesY result are empty and do not enter the fit. As in
// plot_fits_overlaid_stavechip.C, residual_vertexX is -DCA_xy, so
// A*(cos phi0, sin phi0) is the event vertex minus the point the tracks
// point back to. A is biased upwards when it is comparable to its error.
bool fit_and_draw_phi_modulation(TH1D *mean_hist)
{
  const double phi_min = mean_hist->GetXaxis()->GetXmin();
  const double phi_max = mean_hist->GetXaxis()->GetXmax();
  TF1 linear_fit(Form("f_linear_%s", mean_hist->GetName()),
                 "[0]*sin(x)+[1]*cos(x)+[2]", phi_min, phi_max);
  linear_fit.SetParameters(0.0, 0.0, 0.0);
  const TFitResultPtr fit_result = mean_hist->Fit(&linear_fit, "QNS");
  const int fit_status = fit_result;
  if(fit_status != 0 || !fit_result.Get())
    {
      std::cerr << "WARNING: A*sin(phi - phi0) + c fit failed for "
                << mean_hist->GetName() << " with status " << fit_status
                << std::endl;
      return false;
    }

  const double a = linear_fit.GetParameter(0);
  const double b = linear_fit.GetParameter(1);
  const double variance_a = fit_result->CovMatrix(0, 0);
  const double variance_b = fit_result->CovMatrix(1, 1);
  const double covariance = fit_result->CovMatrix(0, 1);
  const double amplitude = std::hypot(a, b);
  const double phi0 = std::atan2(-b, a);
  const double constant = linear_fit.GetParameter(2);
  const double constant_error = linear_fit.GetParError(2);
  double amplitude_error = std::numeric_limits<double>::quiet_NaN();
  double phi0_error = std::numeric_limits<double>::quiet_NaN();
  if(amplitude > 0.0)
    {
      amplitude_error = std::sqrt(std::max(0.0,
          (a * a * variance_a + 2.0 * a * b * covariance +
           b * b * variance_b) / (amplitude * amplitude)));
      phi0_error = std::sqrt(std::max(0.0,
          (b * b * variance_a - 2.0 * a * b * covariance +
           a * a * variance_b) / std::pow(amplitude, 4)));
    }

  TF1 *curve = new TF1(Form("f_phi_modulation_%s", mean_hist->GetName()),
                       "[0]*sin(x-[1])+[2]", phi_min, phi_max);
  curve->SetParameters(amplitude, phi0, constant);
  curve->SetLineColor(kRed);
  curve->SetLineWidth(2);
  curve->Draw("same");

  TPaveText *fit_box =
      new TPaveText(box_left_edge(), 0.14, 0.88, 0.30, "NDC");
  fit_box->SetFillColor(0);
  fit_box->SetFillStyle(0);
  fit_box->SetBorderSize(0);
  fit_box->SetTextAlign(12);
  fit_box->SetTextSize(0.04);
  fit_box->SetTextColor(kRed);
  fit_box->AddText("A sin(#phi - #phi_{0}) + c:");
  fit_box->AddText(Form("A = %.1f #pm %.1f #mum, #phi_{0} = %.2f #pm %.2f rad",
                        1000.0 * amplitude, 1000.0 * amplitude_error,
                        phi0, phi0_error));
  fit_box->AddText(Form("c = %.1f #pm %.1f #mum, #chi^{2}/ndf = %.1f/%d",
                        1000.0 * constant, 1000.0 * constant_error,
                        linear_fit.GetChisquare(), linear_fit.GetNDF()));
  fit_box->Draw();

  std::cout << mean_hist->GetName() << ": A*sin(phi - phi0) + c fit: A = "
            << 1000.0 * amplitude << " +- " << 1000.0 * amplitude_error
            << " um, phi0 = " << phi0 << " +- " << phi0_error
            << " rad, c = " << 1000.0 * constant << " +- "
            << 1000.0 * constant_error << " um, chi2/ndf = "
            << linear_fit.GetChisquare() << "/" << linear_fit.GetNDF()
            << std::endl;
  return true;
}

// A track_ntp variable for draw_vertex_residuals_vs_track_variables.
struct TrackVariable
{
  const char *branch;
  const char *name;        // histogram name suffix
  const char *symbol;      // in the histogram titles
  const char *axis_title;
  int nbins;
  double min;
  double max;
};

// Two pages of track-to-vertex residuals versus two track variables:
//   <output_base>: 2D histograms, from top left to bottom right DCA_xy vs
//     variable 0, DCA_z vs variable 0, DCA_xy vs variable 1, DCA_z vs
//     variable 1;
//   <output_base>_slices: Gaussian mean (left) and sigma (right) of the
//     residuals per slice of 4 grouped bins, one row per 2D histogram in the
//     same order. Slices with fewer than 10 entries are not fitted.
// selection_text (if not empty) describes cuts beyond the finite-residual
// cut; it is added to the titles and shown in a box on every pad. With
// fit_phi_modulation, A*sin(phi - phi0) + c is fitted to the DCA_xy mean vs
// variable 1, which must then be track phi. Both pages go to the combined
// PDF (see add_to_combined_pdf). The slice plots have fixed y ranges in mm,
// index 0 = DCA_rphi, 1 = DCA_z: mean within +-mean_axis_limit, sigma
// within 0 to sigma_axis_max. Off-scale slices are marked with arrows. The
// 2D plots show the DCA axis within +-dca_2d_axis_limit [mm]; this is a
// display range only, the slice fits use the full +-2 mm histograms.
void draw_vertex_residuals_vs_track_variables(
    TChain *track_ntp,
    const TCut &cut,
    const TrackVariable variables[2],
    const TString &output_base,
    const TString &selection_text,
    bool fit_phi_modulation,
    bool save_individual_files,
    const double mean_axis_limit[2],
    const double sigma_axis_max[2],
    const double dca_2d_axis_limit[2])
{
  for(int v = 0; v < 2; ++v)
    {
      if(!track_ntp->GetBranch(variables[v].branch))
        {
          std::cerr << "WARNING: " << variables[v].branch
                    << " is missing from tree track_ntp; skipping "
                    << output_base << "." << std::endl;
          return;
        }
    }

  const char *residual_branches[2] = {"residual_vertexX", "residual_vertexY"};
  const char *residual_letters[2] = {"x", "y"};
  // residual_vertexX/Y store -DCA_xy/-DCA_z (MakeMilleFiles vtx_residual).
  const char *residual_kinds[2] = {"#minusDCA_{r#phi}", "#minusDCA_{z}"};
  // Left margin and y-title offset wide enough for the y-axis titles; the
  // right margin of the 2D pads leaves room for the colz palette.
  const double left_margin = 0.17;
  const double right_margin_2d = 0.15;
  const double y_title_offset = 1.9;
  const TString title_suffix =
      selection_text.IsNull() ? TString("") : " (" + selection_text + ")";

  TCanvas *canvas = new TCanvas(
      ("c_" + output_base).Data(), "", 50, 300, 1600, 1600);
  canvas->Divide(2, 2);

  TH2D *hists[4];
  TString hist_titles[4];
  for(int i = 0; i < 4; ++i)
    {
      const TrackVariable &variable = variables[i / 2];
      const int residual = i % 2;
      const TString hist_name = Form(
          "h_vtx_%s_%s", residual_letters[residual], variable.name);
      hist_titles[i] = Form("%s vs %s",
                            residual_kinds[residual], variable.symbol);
      hist_titles[i] += title_suffix;

      canvas->cd(i + 1);
      gPad->SetLeftMargin(left_margin);
      gPad->SetRightMargin(right_margin_2d);
      hists[i] = new TH2D(
          hist_name.Data(), hist_titles[i].Data(),
          variable.nbins, variable.min, variable.max, 400, -2.0, 2.0);
      track_ntp->Draw(
          Form("%s:%s>>%s", residual_branches[residual], variable.branch,
               hist_name.Data()),
          cut, "colz");
      hists[i]->GetXaxis()->SetTitle(variable.axis_title);
      hists[i]->GetYaxis()->SetTitle(
          Form("%s [mm]", residual_kinds[residual]));
      hists[i]->GetYaxis()->SetTitleOffset(y_title_offset);
      hists[i]->GetYaxis()->SetRangeUser(-dca_2d_axis_limit[residual],
                                         dca_2d_axis_limit[residual]);
      draw_selection_box(selection_text);
    }

  add_to_combined_pdf(canvas, output_base, save_individual_files);

  TCanvas *slice_canvas = new TCanvas(
      ("c_" + output_base + "_slices").Data(), "", 50, 300, 1600, 2400);
  slice_canvas->Divide(2, 4);
  TF1 *slice_fit = new TF1(("f_" + output_base).Data(), "gaus", -0.5, 0.5);

  for(int i = 0; i < 4; ++i)
    {
      const TrackVariable &variable = variables[i / 2];
      const int residual = i % 2;
      // Bins 1..N, not the default 0..-1: with grouping, starting at the
      // underflow bin shifts every group of 4 down by one bin relative to
      // the output bin it is drawn in. FitSlicesY only uses the DCA bins
      // within the y-axis range, so the display range of the 2D plot is
      // removed for the fit and restored afterwards.
      hists[i]->GetYaxis()->SetRange(0, 0);
      hists[i]->FitSlicesY(slice_fit, 1, hists[i]->GetNbinsX(), 10, "QNRG4");
      hists[i]->GetYaxis()->SetRangeUser(-dca_2d_axis_limit[residual],
                                         dca_2d_axis_limit[residual]);
      TH1D *h_mean = (TH1D*)gDirectory->Get(
          Form("%s_1", hists[i]->GetName()));
      TH1D *h_sigma = (TH1D*)gDirectory->Get(
          Form("%s_2", hists[i]->GetName()));
      if(!h_mean || !h_sigma)
        {
          std::cerr << "WARNING: FitSlicesY output histograms for "
                    << hists[i]->GetName() << " were not created."
                    << std::endl;
          continue;
        }

      h_mean->SetTitle((hist_titles[i] + ": Gaussian mean").Data());
      h_mean->GetXaxis()->SetTitle(variable.axis_title);
      h_mean->GetYaxis()->SetTitle(
          Form("%s mean [mm]", residual_kinds[residual]));
      h_sigma->SetTitle((hist_titles[i] + ": Gaussian #sigma").Data());
      h_sigma->GetXaxis()->SetTitle(variable.axis_title);
      h_sigma->GetYaxis()->SetTitle(
          Form("%s #sigma [mm]", residual_kinds[residual]));
      const double mean_limit = mean_axis_limit[residual];
      const double sigma_max = sigma_axis_max[residual];
      h_mean->SetMinimum(-mean_limit);
      h_mean->SetMaximum(mean_limit);
      h_sigma->SetMinimum(0.0);
      h_sigma->SetMaximum(sigma_max);
      h_mean->GetYaxis()->SetTitleOffset(y_title_offset);
      h_sigma->GetYaxis()->SetTitleOffset(y_title_offset);

      slice_canvas->cd(2 * i + 1);
      gPad->SetLeftMargin(left_margin);
      h_mean->DrawCopy();
      mark_off_scale_slices(h_mean, -mean_limit, mean_limit);
      draw_fit_range_box(slice_fit);
      draw_selection_box(selection_text);
      if(fit_phi_modulation && i == 2)
        {
          fit_and_draw_phi_modulation(h_mean);
        }

      slice_canvas->cd(2 * i + 2);
      gPad->SetLeftMargin(left_margin);
      h_sigma->DrawCopy();
      mark_off_scale_slices(h_sigma, 0.0, sigma_max);
      draw_fit_range_box(slice_fit);
      draw_selection_box(selection_text);
    }

  add_to_combined_pdf(slice_canvas, output_base + "_slices",
                      save_individual_files);
}

void plot_fits_vtx()
{
  bool do_plot_layers = true;
  bool do_plot_si_slice = true;
  // Every canvas is a page of kCombinedPdf; with save_individual_files it is
  // also saved as its own PDF and PNG.
  bool save_individual_files = false;
  // y ranges [mm] of the Gaussian mean (+-limit) and sigma (0 to max) slice
  // plots, {DCA_rphi, DCA_z}. Slices outside the range are marked with
  // arrows and listed in the terminal.
  // vertex_residuals_vs_momentum_slices (p and pT, all tracks):
  const double momentum_mean_axis_limit[2] = {0.020, 0.020};
  const double momentum_sigma_axis_max[2] = {0.100, 0.100};
  // vertex_residuals_vs_etaphi_slices (eta and phi, pT >= eta_phi_minimum_pt):
  const double etaphi_mean_axis_limit[2] = {0.020, 0.020};
  const double etaphi_sigma_axis_max[2] = {0.100, 0.100};
  // 1D DCA distributions on the vertex_residuals page, {DCA_rphi, DCA_z}:
  // histogram range +-limit [mm] with nbins bins (entries outside the range
  // go to under/overflow and are not shown or fitted), Gaussian fit within
  // +-fit_limit [mm].
  const double dca_1d_hist_limit[2] = {0.2, 0.2};
  const int dca_1d_nbins[2] = {400, 400};
  const double dca_1d_fit_limit[2] = {0.05, 0.05};
  // DCA axis range [mm] shown in all 2D DCA vs phi/p/pT/eta plots, +-limit,
  // {DCA_rphi, DCA_z}. Display only: the 2D histograms are filled within
  // +-2 mm, so limits above 2.0 have no effect and the Gaussian slice fits
  // are unchanged.
  const double dca_2d_axis_limit[2] = {0.20, 0.20};
  //const double* pt_bins[20] = {0,0.1,0.2,0.3,0.4,0.6};
  gStyle->SetOptStat(1);
  gStyle->SetOptFit(1);


  TCut trcut("");
  //TCut phicut("fabs(track_phi)>0.1");
  TCut lyrcut_intt("layer > 2 && layer < 7");
  TCut lyrcut_mvtx("layer >= 0 && layer < 3");
  TCut lyr0("layer ==0");
  TCut lyr1("layer ==1");
  TCut lyr2("layer ==2");
  TCut lyr34("layer ==3||layer ==4");
  TCut lyr56("layer==5||layer==6");
  TCut valid_stavechip("stave >= 0 && chip >= 0");

  //TCut z1("zglob < -8");
  //TCut z2("zglob > -8 && zglob < 0");
  //TCut z3("zglob > 0 && zglob < 8");
  //TCut z4("zglob > 8");

  //TCut cross1("quality < 1000");
  //TCut cross1vtx("quality < 1000&&fabs(track_eta)<.75");
  //TCut cross2("crossing==0");
  //TCut pos_charge("charge > 0");
  //TCut neg_charge("charge < 0");

  TChain *ntp = new TChain("ntp");
  TChain *track_ntp = new TChain("track_ntp");

  std::ifstream infile("ntuple_list.txt");
  if(!infile.is_open())
    {
      std::cerr << "ERROR: could not open ntuple_list.txt in the current directory."
                << std::endl;
      return;
    }

  std::string filename;
  int files_added = 0;
  int track_files_added = 0;
  while(infile >> filename)
    {	      
      std::cout << "Adding " << filename << std::endl;
      const int add_status = ntp->Add(filename.c_str());
      const int track_add_status = track_ntp->Add(filename.c_str());
      if(add_status > 0)
        {
          files_added += add_status;
        }
      else
        {
          std::cerr << "WARNING: failed to add " << filename << std::endl;
        }

      if(track_add_status > 0)
        {
          track_files_added += track_add_status;
        }
      else
        {
          std::cerr << "WARNING: track_ntp is absent in " << filename
                    << std::endl;
        }
    }

  if(files_added == 0)
    {
      std::cerr << "ERROR: no ROOT files were added to the ntp chain."
                << std::endl;
      return;
    }

  const Long64_t nentries = ntp->GetEntries();
  if(nentries <= 0)
    {
      std::cerr << "ERROR: the ntp chain contains no entries." << std::endl;
      return;
    }

  ntp->LoadTree(0);
  const std::vector<std::string> required_branches = {
      "layer", "residualX", "residualY", "clusphi", "stave", "chip"};

  for(const auto& branch_name : required_branches)
    {
      if(!ntp->GetBranch(branch_name.c_str()))
        {
          std::cerr << "ERROR: required branch '" << branch_name
                    << "' is missing from tree ntp." << std::endl;
          return;
        }
    }

  std::cout << "Loaded " << files_added << " file(s) with " << nentries
            << " ntp entries." << std::endl;

  bool have_track_ntp = track_files_added > 0 && track_ntp->GetEntries() > 0;
  const std::vector<std::string> required_track_branches = {
      "residual_vertexX", "residual_vertexY", "track_phi"};

  if(have_track_ntp)
    {
      track_ntp->LoadTree(0);
      for(const auto& branch_name : required_track_branches)
        {
          if(!track_ntp->GetBranch(branch_name.c_str()))
            {
              std::cerr << "WARNING: required branch '" << branch_name
                        << "' is missing from tree track_ntp; skipping vertex plots."
                        << std::endl;
              have_track_ntp = false;
              break;
            }
        }
    }

  if(have_track_ntp)
    {
      std::cout << "Loaded " << track_files_added << " file(s) with "
                << track_ntp->GetEntries() << " track_ntp entries."
                << std::endl;
    }
  else
    {
      std::cerr << "WARNING: no usable track_ntp tree found; vertex-residual "
                << "plots will be skipped." << std::endl;
    }

  // Detector-specific identifiers. The branch names are singular in ntp.
  ntp->SetAlias("mvtx_stavechip", "9*stave + chip");

  // stave = INTT ladder phi, chip = ladder z
  ntp->SetAlias("intt_stavechip", "4*stave + chip");

  if(have_track_ntp)
    {
      TCut vertex_cut(
          "residual_vertexX == residual_vertexX && "
          "residual_vertexY == residual_vertexY");

      TCanvas *c_vtx = new TCanvas(
          "c_vtx", "Track-to-vertex residuals", 50, 300, 1600, 1600);
      c_vtx->Divide(2, 2);

      c_vtx->cd(1);
      TH2D *h_vtx_x_phi = new TH2D(
          "h_vtx_x_phi", "Transverse track-to-vertex residual vs #phi",
          200, -3.5, 3.5, 400, -2.0, 2.0);
      track_ntp->Draw(
          "residual_vertexX:track_phi>>h_vtx_x_phi", vertex_cut, "colz");
      h_vtx_x_phi->GetXaxis()->SetTitle("Track #phi [rad]");
      h_vtx_x_phi->GetYaxis()->SetTitle("Transverse DCA residual [mm]");
      h_vtx_x_phi->GetYaxis()->SetRangeUser(-dca_2d_axis_limit[0],
                                            dca_2d_axis_limit[0]);

      c_vtx->cd(2);
      TH2D *h_vtx_y_phi = new TH2D(
          "h_vtx_y_phi", "Longitudinal track-to-vertex residual vs #phi",
          200, -3.5, 3.5, 400, -2.0, 2.0);
      track_ntp->Draw(
          "residual_vertexY:track_phi>>h_vtx_y_phi", vertex_cut, "colz");
      h_vtx_y_phi->GetXaxis()->SetTitle("Track #phi [rad]");
      h_vtx_y_phi->GetYaxis()->SetTitle("Longitudinal DCA residual [mm]");
      h_vtx_y_phi->GetYaxis()->SetRangeUser(-dca_2d_axis_limit[1],
                                            dca_2d_axis_limit[1]);

      c_vtx->cd(3);
      TH1D *h_vtx_x = new TH1D(
          "h_vtx_x", "Transverse track-to-vertex residual",
          dca_1d_nbins[0], -dca_1d_hist_limit[0], dca_1d_hist_limit[0]);
      track_ntp->Draw("residual_vertexX>>h_vtx_x", vertex_cut);
      h_vtx_x->GetXaxis()->SetTitle("Transverse DCA residual [mm]");
      TF1 *f_vtx_x = new TF1("f_vtx_x", "gaus",
                             -dca_1d_fit_limit[0], dca_1d_fit_limit[0]);
      fit_gaussian_if_possible(h_vtx_x, f_vtx_x);

      c_vtx->cd(4);
      TH1D *h_vtx_y = new TH1D(
          "h_vtx_y", "Longitudinal track-to-vertex residual",
          dca_1d_nbins[1], -dca_1d_hist_limit[1], dca_1d_hist_limit[1]);
      track_ntp->Draw("residual_vertexY>>h_vtx_y", vertex_cut);
      h_vtx_y->GetXaxis()->SetTitle("Longitudinal DCA residual [mm]");
      TF1 *f_vtx_y = new TF1("f_vtx_y", "gaus",
                             -dca_1d_fit_limit[1], dca_1d_fit_limit[1]);
      fit_gaussian_if_possible(h_vtx_y, f_vtx_y);

      add_to_combined_pdf(c_vtx, "vertex_residuals", save_individual_files);

      // Track-to-vertex residuals versus track p and pT, then versus track
      // eta and phi for tracks above eta_phi_minimum_pt, each as a page of
      // 2D histograms and a page of Gaussian mean and sigma per slice (see
      // draw_vertex_residuals_vs_track_variables). The DCA_xy mean vs phi is
      // fitted with A*sin(phi - phi0) + c.
      const TrackVariable momentum_variables[2] = {
          {"track_p", "p", "p", "Track p [GeV/c]", 100, 0.0, 5.0},
          {"track_pt", "pt", "p_{T}", "Track p_{T} [GeV/c]", 100, 0.0, 5.0}};
      draw_vertex_residuals_vs_track_variables(
          track_ntp, vertex_cut, momentum_variables,
          "vertex_residuals_vs_momentum", "", false, save_individual_files,
          momentum_mean_axis_limit, momentum_sigma_axis_max,
          dca_2d_axis_limit);

      const double eta_phi_minimum_pt = 0.8;  // GeV/c
      const TCut eta_phi_pt_cut(Form("track_pt >= %g", eta_phi_minimum_pt));
      const TString eta_phi_selection =
          Form("p_{T} #geq %g GeV/c", eta_phi_minimum_pt);
      const TrackVariable angle_variables[2] = {
          {"track_eta", "eta_ptcut", "#eta", "Track #eta", 60, -1.5, 1.5},
          {"track_phi", "phi_ptcut", "#phi", "Track #phi [rad]",
           56, -TMath::Pi(), TMath::Pi()}};
      draw_vertex_residuals_vs_track_variables(
          track_ntp, vertex_cut && eta_phi_pt_cut, angle_variables,
          "vertex_residuals_vs_etaphi", eta_phi_selection, true,
          save_individual_files, etaphi_mean_axis_limit,
          etaphi_sigma_axis_max, dca_2d_axis_limit);
    }

//TCanvas *c1 = new TCanvas("c1","",50,50,800,800);
// 
//  ntp->Draw("yglob:xglob",trcut);
//  //ntp->Draw("yglob:xglob >> hglob",trcut);
//  //ntp->Draw("yfit:xfit >> hfit",trcut);
//  //ntp->Draw("sensy:sensx >> hsens",trcut);
//  //ntp->Draw("pcay:pcax >> hpoint",trcut);
//  //ntp->Draw("(pcay+tangy):(pcax+tangx) >> htang",trcut);
//
//  //hglob->DrawCopy();
//  c1->SaveAs("global_xy.pdf");
//  c1->SaveAs("global_xy.png");
//  c1->Close();
//  TCanvas *c1 = new TCanvas("c1","",50,300,1600,1600);
//  c1->Divide(2,2);
//  c1->cd(1);
//  TH2D *hXb = new TH2D("hXb","Cluster X residual vs phi TPC",200,-3.5,3.5,200,-dX,dX);
//  ntp->Draw("residualX:clusphi>>hXb",trcut && lyrcut_tpc);
//  hXb->GetXaxis()->SetTitle("phi (rad)");
//  hXb->DrawCopy("colz");
//
//  c1->cd(2);
//  TH2D *hYb = new TH2D("hYb","Cluster Y residual vs phi TPC",200,-3.5,3.5,200,-4,4);
//  ntp->Draw("residualY:phi>>hYb", trcut && lyrcut_tpc);
//  hYb->GetXaxis()->SetTitle("phi (rad)");
//  hYb->DrawCopy("colz");
//
//  c1->cd(3);
//  TH1D *hX1b = new TH1D("hX1b","Cluster X residual TPC",200,-0.05,0.05);
//  ntp->Draw("residualX>>hX1b",trcut && lyrcut_tpc);
//  hX1b->GetXaxis()->SetTitle("X residual [mm]");
//  hX1b->DrawCopy("colz");
//
//  //  TF1 *fg5 = new TF1("fg5","gaus",-0.6,0.6);
//  // hX1a->Fit(fg5,"R");
//
//  c1->cd(4);
//  TH1D *hY1b = new TH1D("hY1b","Cluster Y residual TPC",200,-4,4);
//  ntp->Draw("residualY>>hY1b",trcut && lyrcut_tpc);
//  hY1b->GetXaxis()->SetTitle("Y residual [mm]");
//  hY1b->DrawCopy("colz");

//  TCanvas *c2 = new TCanvas("c2","",50,300,1600,1600);
//  c2->Divide(2,2);
//
//  c2->cd(1);
//  TH2D *hXa = new TH2D("hXa","Cluster X residual vs phi INTT",200,-3.5,3.5,200,-dX,dX);
//  ntp->Draw("residualX:clusphi>>hXa",trcut && lyrcut_intt);
//  hXa->GetXaxis()->SetTitle("phi (rad)");
//  hXa->DrawCopy("colz");
//
//  c2->cd(2);
//  TH2D *hYa = new TH2D("hYa","Cluster Y residual vs phi INTT",200,-3.5,3.5,200,-4,4);
//  ntp->Draw("residualY:phi>>hYa", trcut && lyrcut_intt);
//  hYa->GetXaxis()->SetTitle("phi (rad)");
//  hYa->DrawCopy("colz");
//
//  c2->cd(3);
//  TH1D *hX1a = new TH1D("hX1a","Cluster X residual INTT",200,-0.05,0.05);
//  ntp->Draw("residualX>>hX1a",trcut && lyrcut_intt);
//  hX1a->GetXaxis()->SetTitle("X residual [mm]");
//  hX1a->DrawCopy("colz");
//
//  //  TF1 *fg5 = new TF1("fg5","gaus",-0.6,0.6);
//  // hX1a->Fit(fg5,"R");
//
//  c2->cd(4);
//  TH1D *hY1a = new TH1D("hY1a","Cluster Y residual INTT",200,-4,4);
//  ntp->Draw("residualY>>hY1a",trcut && lyrcut_intt);
//  hY1a->GetXaxis()->SetTitle("Y residual [mm]");
//  hY1a->DrawCopy("colz");


  TCanvas *c3 = new TCanvas("c3","",50,300,1600,1600);
  c3->Divide(2,2);

  c3->cd(1);
  TH2D *hX = new TH2D("hX","Cluster X residual vs phi MVTX",200,-3.5,3.5,200,-0.05,0.05);
  ntp->Draw("residualX:clusphi>>hX",trcut && lyrcut_mvtx);
  hX->GetXaxis()->SetTitle("phi (rad)");
  hX->Draw("colz");

  c3->cd(2);
  TH2D *hY = new TH2D("hY","Cluster Y residual vs phi MVTX",200,-3.5,3.5,200,-0.05,0.05);
  ntp->Draw("residualY:clusphi>>hY", trcut && lyrcut_mvtx);
  hY->GetXaxis()->SetTitle("phi (rad)");
  hY->Draw("colz");

  c3->cd(3);
  TH1D *hX1 = new TH1D("hX1","Cluster X residual MVTX",500,-0.05,0.05);
  ntp->Draw("residualX>>hX1",trcut && lyrcut_mvtx);
  hX1->GetXaxis()->SetTitle("X residual [mm]");
  hX1->Draw();

  TF1 *fg3 = new TF1("fg3","gaus",-0.015,0.015);
  fit_gaussian_if_possible(hX1, fg3);

  c3->cd(4);
  TH1D *hY1 = new TH1D("hY1","Cluster Y residual MVTX",500, -0.05,0.05);
  ntp->Draw("residualY>>hY1",trcut && lyrcut_mvtx);
  hY1->GetXaxis()->SetTitle("Y residual [mm]");
  hY1->Draw();

  TF1 *fg4 = new TF1("fg4","gaus",-0.015,0.015);
  fit_gaussian_if_possible(hY1, fg4);
  add_to_combined_pdf(c3, "mvtx_plot", save_individual_files);

  // MVTX residuals versus the combined stave-chip identifier.
  // The x-axis has unit-width bins centered on the integer identifier values.
  TCanvas *c3_stavechip = new TCanvas("c3_stavechip","",50,300,1600,800);
  c3_stavechip->Divide(2,1);

  c3_stavechip->cd(1);
  TH2D *hX_stavechip_mvtx = new TH2D(
      "hX_stavechip_mvtx",
      "Cluster X residual vs stave-chip ID MVTX",
      182,-0.5,181.5,200,-0.05,0.05);
  ntp->Draw("residualX:mvtx_stavechip>>hX_stavechip_mvtx",
            trcut && lyrcut_mvtx && valid_stavechip);
  hX_stavechip_mvtx->GetXaxis()->SetTitle("9 #times stave + chip");
  hX_stavechip_mvtx->GetYaxis()->SetTitle("X residual [mm]");
  hX_stavechip_mvtx->Draw("colz");

  c3_stavechip->cd(2);
  TH2D *hY_stavechip_mvtx = new TH2D(
      "hY_stavechip_mvtx",
      "Cluster Y residual vs stave-chip ID MVTX",
      182,-0.5,181.5,200,-0.05,0.05);
  ntp->Draw("residualY:mvtx_stavechip>>hY_stavechip_mvtx",
            trcut && lyrcut_mvtx && valid_stavechip);
  hY_stavechip_mvtx->GetXaxis()->SetTitle("9 #times stave + chip");
  hY_stavechip_mvtx->GetYaxis()->SetTitle("Y residual [mm]");
  hY_stavechip_mvtx->Draw("colz");

  add_to_combined_pdf(c3_stavechip, "mvtx_plot_stavechip", save_individual_files);

  // MVTX X/Y residuals versus 9*stave+chip, saved separately by layer.
  if(do_plot_layers)
  {
    const int mvtx_layers[3] = {0, 1, 2};
    const int mvtx_nbins[3] = {110, 146, 182};
    const double mvtx_xmax[3] = {109.5, 145.5, 181.5};

    for(int i = 0; i < 3; ++i)
    {
      const int layer = mvtx_layers[i];
      const std::string layer_string = std::to_string(layer);
      const std::string canvas_name =
          "c_residual_stavechip_mvtx_L" + layer_string;

      TCanvas *canvas = new TCanvas(
          canvas_name.c_str(), "", 50, 300, 1600, 800);
      canvas->Divide(2, 1);

      const std::string layer_expression = "layer==" + layer_string;
      TCut layer_cut(layer_expression.c_str());

      const std::string hist_x_name =
          "hXResidualStaveChipMVTX_L" + layer_string;
      const std::string hist_x_title =
          "Cluster X residual vs stave-chip ID, MVTX layer " + layer_string;
      TH2D *hist_x = new TH2D(
          hist_x_name.c_str(), hist_x_title.c_str(),
          mvtx_nbins[i], -0.5, mvtx_xmax[i], 200, -0.05, 0.05);
      hist_x->GetXaxis()->SetTitle("9 #times stave + chip");
      hist_x->GetYaxis()->SetTitle("Cluster X residual [mm]");

      canvas->cd(1);
      const std::string draw_x =
          "residualX:mvtx_stavechip>>" + hist_x_name;
      ntp->Draw(
          draw_x.c_str(),
          trcut && valid_stavechip && layer_cut,
          "colz");

      const std::string hist_y_name =
          "hYResidualStaveChipMVTX_L" + layer_string;
      const std::string hist_y_title =
          "Cluster Y residual vs stave-chip ID, MVTX layer " + layer_string;
      TH2D *hist_y = new TH2D(
          hist_y_name.c_str(), hist_y_title.c_str(),
          mvtx_nbins[i], -0.5, mvtx_xmax[i], 200, -0.05, 0.05);
      hist_y->GetXaxis()->SetTitle("9 #times stave + chip");
      hist_y->GetYaxis()->SetTitle("Cluster Y residual [mm]");

      canvas->cd(2);
      const std::string draw_y =
          "residualY:mvtx_stavechip>>" + hist_y_name;
      ntp->Draw(
          draw_y.c_str(),
          trcut && valid_stavechip && layer_cut,
          "colz");

      const std::string output_base =
          "mvtx_residual_vs_stavechip_L" + layer_string;
      add_to_combined_pdf(canvas, output_base.c_str(), save_individual_files);
    }
  }

  // Cluster phi versus 9*stave+chip for each MVTX layer.
  if(do_plot_layers)
  {
    const int mvtx_layers[3] = {0, 1, 2};

    // Integer x-axis ranges (endpoints included):
    // L0: [0,109], L1: [0,145], L2: [0,181].
    const int mvtx_nbins[3] = {110, 146, 182};
    const double mvtx_xmax[3] = {109.5, 145.5, 181.5};

    TCanvas *c_phi_stavechip_mvtx = new TCanvas(
        "c_phi_stavechip_mvtx", "", 50, 300, 1800, 600);
    c_phi_stavechip_mvtx->Divide(3, 1);

    for(int i = 0; i < 3; ++i)
    {
      const int layer = mvtx_layers[i];

      const std::string hist_name =
          "hPhiStaveChipMVTX_L" + std::to_string(layer);
      const std::string hist_title =
          "Cluster #phi vs stave-chip ID, MVTX layer "
          + std::to_string(layer);

      TH2D *hist = new TH2D(
          hist_name.c_str(), hist_title.c_str(),
          mvtx_nbins[i], -0.5, mvtx_xmax[i],
          200, -3.5, 3.5);

      const std::string layer_expression =
          "layer==" + std::to_string(layer);
      TCut layer_cut(layer_expression.c_str());

      const std::string draw_expression =
          "clusphi:mvtx_stavechip>>" + hist_name;

      c_phi_stavechip_mvtx->cd(i + 1);
      ntp->Draw(
          draw_expression.c_str(),
          trcut && valid_stavechip && layer_cut,
          "colz");

      hist->GetXaxis()->SetTitle("9 #times stave + chip");
      hist->GetYaxis()->SetTitle("Cluster #phi [rad]");
    }

    add_to_combined_pdf(c_phi_stavechip_mvtx, "mvtx_phi_vs_stavechip_layers",
                        save_individual_files);

    // Number of entries versus 9*stave+chip for each MVTX layer.
    TCanvas *c_entries_stavechip_mvtx = new TCanvas(
        "c_entries_stavechip_mvtx", "", 50, 300, 1800, 600);
    c_entries_stavechip_mvtx->Divide(3, 1);

    for(int i = 0; i < 3; ++i)
    {
      const int layer = mvtx_layers[i];

      const std::string hist_name =
          "hEntriesStaveChipMVTX_L" + std::to_string(layer);
      const std::string hist_title =
          "Number of entries vs stave-chip ID, MVTX layer "
          + std::to_string(layer);

      TH1D *hist = new TH1D(
          hist_name.c_str(), hist_title.c_str(),
          mvtx_nbins[i], -0.5, mvtx_xmax[i]);

      const std::string layer_expression =
          "layer==" + std::to_string(layer);
      TCut layer_cut(layer_expression.c_str());

      const std::string draw_expression =
          "mvtx_stavechip>>" + hist_name;

      c_entries_stavechip_mvtx->cd(i + 1);
      ntp->Draw(
          draw_expression.c_str(),
          trcut && valid_stavechip && layer_cut,
          "hist");

      hist->GetXaxis()->SetTitle("9 #times stave + chip");
      hist->GetYaxis()->SetTitle("Number of entries");
      hist->SetMinimum(0.0);
    }

    add_to_combined_pdf(c_entries_stavechip_mvtx,
                        "mvtx_entries_vs_stavechip_layers",
                        save_individual_files);
  }

  if(do_plot_layers){
    TCanvas *c30 = new TCanvas("c30","",50,300,1600,1600);
    c30->Divide(2,2);

    c30->cd(1);
    TH2D *h0X = new TH2D("h0X","Cluster X residual vs phi MVTX L0",200,-3.5,3.5,200,-0.05,0.05);
    ntp->Draw("residualX:clusphi>>h0X",trcut && lyr0);
    h0X->GetXaxis()->SetTitle("phi (rad)");
    h0X->Draw("colz");

    c30->cd(2);
    TH2D *h0Y = new TH2D("h0Y","Cluster Y residual vs phi MVTX L0",200,-3.5,3.5,200,-0.05,0.05);
    ntp->Draw("residualY:clusphi>>h0Y", trcut && lyr0);
    h0Y->GetXaxis()->SetTitle("phi (rad)");
    h0Y->Draw("colz");

    c30->cd(3);
    TH1D *h0X1 = new TH1D("h0X1","Cluster X residual MVTX L0",500,-0.05,0.05);
    ntp->Draw("residualX>>h0X1",trcut && lyr0);
    h0X1->GetXaxis()->SetTitle("X residual [mm]");
    h0X1->Draw();

    //TF1 *fg3 = new TF1("fg3","gaus",-0.0015,0.0015);
    fit_gaussian_if_possible(h0X1, fg3);

    c30->cd(4);
    TH1D *h0Y1 = new TH1D("h0Y1","Cluster Y residual MVTX L0",500, -0.05,0.05);
    ntp->Draw("residualY>>h0Y1",trcut && lyr0);
    h0Y1->GetXaxis()->SetTitle("Y residual [mm]");
    h0Y1->Draw();

    //TF1 *fg4 = new TF1("fg4","gaus",-0.0015,0.0015);
    fit_gaussian_if_possible(h0Y1, fg4);
    add_to_combined_pdf(c30, "mvtx_plot_l0", save_individual_files);


    TCanvas *c31 = new TCanvas("c31","",50,300,1600,1600);
    c31->Divide(2,2);

    c31->cd(1);
    TH2D *h1X = new TH2D("h1X","Cluster X residual vs phi MVTX L1",200,-3.5,3.5,200,-0.05,0.05);
    ntp->Draw("residualX:clusphi>>h1X",trcut && lyr1);
    h1X->GetXaxis()->SetTitle("phi (rad)");
    h1X->Draw("colz");

    c31->cd(2);
    TH2D *h1Y = new TH2D("h1Y","Cluster Y residual vs phi MVTX L1",200,-3.5,3.5,200,-0.05,0.05);
    ntp->Draw("residualY:clusphi>>h1Y", trcut && lyr1);
    h1Y->GetXaxis()->SetTitle("phi (rad)");
    h1Y->Draw("colz");

    c31->cd(3);
    TH1D *h1X1 = new TH1D("h1X1","Cluster X residual MVTX L1",500,-0.05,0.05);
    ntp->Draw("residualX>>h1X1",trcut && lyr1);
    h1X1->GetXaxis()->SetTitle("X residual [mm]");
    h1X1->Draw();

    //TF1 *fg3 = new TF1("fg3","gaus",-0.0015,0.0015);
    fit_gaussian_if_possible(h1X1, fg3);

    c31->cd(4);
    TH1D *h1Y1 = new TH1D("h1Y1","Cluster Y residual MVTX L1",500, -0.05,0.05);
    ntp->Draw("residualY>>h1Y1",trcut && lyr1);
    h1Y1->GetXaxis()->SetTitle("Y residual [mm]");
    h1Y1->Draw();

    //TF1 *fg4 = new TF1("fg4","gaus",-0.0015,0.0015);
    fit_gaussian_if_possible(h1Y1, fg4);
    add_to_combined_pdf(c31, "mvtx_plot_l1", save_individual_files);

    TCanvas *c32 = new TCanvas("c32","",50,300,1600,1600);
    c32->Divide(2,2);

    c32->cd(1);
    TH2D *h2X = new TH2D("h2X","Cluster X residual vs phi MVTX L2",200,-3.5,3.5,200,-0.05,0.05);
    ntp->Draw("residualX:clusphi>>h2X",trcut && lyr2);
    h2X->GetXaxis()->SetTitle("phi (rad)");
    h2X->Draw("colz");

    c32->cd(2);
    TH2D *h2Y = new TH2D("h2Y","Cluster Y residual vs phi MVTX L2",200,-3.5,3.5,200,-0.05,0.05);
    ntp->Draw("residualY:clusphi>>h2Y", trcut && lyr2);
    h2Y->GetXaxis()->SetTitle("phi (rad)");
    h2Y->Draw("colz");

    c32->cd(3);
    TH1D *h2X1 = new TH1D("h2X1","Cluster X residual MVTX L2",500,-0.05,0.05);
    ntp->Draw("residualX>>h2X1",trcut && lyr2);
    h2X1->GetXaxis()->SetTitle("X residual [mm]");
    h2X1->Draw();

    //TF1 *fg3 = new TF1("fg3","gaus",-0.0015,0.0015);
    fit_gaussian_if_possible(h2X1, fg3);

    c32->cd(4);
    TH1D *h2Y1 = new TH1D("h2Y1","Cluster Y residual MVTX L2",500, -0.05,0.05);
    ntp->Draw("residualY>>h2Y1",trcut && lyr2);
    h2Y1->GetXaxis()->SetTitle("Y residual [mm]");
    h2Y1->Draw();

    //TF1 *fg4 = new TF1("fg4","gaus",-0.0015,0.0015);
    fit_gaussian_if_possible(h2Y1, fg4);
    add_to_combined_pdf(c32, "mvtx_plot_l2", save_individual_files);
    if(do_plot_si_slice){
      TCanvas *c300 = new TCanvas("c300","",50,300,1800,2000);
      c300->Divide(3,4);

      TF1 *fg300 = new TF1("fg300","gaus",-0.013,0.013); //MVTX X residual fit range
      TF1 *fg300Y = new TF1("fg300Y","gaus",-0.013,0.013); //MVTX Y residual fit range

      h0X->FitSlicesY(fg300, 0, -1, 1, "QNRG4");
      TH1D *h0X_1 = (TH1D*)gDirectory->Get("h0X_1");
      TH1D *h0X_2 = (TH1D*)gDirectory->Get("h0X_2");
      h1X->FitSlicesY(fg300, 0, -1, 1, "QNRG4");
      TH1D *h1X_1 = (TH1D*)gDirectory->Get("h1X_1");
      TH1D *h1X_2 = (TH1D*)gDirectory->Get("h1X_2");
      h2X->FitSlicesY(fg300, 0, -1, 1, "QNRG4");
      TH1D *h2X_1 = (TH1D*)gDirectory->Get("h2X_1");
      TH1D *h2X_2 = (TH1D*)gDirectory->Get("h2X_2");
      h0Y->FitSlicesY(fg300Y, 0, -1, 1, "QNRG4");
      TH1D *h0Y_1 = (TH1D*)gDirectory->Get("h0Y_1");
      TH1D *h0Y_2 = (TH1D*)gDirectory->Get("h0Y_2");
      h1Y->FitSlicesY(fg300Y, 0, -1, 1, "QNRG4");
      TH1D *h1Y_1 = (TH1D*)gDirectory->Get("h1Y_1");
      TH1D *h1Y_2 = (TH1D*)gDirectory->Get("h1Y_2");
      h2Y->FitSlicesY(fg300Y, 0, -1, 1, "QNRG4");
      TH1D *h2Y_1 = (TH1D*)gDirectory->Get("h2Y_1");
      TH1D *h2Y_2 = (TH1D*)gDirectory->Get("h2Y_2");

      if(!h0X_1 || !h0X_2 || !h1X_1 || !h1X_2 || !h2X_1 || !h2X_2 ||
         !h0Y_1 || !h0Y_2 || !h1Y_1 || !h1Y_2 || !h2Y_1 || !h2Y_2)
      {
        std::cerr << "WARNING: one or more MVTX FitSlicesY output histograms "
                  << "were not created; skipping the MVTX slice canvas."
                  << std::endl;
      }
      else
      {
      h0X_1->SetTitle("X-residual Mean MVTX Layer 0");
      h0X_2->SetTitle("X-residual Sigma MVTX Layer 0");
      h1X_1->SetTitle("X-residual Mean MVTX Layer 1");
      h1X_2->SetTitle("X-residual Sigma MVTX Layer 1");
      h2X_1->SetTitle("X-residual Mean MVTX Layer 2");
      h2X_2->SetTitle("X-residual Sigma MVTX Layer 2");
      h0Y_1->SetTitle("Y-residual Mean MVTX Layer 0");
      h0Y_2->SetTitle("Y-residual Sigma MVTX Layer 0");
      h1Y_1->SetTitle("Y-residual Mean MVTX Layer 1");
      h1Y_2->SetTitle("Y-residual Sigma MVTX Layer 1");
      h2Y_1->SetTitle("Y-residual Mean MVTX Layer 2");
      h2Y_2->SetTitle("Y-residual Sigma MVTX Layer 2");

      c300->cd(1);
      h0X_1->GetYaxis()->SetRangeUser(-0.0065,0.0065);
      h0X_1->DrawCopy();
      draw_fit_range_box(fg300);

      c300->cd(2);
      h1X_1->GetYaxis()->SetRangeUser(-0.0065,0.0065);
      h1X_1->DrawCopy();
      draw_fit_range_box(fg300);

      c300->cd(3);
      h2X_1->GetYaxis()->SetRangeUser(-0.0065,0.0065);
      h2X_1->DrawCopy();
      draw_fit_range_box(fg300);

      c300->cd(4);
      h0X_2->GetYaxis()->SetRangeUser(-0.000,0.01);
      h0X_2->DrawCopy();
      draw_fit_range_box(fg300);

      c300->cd(5);
      h1X_2->GetYaxis()->SetRangeUser(-0.000,0.01);
      h1X_2->DrawCopy();
      draw_fit_range_box(fg300);

      c300->cd(6);
      h2X_2->GetYaxis()->SetRangeUser(-0.000,0.01);
      h2X_2->DrawCopy();
      draw_fit_range_box(fg300);

      c300->cd(7);
      h0Y_1->GetYaxis()->SetRangeUser(-0.0065,0.0065);
      h0Y_1->DrawCopy();
      draw_fit_range_box(fg300Y);

      c300->cd(8);
      h1Y_1->GetYaxis()->SetRangeUser(-0.0065,0.0065);
      h1Y_1->DrawCopy();
      draw_fit_range_box(fg300Y);

      c300->cd(9);
      h2Y_1->GetYaxis()->SetRangeUser(-0.0065,0.0065);
      h2Y_1->DrawCopy();
      draw_fit_range_box(fg300Y);

      c300->cd(10);
      h0Y_2->GetYaxis()->SetRangeUser(0.0,0.01);
      h0Y_2->DrawCopy();
      draw_fit_range_box(fg300Y);

      c300->cd(11);
      h1Y_2->GetYaxis()->SetRangeUser(0.0,0.01);
      h1Y_2->DrawCopy();
      draw_fit_range_box(fg300Y);

      c300->cd(12);
      h2Y_2->GetYaxis()->SetRangeUser(0.0,0.01);
      h2Y_2->DrawCopy();
      draw_fit_range_box(fg300Y);

      add_to_combined_pdf(c300, "mvtx_plot_Slice", save_individual_files);
      }

     
    }
//    if(do_plot_z_bin)
//    {
//      TCanvas *c301 = new TCanvas("c301","",50,300,1600,1600);
//      c301->Divide(2,2);
//      c301->cd(1);
//      TH2D *h0Xb1 = new TH2D("h0Xb1","Cluster X residual vs phi MVTX L0, z<-8cm",200,-3.5,3.5,200,-0.05,0.05);
//      ntp->Draw("residualX:clusphi>>h0Xb1",trcut && lyr2 && z1);
//      h0Xb1->GetXaxis()->SetTitle("phi (rad)");
//      h0Xb1->DrawCopy("colz");
//      c301->cd(2);
//      TH2D *h0Xb2 = new TH2D("h0Xb2","Cluster X residual vs phi MVTX L0, -8cm<z<0cm",200,-3.5,3.5,200,-0.05,0.05);
//      ntp->Draw("residualX:clusphi>>h0Xb2",trcut && lyr2 && z2);
//      h0Xb2->GetXaxis()->SetTitle("phi (rad)");
//      h0Xb2->DrawCopy("colz");
//      c301->cd(3);
//      TH2D *h0Xb3 = new TH2D("h0Xb3","Cluster X residual vs phi MVTX L0, 0cm<z<8cm",200,-3.5,3.5,200,-0.05,0.05);
//      ntp->Draw("residualX:clusphi>>h0Xb3",trcut && lyr2 && z3);
//      h0Xb3->GetXaxis()->SetTitle("phi (rad)");
//      h0Xb3->DrawCopy("colz");
//      c301->cd(4);
//      TH2D *h0Xb4 = new TH2D("h0Xb4","Cluster X residual vs phi MVTX L0, z>8cm",200,-3.5,3.5,200,-0.05,0.05);
//      ntp->Draw("residualX:clusphi>>h0Xb4",trcut && lyr2 && z4);
//      h0Xb4->GetXaxis()->SetTitle("phi (rad)");
//      h0Xb4->DrawCopy("colz");
//      c301->SaveAs("mvtx_plot_zbin.pdf");
//      c301->SaveAs("mvtx_plot_zbin.png");
//    }
//    if(do_plot_crossing_bin)
//    {
//      TCanvas *c321 = new TCanvas("c321","",50,300,1600,1600);
//      c321->Divide(2,2);
//      c321->cd(1);
//      TH2D *h0Xc1 = new TH2D("h0Xc1","Cluster X residual vs phi MVTX L2, crossing 0",200,-3.5,3.5,200,-0.05,0.05);
//      ntp->Draw("residualX:clusphi>>h0Xc1",trcut && lyr2 && cross1);
//      h0Xc1->GetXaxis()->SetTitle("phi (rad)");
//      h0Xc1->DrawCopy("colz");
//      c321->cd(2);
//      TH2D *h0Xc2 = new TH2D("h0Xc2","Cluster X residual vs phi MVTX L2, crossing > 0",200,-3.5,3.5,200,-0.05,0.05);
//      ntp->Draw("residualX:clusphi>>h0Xc2",trcut && lyr2 && cross2);
//      h0Xc2->GetXaxis()->SetTitle("phi (rad)");
//      h0Xc2->DrawCopy("colz");
//
//      c321->cd(3);
//      TH1D *h0Xc3 = new TH1D("h0Xc3","Cluster X residual MVTX L2, crossing 0",500,-0.05,0.05);
//      ntp->Draw("residualX>>h0Xc3",trcut && lyr2 && cross1);
//      h0Xc3->GetXaxis()->SetTitle("X residual [mm]");
//      h0Xc3->DrawCopy("colz");
//      TF1 *fg321 = new TF1("fg321","gaus",-0.0015,0.0015);
//      h0Xc3->Fit(fg321,"R");
//      fg321->Draw("same");
//
//      c321->cd(4);
//      TH1D *h0Xc4 = new TH1D("h0Xc4","Cluster X residual MVTX L2, crossing > 0",500,-0.05,0.05);
//      ntp->Draw("residualX>>h0Xc4",trcut && lyr2 && cross2);
//      h0Xc4->GetXaxis()->SetTitle("X residual [mm]");
//      h0Xc4->DrawCopy("colz");
//      h0Xc4->Fit(fg321,"R");
//      fg321->Draw("same");
//    
//      c321->SaveAs("mvtx_plot_quality.pdf");
//      c321->SaveAs("mvtx_plot_quality.png");
//    }
//    if(do_plot_glob_xy)
//    {
//      TCanvas *c310 = new TCanvas("c310","",50,300,1600, 1200);
//      c310->Divide(3,2);
//      c310->cd(1);
//      TH2D *h0Xc1 = new TH2D("h0Xc1","Cluster x residual vs phi MVTX L0",200,-3.5,3.5,200,-0.05,0.05);
//      ntp->Draw("xfit-xglob:clusphi>>h0Xc1",trcut && lyr0);
//      h0Xc1->GetXaxis()->SetTitle("phi (rad)");
//      h0Xc1->DrawCopy("colz");
//      c310->cd(4);
//      TH2D *h0Xc4 = new TH2D("h0Xc4","Cluster y residual vs phi MVTX L0",200,-3.5,3.5,200,-0.05,0.05);
//      ntp->Draw("yfit-yglob:clusphi>>h0Xc4",trcut && lyr0);
//      h0Xc4->GetXaxis()->SetTitle("phi (rad)");
//      h0Xc4->DrawCopy("colz");
//      c310->cd(2);
//      TH2D *h0Xc2 = new TH2D("h0Xc2","Cluster x residual vs phi MVTX L1",200,-3.5,3.5,200,-0.05,0.05);
//      ntp->Draw("xfit-xglob:clusphi>>h0Xc2",trcut && lyr1);
//      h0Xc2->GetXaxis()->SetTitle("phi (rad)");
//      h0Xc2->DrawCopy("colz");
//      c310->cd(5);
//      TH2D *h0Xc5 = new TH2D("h0Xc5","Cluster y residual vs phi MVTX L1",200,-3.5,3.5,200,-0.05,0.05);
//      ntp->Draw("yfit-yglob:clusphi>>h0Xc5",trcut && lyr1);
//      h0Xc5->GetXaxis()->SetTitle("phi (rad)");
//      h0Xc5->DrawCopy("colz");
//      c310->cd(3);
//      TH2D *h0Xc3 = new TH2D("h0Xc3","Cluster x residual vs phi MVTX L2",200,-3.5,3.5,200,-0.05,0.05);
//      ntp->Draw("xfit-xglob:clusphi>>h0Xc3",trcut && lyr2);
//      h0Xc3->GetXaxis()->SetTitle("phi (rad)");
//      h0Xc3->DrawCopy("colz");
//      c310->cd(6);
//      TH2D *h0Xc6 = new TH2D("h0Xc6","Cluster y residual vs phi MVTX L2",200,-3.5,3.5,200,-0.05,0.05);
//      ntp->Draw("yfit-yglob:clusphi>>h0Xc6",trcut && lyr2);
//      h0Xc6->GetXaxis()->SetTitle("phi (rad)");
//      h0Xc6->DrawCopy("colz");
//
//      c310->SaveAs("mvtx_plot_glob.pdf");
//      c310->SaveAs("mvtx_plot_glob.png");
//    }
//    if(do_plot_charge)
//    {
//      TCanvas *c330 = new TCanvas("c330","",50,300,1600, 1200);
//      c330->Divide(3,2);
//      c330->cd(1);
//      TH2D *h331 = new TH2D("h331","Cluster X residual vs phi MVTX L0, charge +1",200,-3.5,3.5,200,-0.05,0.05);
//      ntp->Draw("residualX:clusphi>>h331",trcut && lyr0 && pos_charge);
//      h331->GetXaxis()->SetTitle("phi (rad)");
//      h331->GetYaxis()->SetTitle("residual [mm]");
//      h331->DrawCopy("colz");
//      c330->cd(2);
//      TH2D *h333 = new TH2D("h333","Cluster X residual vs phi MVTX L1, charge +1",200,-3.5,3.5,200,-0.05,0.05);
//      ntp->Draw("residualX:clusphi>>h333",trcut && lyr1 && pos_charge);
//      h333->GetXaxis()->SetTitle("phi (rad)");
//      h333->GetYaxis()->SetTitle("residual [mm]");
//      h333->DrawCopy("colz");
//      c330->cd(3);
//      TH2D *h335 = new TH2D("h335","Cluster X residual vs phi MVTX L2, charge +1",200,-3.5,3.5,200,-0.05,0.05);
//      ntp->Draw("residualX:clusphi>>h335",trcut && lyr2 && pos_charge);
//      h335->GetXaxis()->SetTitle("phi (rad)");
//      h335->GetYaxis()->SetTitle("residual [mm]");
//      h335->DrawCopy("colz");
//      c330->cd(4);
//      TH2D *h332 = new TH2D("h332","Cluster X residual vs phi MVTX L0, charge -1",200,-3.5,3.5,200,-0.05,0.05);
//      ntp->Draw("residualX:clusphi>>h332",trcut && lyr0 && neg_charge);
//      h332->GetXaxis()->SetTitle("phi (rad)");
//      h332->GetYaxis()->SetTitle("residual [mm]");
//      h332->DrawCopy("colz");
//      c330->cd(5);
//      TH2D *h334 = new TH2D("h334","Cluster X residual vs phi MVTX L1, charge -1",200,-3.5,3.5,200,-0.05,0.05);
//      ntp->Draw("residualX:clusphi>>h334",trcut && lyr1 && neg_charge);
//      h334->GetXaxis()->SetTitle("phi (rad)");
//      h334->GetYaxis()->SetTitle("residual [mm]");
//      h334->DrawCopy("colz");
//      c330->cd(6);
//      TH2D *h336 = new TH2D("h336","Cluster X residual vs phi MVTX L2, charge -1",200,-3.5,3.5,200,-0.05,0.05);
//      ntp->Draw("residualX:clusphi>>h336",trcut && lyr2 && neg_charge);
//      h336->GetXaxis()->SetTitle("phi (rad)");
//      h336->GetYaxis()->SetTitle("residual [mm]");
//      h336->DrawCopy("colz");
//
//      c330->SaveAs("mvtx_plot_glob.pdf");
//      c330->SaveAs("mvtx_plot_glob.png");
//    }
  }

  TCanvas *c4 = new TCanvas("c4","",50,300,1600,1600);
  c4->Divide(2,2);

  c4->cd(1);
  TH2D *hXa = new TH2D("hXa","Cluster X residual vs phi INTT",200,-3.5,3.5,200,-0.6,0.6);
  ntp->Draw("residualX:clusphi>>hXa",trcut && lyrcut_intt);
  hXa->GetXaxis()->SetTitle("phi (rad)");
  hXa->Draw("colz");

  c4->cd(2);
  TH2D *hYa = new TH2D("hYa","Cluster Y residual vs phi INTT",200,-3.5,3.5,200,-20,20);
  ntp->Draw("residualY:clusphi>>hYa", trcut && lyrcut_intt);
  hYa->GetXaxis()->SetTitle("phi (rad)");
  hYa->Draw("colz");

  c4->cd(3);
  TH1D *hX2a = new TH1D("hX2a","Cluster X residual INTT",500,-0.5,0.5);
  ntp->Draw("residualX>>hX2a",trcut && lyrcut_intt);
  hX2a->GetXaxis()->SetTitle("X residual [mm]");
  hX2a->Draw();

  TF1 *fg3a = new TF1("fg3a","gaus",-0.015,0.015);
  fit_gaussian_if_possible(hX2a, fg3a);

  c4->cd(4);
  TH1D *hY1a = new TH1D("hY1a","Cluster Y residual INTT",500, -20, 20);
  ntp->Draw("residualY>>hY1a",trcut && lyrcut_intt);
  hY1a->GetXaxis()->SetTitle("Y residual [mm]");
  hY1a->Draw();

  TF1 *fg4a = new TF1("fg4a","gaus",-1,1);
  fit_gaussian_if_possible(hY1a, fg4a);
  add_to_combined_pdf(c4, "intt_plot", save_individual_files);

  // INTT residuals versus the combined stave-chip identifier.
  TCanvas *c4_stavechip = new TCanvas("c4_stavechip","",50,300,1600,800);
  c4_stavechip->Divide(2,1);

  c4_stavechip->cd(1);
  TH2D *hX_stavechip_intt = new TH2D(
      "hX_stavechip_intt",
      "Cluster X residual vs stave-chip ID INTT",
      66,-0.5,65.5,200,-0.6,0.6);
  ntp->Draw("residualX:intt_stavechip>>hX_stavechip_intt",
            trcut && lyrcut_intt && valid_stavechip);
  hX_stavechip_intt->GetXaxis()->SetTitle("4 #times ladder + chip");
  hX_stavechip_intt->GetYaxis()->SetTitle("X residual [mm]");
  hX_stavechip_intt->Draw("colz");

  c4_stavechip->cd(2);
  TH2D *hY_stavechip_intt = new TH2D(
      "hY_stavechip_intt",
      "Cluster Y residual vs stave-chip ID INTT",
      66,-0.5,65.5,200,-20,20);
  ntp->Draw("residualY:intt_stavechip>>hY_stavechip_intt",
            trcut && lyrcut_intt && valid_stavechip);
  hY_stavechip_intt->GetXaxis()->SetTitle("4 #times ladder + chip");
  hY_stavechip_intt->GetYaxis()->SetTitle("Y residual [mm]");
  hY_stavechip_intt->Draw("colz");

  add_to_combined_pdf(c4_stavechip, "intt_plot_stavechip", save_individual_files);

  // INTT X/Y residuals versus 4*chip+stave, saved separately by layer.
  if(do_plot_layers)
  {
    const int intt_layers[4] = {3, 4, 5, 6};
    const int intt_nbins[4] = {48, 48, 66, 66};
    const double intt_xmax[4] = {47.5, 47.5, 65.5, 65.5};

    for(int i = 0; i < 4; ++i)
    {
      const int layer = intt_layers[i];
      const std::string layer_string = std::to_string(layer);
      const std::string canvas_name =
          "c_residual_stavechip_intt_L" + layer_string;

      TCanvas *canvas = new TCanvas(
          canvas_name.c_str(), "", 50, 300, 1600, 800);
      canvas->Divide(2, 1);

      const std::string layer_expression = "layer==" + layer_string;
      TCut layer_cut(layer_expression.c_str());

      const std::string hist_x_name =
          "hXResidualStaveChipINTT_L" + layer_string;
      const std::string hist_x_title =
          "Cluster X residual vs stave-chip ID, INTT layer " + layer_string;
      TH2D *hist_x = new TH2D(
          hist_x_name.c_str(), hist_x_title.c_str(),
          intt_nbins[i], -0.5, intt_xmax[i], 200, -0.6, 0.6);
      hist_x->GetXaxis()->SetTitle("4 #times ladder + chip");
      hist_x->GetYaxis()->SetTitle("Cluster X residual [mm]");

      canvas->cd(1);
      const std::string draw_x =
          "residualX:intt_stavechip>>" + hist_x_name;
      ntp->Draw(
          draw_x.c_str(),
          trcut && valid_stavechip && layer_cut,
          "colz");

      const std::string hist_y_name =
          "hYResidualStaveChipINTT_L" + layer_string;
      const std::string hist_y_title =
          "Cluster Y residual vs stave-chip ID, INTT layer " + layer_string;
      TH2D *hist_y = new TH2D(
          hist_y_name.c_str(), hist_y_title.c_str(),
          intt_nbins[i], -0.5, intt_xmax[i], 200, -20.0, 20.0);
      hist_y->GetXaxis()->SetTitle("4 #times ladder + chip");
      hist_y->GetYaxis()->SetTitle("Cluster Y residual [mm]");

      canvas->cd(2);
      const std::string draw_y =
          "residualY:intt_stavechip>>" + hist_y_name;
      ntp->Draw(
          draw_y.c_str(),
          trcut && valid_stavechip && layer_cut,
          "colz");

      const std::string output_base =
          "intt_residual_vs_stavechip_L" + layer_string;
      add_to_combined_pdf(canvas, output_base.c_str(), save_individual_files);
    }
  }

  // Cluster phi versus 4*chip+stave for each INTT layer.
  if(do_plot_layers)
  {
    const int intt_layers[4] = {3, 4, 5, 6};

    // Integer x-axis ranges (endpoints included):
    // L3/L4: [0,45], L5/L6: [0,65].
    const int intt_nbins[4] = {50, 50, 66, 66};
    const double intt_xmax[4] = {49.5, 49.5, 65.5, 65.5};

    TCanvas *c_phi_stavechip_intt = new TCanvas(
        "c_phi_stavechip_intt", "", 50, 300, 1600, 1200);
    c_phi_stavechip_intt->Divide(2, 2);

    for(int i = 0; i < 4; ++i)
    {
      const int layer = intt_layers[i];

      const std::string hist_name =
          "hPhiStaveChipINTT_L" + std::to_string(layer);
      const std::string hist_title =
          "Cluster #phi vs stave-chip ID, INTT layer "
          + std::to_string(layer);

      TH2D *hist = new TH2D(
          hist_name.c_str(), hist_title.c_str(),
          intt_nbins[i], -0.5, intt_xmax[i],
          200, -3.5, 3.5);

      const std::string layer_expression =
          "layer==" + std::to_string(layer);
      TCut layer_cut(layer_expression.c_str());

      const std::string draw_expression =
          "clusphi:intt_stavechip>>" + hist_name;

      c_phi_stavechip_intt->cd(i + 1);
      ntp->Draw(
          draw_expression.c_str(),
          trcut && valid_stavechip && layer_cut,
          "colz");

      hist->GetXaxis()->SetTitle("4 #times ladder + chip");
      hist->GetYaxis()->SetTitle("Cluster #phi [rad]");
    }

    add_to_combined_pdf(c_phi_stavechip_intt, "intt_phi_vs_stavechip_layers",
                        save_individual_files);

    // Number of entries versus 4*chip+stave for each INTT layer.
    TCanvas *c_entries_stavechip_intt = new TCanvas(
        "c_entries_stavechip_intt", "", 50, 300, 1600, 1200);
    c_entries_stavechip_intt->Divide(2, 2);

    for(int i = 0; i < 4; ++i)
    {
      const int layer = intt_layers[i];

      const std::string hist_name =
          "hEntriesStaveChipINTT_L" + std::to_string(layer);
      const std::string hist_title =
          "Number of entries vs stave-chip ID, INTT layer "
          + std::to_string(layer);

      TH1D *hist = new TH1D(
          hist_name.c_str(), hist_title.c_str(),
          intt_nbins[i], -0.5, intt_xmax[i]);

      const std::string layer_expression =
          "layer==" + std::to_string(layer);
      TCut layer_cut(layer_expression.c_str());

      const std::string draw_expression =
          "intt_stavechip>>" + hist_name;

      c_entries_stavechip_intt->cd(i + 1);
      ntp->Draw(
          draw_expression.c_str(),
          trcut && valid_stavechip && layer_cut,
          "hist");

      hist->GetXaxis()->SetTitle("4 #times ladder + chip");
      hist->GetYaxis()->SetTitle("Number of entries");
      hist->SetMinimum(0.0);
    }

    add_to_combined_pdf(c_entries_stavechip_intt,
                        "intt_entries_vs_stavechip_layers",
                        save_individual_files);
  }

  if(do_plot_layers){
    TCanvas *c41 = new TCanvas("c41","",50,300,1600,1600);
    c41->Divide(2,2);

    c41->cd(1);
    TH2D *h34Xa = new TH2D("h34Xa","Cluster X residual vs phi INTT L34",200,-3.5,3.5,200,-0.6,0.6);
    ntp->Draw("residualX:clusphi>>h34Xa",trcut && lyr34);
    h34Xa->GetXaxis()->SetTitle("phi (rad)");
    h34Xa->Draw("colz");

    c41->cd(2);
    TH2D *h34Ya = new TH2D("h34Ya","Cluster Y residual vs phi INTT L34",200,-3.5,3.5,200,-20,20);
    ntp->Draw("residualY:clusphi>>h34Ya", trcut && lyr34);
    h34Ya->GetXaxis()->SetTitle("phi (rad)");
    h34Ya->Draw("colz");

    c41->cd(3);
    TH1D *h34X2a = new TH1D("h34X2a","Cluster X residual INTT L34",500,-0.5,0.5);
    ntp->Draw("residualX>>h34X2a",trcut && lyr34);
    h34X2a->GetXaxis()->SetTitle("X residual [mm]");
    h34X2a->Draw();

    //TF1 *fg3a = new TF1("fg3a","gaus",-0.0015,0.0015);
    fit_gaussian_if_possible(h34X2a, fg3a);

    c41->cd(4);
    TH1D *h34Y2a = new TH1D("h34Y2a","Cluster Y residual INTT L34",500, -20, 20);
    ntp->Draw("residualY>>h34Y2a",trcut && lyr34);
    h34Y2a->GetXaxis()->SetTitle("Y residual [mm]");
    h34Y2a->Draw();

    //TF1 *fg4a = new TF1("fg4a","gaus",-0.1,0.1);
    fit_gaussian_if_possible(h34Y2a, fg4a);
    add_to_combined_pdf(c41, "intt_plot_l34", save_individual_files);

    TCanvas *c42 = new TCanvas("c42","",50,300,1600,1600);
    c42->Divide(2,2);

    c42->cd(1);
    TH2D *h5Xa = new TH2D("h5Xa","Cluster X residual vs phi INTT L56",200,-3.5,3.5,200,-0.6,0.6);
    ntp->Draw("residualX:clusphi>>h5Xa", trcut && lyr56);
    h5Xa->GetXaxis()->SetTitle("phi (rad)");
    h5Xa->Draw("colz");

    c42->cd(2);
    TH2D *h5Ya = new TH2D("h5Ya","Cluster Y residual vs phi INTT L56",200,-3.5,3.5,200,-20,20);
    ntp->Draw("residualY:clusphi>>h5Ya", trcut && lyr56);
    h5Ya->GetXaxis()->SetTitle("phi (rad)");
    h5Ya->Draw("colz");

    c42->cd(3);
    TH1D *h5X2a = new TH1D("h5X2a","Cluster X residual INTT L56",500,-0.5,0.5);
    ntp->Draw("residualX>>h5X2a",trcut && lyr56);
    h5X2a->GetXaxis()->SetTitle("X residual [mm]");
    h5X2a->Draw();

    //TF1 *fg3a = new TF1("fg3a","gaus",-0.0015,0.0015);
    fit_gaussian_if_possible(h5X2a, fg3a);

    c42->cd(4);
    TH1D *h5Y2a = new TH1D("h5Y2a","Cluster Y residual INTT L56",500, -20, 20);
    ntp->Draw("residualY>>h5Y2a",trcut && lyr56);
    h5Y2a->GetXaxis()->SetTitle("Y residual [mm]");
    h5Y2a->Draw();

    //TF1 *fg4a = new TF1("fg4a","gaus",-0.1,0.1);
    fit_gaussian_if_possible(h5Y2a, fg4a);
   add_to_combined_pdf(c42, "intt_plot_l56", save_individual_files);

   if(do_plot_si_slice)
   {
      TCanvas *c301 = new TCanvas("c301","",50,300,1600,2400);
      c301->Divide(2,4);

      TF1 *fg301 = new TF1("fg301","gaus",-0.2,0.2); // INTT X residual fit range
      TF1 *fg301Y = new TF1("fg301Y","gaus",-10.0,10.0); // INTT Y residual fit range

      h34Xa->FitSlicesY(fg301, 0, -1, 1, "QNRG4");
      TH1D *h34Xa_1 = (TH1D*)gDirectory->Get("h34Xa_1");
      TH1D *h34Xa_2 = (TH1D*)gDirectory->Get("h34Xa_2");
      h5Xa->FitSlicesY(fg301, 0, -1, 1, "QNRG4");
      TH1D *h5Xa_1 = (TH1D*)gDirectory->Get("h5Xa_1");
      TH1D *h5Xa_2 = (TH1D*)gDirectory->Get("h5Xa_2");
      h34Ya->FitSlicesY(fg301Y, 0, -1, 1, "QNRG4");
      TH1D *h34Ya_1 = (TH1D*)gDirectory->Get("h34Ya_1");
      TH1D *h34Ya_2 = (TH1D*)gDirectory->Get("h34Ya_2");
      h5Ya->FitSlicesY(fg301Y, 0, -1, 1, "QNRG4");
      TH1D *h5Ya_1 = (TH1D*)gDirectory->Get("h5Ya_1");
      TH1D *h5Ya_2 = (TH1D*)gDirectory->Get("h5Ya_2");

      if(!h34Xa_1 || !h34Xa_2 || !h5Xa_1 || !h5Xa_2 ||
         !h34Ya_1 || !h34Ya_2 || !h5Ya_1 || !h5Ya_2)
      {
        std::cerr << "WARNING: one or more INTT FitSlicesY output histograms "
                  << "were not created; skipping the INTT slice canvas."
                  << std::endl;
      }
      else
      {
      h34Xa_1->SetTitle("X-residual Mean INTT Layers 3-4");
      h34Xa_2->SetTitle("X-residual Sigma INTT Layers 3-4");
      h5Xa_1->SetTitle("X-residual Mean INTT Layers 5-6");
      h5Xa_2->SetTitle("X-residual Sigma INTT Layers 5-6");
      h34Ya_1->SetTitle("Y-residual Mean INTT Layers 3-4");
      h34Ya_2->SetTitle("Y-residual Sigma INTT Layers 3-4");
      h5Ya_1->SetTitle("Y-residual Mean INTT Layers 5-6");
      h5Ya_2->SetTitle("Y-residual Sigma INTT Layers 5-6");

      c301->cd(1);
      h34Xa_1->GetYaxis()->SetRangeUser(-0.2,0.2);
      h34Xa_1->DrawCopy();
      draw_fit_range_box(fg301);

      c301->cd(2);
      h34Xa_2->GetYaxis()->SetRangeUser(0.0,0.3);
      h34Xa_2->DrawCopy();
      draw_fit_range_box(fg301);

      c301->cd(3);
      h5Xa_1->GetYaxis()->SetRangeUser(-0.5,0.5);
      h5Xa_1->DrawCopy();
      draw_fit_range_box(fg301);

      c301->cd(4);
      h5Xa_2->GetYaxis()->SetRangeUser(-0.000,0.5);
      h5Xa_2->DrawCopy();
      draw_fit_range_box(fg301);

      c301->cd(5);
      h34Ya_1->DrawCopy();
      draw_fit_range_box(fg301Y);

      c301->cd(6);
      h34Ya_2->DrawCopy();
      draw_fit_range_box(fg301Y);

      c301->cd(7);
      h5Ya_1->DrawCopy();
      draw_fit_range_box(fg301Y);

      c301->cd(8);
      h5Ya_2->DrawCopy();
      draw_fit_range_box(fg301Y);

      add_to_combined_pdf(c301, "intt_plot_Slice", save_individual_files);
      }
   }
  }

//  TCanvas *c6 = new TCanvas("c6","",50,300,1600,1600);
//  c6->Divide(2,2);
//
//  c6->cd(1);
//  TH2D *hetay = new TH2D("hetay","event vertex Y residual vs eta",200, -2, 2, 200, -vdY*2, vdY*2);
//  track_ntp->Draw("residual_vertexY:track_eta >> hetay",phicut);
//  hetay->GetXaxis()->SetTitle("vertex Y residual [mm]");
//  hetay->GetYaxis()->SetTitle("track eta");
//  hetay->DrawCopy("colz");
//
//  c6->cd(2);
//  TH1D *heta = new TH1D("heta","track eta",200,-20,20);
//  track_ntp->Draw("track_eta>> heta", phicut);
//  heta->GetXaxis()->SetTitle("track eta");
//  heta->DrawCopy();
//
//  TF1 *fg01 = new TF1("fg01","gaus",-0.05,0.05);
//  hetay->FitSlicesY(fg01, 0, -1, 1, "QNRG4");
//
//  c6->cd(3);
//  TH1D *hetay_1 = (TH1D*)gDirectory->Get("hetay_1");
//  hetay_1->GetXaxis()->SetTitle("eta");
//  hetay_1->GetYaxis()->SetTitle("mean of resid [cm]");
//  hetay_1->Draw();
//
//  c6->cd(4);
//  TH1D *hetay_2 = (TH1D*)gDirectory->Get("hetay_2");
//  hetay_2->GetXaxis()->SetTitle("eta");
//  hetay_2->GetYaxis()->SetTitle("sigma of resid [cm]");
//  hetay_2->Draw();
//
//  c6->SaveAs("vtx_track_eta.png");  
//  c6->SaveAs("vtx_track_eta.pdf");  

  /*
  TCanvas *c6 = new TCanvas("c6","vtx X local derivatives", 40,40,1200,1200);
  c6->Divide(2,2);
  c6->cd(1);
  TH2D *hdxdxys = new TH2D("hdxdxys","hdxdxys vs phi",200,-3.5,3.5,200,-20,20);
  track_ntp->Draw("dXdXYs:track_phi >>hdxdxys");
  hdxdxys->DrawCopy();

  c6->cd(2);
  TH2D *hdxdy0 = new TH2D("hdxdy0","hdxdy0 vs phi",200,-3.5,3.5,200,-20,20);
  track_ntp->Draw("dXdY0:track_phi >>hdxdy0");
  hdxdy0->DrawCopy();

  c6->cd(3);
  TH2D *hdxdzs = new TH2D("hdxdzs","hdxdZs vs phi",200,-3.5,3.5,200,-20,20);
  track_ntp->Draw("dXdZs:track_phi >>hdxdzs");
  hdxdzs->DrawCopy();

  c6->cd(4);
  TH2D *hdxdz0 = new TH2D("hdxdz0","hdxdZ0 vs phi",200,-3.5,3.5,200,-20,20);
  track_ntp->Draw("dXdZ0:track_phi >>hdxdz0");
  hdxdz0->DrawCopy();
  */

  /*
  TCanvas *c7 = new TCanvas("c7","vtx Y local derivatives", 40,40,1200,1200);
  c7->Divide(2,2);
  c7->cd(1);
  TH2D *hdydxys = new TH2D("hdydxys","hdydxys vs phi",200,-3.5,3.5,200,-20,20);
  track_ntp->Draw("dYdXYs:track_phi >>hdydxys");
  hdydxys->DrawCopy();

  c7->cd(2);
  TH2D *hdydy0 = new TH2D("hdydy0","hdydy0 vs phi",200,-3.5,3.5,200,-20,20);
  track_ntp->Draw("dYdY0:track_phi >>hdydy0");
  hdydy0->DrawCopy();

  c7->cd(3);
  TH2D *hdydzs = new TH2D("hdydzs","hdydZs vs phi",200,-3.5,3.5,200,-20,20);
  track_ntp->Draw("dYdZs:track_phi >>hdydzs");
  hdydzs->DrawCopy();

  c7->cd(4);
  TH2D *hdydz0 = new TH2D("hdydz0","hdydZ0 vs phi",200,-3.5,3.5,200,-20,20);
  track_ntp->Draw("dYdZ0:track_phi >>hdydz0");
  hdydz0->DrawCopy();
  */
  /*
  TCanvas *c8 = new TCanvas("c8","vtx global derivatives", 40,40,1200,800);
  c8->Divide(3,2);

  c8->cd(1);
  TH2D *hdxdx = new TH2D("hdxdx","hdxdx vs phi",200,-3.5,3.5,200,-20,20);
  track_ntp->Draw("dXdx:track_phi >>hdxdx");
  hdxdx->DrawCopy();

  c8->cd(2);
  TH2D *hdxdy = new TH2D("hdxdy","hdxdy vs phi",200,-3.5,3.5,200,-20,20);
  track_ntp->Draw("dXdy:track_phi >>hdxdy");
  hdxdy->DrawCopy();

  c8->cd(3);
  TH2D *hdxdz = new TH2D("hdxdz","hdxdz vs phi",200,-3.5,3.5,200,-20,20);
  track_ntp->Draw("dXdz:track_phi >>hdxdz");
  hdxdz->DrawCopy();

  c8->cd(4);
  TH2D *hdydx = new TH2D("hdydx","hdydx vs phi",200,-3.5,3.5,200,-20,20);
  track_ntp->Draw("dYdx:track_phi >>hdydx");
  hdydx->DrawCopy();

  c8->cd(5);
  TH2D *hdydy = new TH2D("hdydy","hdydy vs phi",200,-3.5,3.5,200,-20,20);
  track_ntp->Draw("dYdy:track_phi >>hdydy");
  hdydy->DrawCopy();

  c8->cd(6);
  TH2D *hdydz = new TH2D("hdydz","hdydz vs phi",200,-3.5,3.5,200,-20,20);
  track_ntp->Draw("dYdz:track_phi >>hdydz");
  hdydz->DrawCopy();
  */
  /*
  TCanvas *c9 = new TCanvas("c9","",50,50,800,800);
  
  track_ntp->Draw("track_phi","abs(track_xvtx) < 0.1 && abs(track_yvtx) < 0.1");
  */

  write_combined_pdf();
}
