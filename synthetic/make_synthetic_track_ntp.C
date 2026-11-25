#include <TFile.h>
#include <TNtuple.h>
#include <TRandom3.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Closure test of macros/solveDcaZAlignment.C. make_synthetic_track_ntp writes
// a ROOT file with the ntp/track_ntp columns the solver reads (MakeMilleFiles
// format, one ntp row per cluster) for straight s-z tracks from vertices
// through the three MVTX layers, with per-chip z offsets injected, and a CSV
// with the dz the solver should return. checkDzClosure compares a solution
// with it.
//
// Model (the solver's, so pulls should be ~1):
//   - per event 1-2 vertices (x, y ~ N(0, 0.1 mm), z ~ N(0, 80 mm)) with
//     2-12 tracks each; |eta| < 1, uniform phi, pT = 0.3 + Exp(0.6) GeV (the
//     tracks below the solver's pT cut test that cut);
//   - flat staves at rMid with phi0 of PHG4MvtxDefs.h, 9 chips of 30 mm;
//   - scattering kinks at the beam pipe (r = 20 mm), L0 and L1; 5 um hit
//     smearing; one INTT row per track (must be ignored);
//   - "reco" vertex z = mean z0 of straight-line fits of its tracks, and
//     residual_vertexY = vertex z - track z0; event_x/yvtx smeared by 30 um;
//     n_tracks_vtx = tracks in that mean, track_in_vtx = 1 for them;
//   - nJobs blocks of events numbered from 1 each (an hadd-merged file).
// Injection: a chip really at geometry + m_c gives zglob = z_true - m_c and
// the solver should return dz = +m_c. Modes:
//   none       m = 0;
//   random     m ~ N(0, sizeMm) per chip;
//   single     L0 stave 3 chip 4 at +sizeMm (sign check);
//   clamshell  +sizeMm for staves with cos(phi) > 0, -sizeMm otherwise (a
//              phi-dependent z shift that hits alone cannot see);
//   weakmodes  0.02 mm + 0.5e-3 * r + 1e-4 * z (patterns no track sees;
//              the solver should return ~0).
// Except for weakmodes the injection is projected to sum(m) = 0 per layer and
// sum(z_chip * m) = 0, the constraints of constrainLayerMeans, so that the
// expected solution is m itself; the truth CSV holds the projected values.
//
//   root -l -b -q 'make_synthetic_track_ntp.C+("random", 0.03)'
//   ls synth_tracks_random.root > list_synth.txt
//   root -l -b -q '../macros/solveDcaZAlignment.C+("list_synth.txt:synth_random")'
//   root -l -b -q 'make_synthetic_track_ntp.C+' \
//     -e 'checkDzClosure("dz_solution_synth_random.csv", "synth_tracks_random_truth.csv")'
// Outputs are test data; do not commit them.
// Runs locally only; nothing is sent to any external service.

namespace
{
  constexpr int kMvtxLayers = 3;
  constexpr int kChipsPerStave = 9;
  constexpr int kStavesPerLayer[kMvtxLayers] = {12, 16, 20};
  constexpr int kFirstChipOfLayer[kMvtxLayers] = {0, 108, 252};
  constexpr int kMvtxChips = 432;
  // rMid and phi0 of PHG4MvtxDefs.h (mm, rad).
  constexpr double kMvtxRadiusMm[kMvtxLayers] = {25.23, 33.36, 41.48};
  constexpr double kMvtxPhi0[kMvtxLayers] = {0.2382, 0.1937, 0.1481};
  constexpr double kChipLengthMm = 30.0;
  constexpr double kHitSigmaMm = 0.005;
  // The solver's material constants.
  constexpr double kBeamPipeRadiusMm = 20.0;
  constexpr double kBeamPipeX0 = 0.0022;
  constexpr double kMvtxLayerX0 = 0.003;
  constexpr double kPionMassGeV = 0.13957;
  constexpr double kVertexXYSpreadMm = 0.1;
  constexpr double kVertexZSpreadMm = 80.0;
  constexpr double kVertexXYErrorMm = 0.03;
  constexpr double kInttRadiusMm = 75.0;

  int chip_index(int layer, int stave, int chip)
  {
    return kFirstChipOfLayer[layer] + kChipsPerStave * stave + chip;
  }

  double stave_phi(int layer, int stave)
  {
    return kMvtxPhi0[layer] + 2.0 * M_PI * stave / kStavesPerLayer[layer];
  }

  double chip_center_z(int chip)
  {
    return (chip - (kChipsPerStave - 1) / 2.0) * kChipLengthMm;
  }

  double scattering_angle(double momentumGeV, double radiationLengths)
  {
    const double beta = momentumGeV /
                        std::sqrt(momentumGeV * momentumGeV + kPionMassGeV * kPionMassGeV);
    return 0.0136 / (beta * momentumGeV) * std::sqrt(radiationLengths) *
           (1.0 + 0.038 * std::log(radiationLengths));
  }

  // Removes the per-layer mean and the component along the chip z.
  void project_to_constraints(std::vector<double>& m)
  {
    for(int layer = 0; layer < kMvtxLayers; ++layer)
    {
      const int first = kFirstChipOfLayer[layer];
      const int n = kChipsPerStave * kStavesPerLayer[layer];
      double mean = 0;
      for(int i = first; i < first + n; ++i)
      {
        mean += m[i] / n;
      }
      for(int i = first; i < first + n; ++i)
      {
        m[i] -= mean;
      }
    }
    double dot = 0;
    double norm = 0;
    for(int i = 0; i < kMvtxChips; ++i)
    {
      const double z = chip_center_z(i % kChipsPerStave);
      dot += m[i] * z;
      norm += z * z;
    }
    for(int i = 0; i < kMvtxChips; ++i)
    {
      m[i] -= dot / norm * chip_center_z(i % kChipsPerStave);
    }
  }

  bool make_injection(const std::string& mode, double sizeMm, TRandom3& random,
                      std::vector<double>& injected, std::vector<double>& expected)
  {
    injected.assign(kMvtxChips, 0.0);
    for(int layer = 0; layer < kMvtxLayers; ++layer)
    {
      for(int stave = 0; stave < kStavesPerLayer[layer]; ++stave)
      {
        for(int chip = 0; chip < kChipsPerStave; ++chip)
        {
          double& m = injected[chip_index(layer, stave, chip)];
          if(mode == "random")
          {
            m = random.Gaus(0.0, sizeMm);
          }
          else if(mode == "single")
          {
            m = (layer == 0 && stave == 3 && chip == 4) ? sizeMm : 0.0;
          }
          else if(mode == "clamshell")
          {
            m = std::cos(stave_phi(layer, stave)) > 0 ? sizeMm : -sizeMm;
          }
          else if(mode == "weakmodes")
          {
            m = 0.02 + 0.5e-3 * kMvtxRadiusMm[layer] + 1e-4 * chip_center_z(chip);
          }
          else if(mode != "none")
          {
            std::cerr << "ERROR: unknown mode '" << mode
                      << "' (none, random, single, clamshell, weakmodes)" << std::endl;
            return false;
          }
        }
      }
    }
    if(mode != "weakmodes")
    {
      project_to_constraints(injected);
    }
    expected = injected;
    project_to_constraints(expected);
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
    return fields;
  }

  // CSV -> rows of named columns.
  bool read_csv(const char* path, std::vector<std::map<std::string, std::string>>& rows)
  {
    std::ifstream input(path);
    std::string text;
    if(!input.is_open() || !std::getline(input, text))
    {
      std::cerr << "ERROR: could not read " << path << std::endl;
      return false;
    }
    const std::vector<std::string> header = split_csv(text);
    while(std::getline(input, text))
    {
      const std::vector<std::string> fields = split_csv(text);
      if(fields.size() != header.size())
      {
        continue;
      }
      std::map<std::string, std::string> row;
      for(size_t i = 0; i < header.size(); ++i)
      {
        row[header[i]] = fields[i];
      }
      rows.push_back(row);
    }
    return true;
  }
}

// Writes <outputDirectory>/synth_tracks_<mode>.root and
// synth_tracks_<mode>_truth.csv (layer,stave,chip,dz_expected_mm).
void make_synthetic_track_ntp(
  const char* mode = "random",
  double sizeMm = 0.03,
  int eventsPerJob = 20000,
  int nJobs = 2,
  unsigned int seed = 12345,
  const char* outputDirectory = ".")
{
  TRandom3 random(seed);
  std::vector<double> injected;
  std::vector<double> expected;
  if(!make_injection(mode, sizeMm, random, injected, expected))
  {
    return;
  }

  const std::string base = std::string(outputDirectory) + "/synth_tracks_" + mode;
  TFile file((base + ".root").c_str(), "RECREATE");
  TNtuple ntp("ntp", "synthetic MakeMilleFiles clusters",
              "layer:stave:chip:xglob:yglob:zglob:errY:event:track_id:"
              "track_crossing:vertex_id:residual_vertexY:track_eta:track_pt");
  TNtuple trackNtp("track_ntp", "synthetic MakeMilleFiles tracks",
                   "event:track_id:event_xvtx:event_yvtx:event_zvtx:n_tracks_vtx:track_in_vtx");

  struct Hit
  {
    int layer;
    int stave;
    int chip;
    double x;
    double y;
    double z;
    double s;
  };
  struct Track
  {
    double eta;
    double pt;
    std::vector<Hit> hits;
    double z0 = std::numeric_limits<double>::quiet_NaN();
  };

  long tracksWritten = 0;
  for(int job = 0; job < nJobs; ++job)
  {
    for(int event = 1; event <= eventsPerJob; ++event)
    {
      int trackId = 0;
      const int nVertices = 1 + static_cast<int>(random.Integer(2));
      for(int vertexId = 0; vertexId < nVertices; ++vertexId)
      {
        const double vx = random.Gaus(0, kVertexXYSpreadMm);
        const double vy = random.Gaus(0, kVertexXYSpreadMm);
        const double vz = random.Gaus(0, kVertexZSpreadMm);
        const int nTracks = 2 + static_cast<int>(random.Integer(11));
        std::vector<Track> tracks(nTracks);
        for(Track& track : tracks)
        {
          track.eta = random.Uniform(-1, 1);
          track.pt = 0.3 + random.Exp(0.6);
          const double phi = random.Uniform(-M_PI, M_PI);
          const double sinTheta = 1.0 / std::cosh(track.eta);
          const double toCot = 1.0 / (sinTheta * sinTheta);
          const double momentum = track.pt * std::cosh(track.eta);
          const double cot = std::sinh(track.eta);
          const double kinkBeamPipe =
            random.Gaus(0, scattering_angle(momentum, kBeamPipeX0 / sinTheta) * toCot);
          const double kinkLayer =
            scattering_angle(momentum, kMvtxLayerX0 / sinTheta) * toCot;
          double kinks[2] = {random.Gaus(0, kinkLayer), random.Gaus(0, kinkLayer)};
          double kinkRadius[2] = {0, 0};
          for(int layer = 0; layer < kMvtxLayers; ++layer)
          {
            // Nearest stave in phi; hit on its flat plane at rMid.
            const double step = 2.0 * M_PI / kStavesPerLayer[layer];
            int stave = static_cast<int>(std::lround((phi - kMvtxPhi0[layer]) / step));
            stave = ((stave % kStavesPerLayer[layer]) + kStavesPerLayer[layer]) %
                    kStavesPerLayer[layer];
            const double staveAngle = stave_phi(layer, stave);
            const double nx = std::cos(staveAngle);
            const double ny = std::sin(staveAngle);
            const double along = std::cos(phi) * nx + std::sin(phi) * ny;
            if(along < 0.5)
            {
              continue;
            }
            const double s = (kMvtxRadiusMm[layer] - (vx * nx + vy * ny)) / along;
            double z = vz + cot * s + kinkBeamPipe * (s - kBeamPipeRadiusMm);
            for(int j = 0; j < layer && j < 2; ++j)
            {
              z += kinks[j] * (s - kinkRadius[j]);
            }
            if(layer < 2)
            {
              kinkRadius[layer] = s;
            }
            const int chip = static_cast<int>(
              std::floor(z / kChipLengthMm + kChipsPerStave / 2.0));
            if(chip < 0 || chip >= kChipsPerStave)
            {
              continue;
            }
            const double measured = z - injected[chip_index(layer, stave, chip)] +
                                    random.Gaus(0, kHitSigmaMm);
            track.hits.push_back(Hit{layer, stave, chip, vx + s * std::cos(phi),
                                     vy + s * std::sin(phi), measured, s});
          }
          // "Reco" z0: unweighted straight-line fit in (s, z).
          if(track.hits.size() >= 2)
          {
            double sumS = 0;
            double sumZ = 0;
            double sumSS = 0;
            double sumSZ = 0;
            const double n = static_cast<double>(track.hits.size());
            for(const Hit& hit : track.hits)
            {
              sumS += hit.s;
              sumZ += hit.z;
              sumSS += hit.s * hit.s;
              sumSZ += hit.s * hit.z;
            }
            const double slope = (n * sumSZ - sumS * sumZ) / (n * sumSS - sumS * sumS);
            track.z0 = (sumZ - slope * sumS) / n;
          }
        }
        double vertexZ = 0;
        int fitted = 0;
        for(const Track& track : tracks)
        {
          if(std::isfinite(track.z0))
          {
            vertexZ += track.z0;
            ++fitted;
          }
        }
        vertexZ = fitted > 0 ? vertexZ / fitted : vz;
        const double vxReco = vx + random.Gaus(0, kVertexXYErrorMm);
        const double vyReco = vy + random.Gaus(0, kVertexXYErrorMm);

        for(const Track& track : tracks)
        {
          const float minusDcaZ = static_cast<float>(vertexZ - track.z0);
          for(const Hit& hit : track.hits)
          {
            ntp.Fill(hit.layer, hit.stave, hit.chip, hit.x / 10.0, hit.y / 10.0,
                     hit.z / 10.0, kHitSigmaMm, event, trackId, 0, vertexId,
                     minusDcaZ, track.eta, track.pt);
          }
          // One INTT row (must be ignored by the solver).
          ntp.Fill(3, 0, 0, kInttRadiusMm / 10.0, 0, 0, 1.0, event, trackId, 0,
                   vertexId, minusDcaZ, track.eta, track.pt);
          const float trackInVertex = std::isfinite(track.z0) ? 1.0f : 0.0f;
          trackNtp.Fill(event, trackId, vxReco, vyReco, vertexZ, fitted, trackInVertex);
          ++trackId;
          ++tracksWritten;
        }
      }
    }
  }
  file.Write();
  file.Close();

  std::ofstream truth(base + "_truth.csv");
  truth << std::setprecision(10) << "layer,stave,chip,dz_expected_mm,injected_mm\n";
  for(int layer = 0; layer < kMvtxLayers; ++layer)
  {
    for(int stave = 0; stave < kStavesPerLayer[layer]; ++stave)
    {
      for(int chip = 0; chip < kChipsPerStave; ++chip)
      {
        const int index = chip_index(layer, stave, chip);
        truth << layer << ',' << stave << ',' << chip << ',' << expected[index] << ','
              << injected[index] << '\n';
      }
    }
  }
  std::cout << "Wrote " << base << ".root (" << tracksWritten << " tracks, " << nJobs
            << " jobs of " << eventsPerJob << " events) and " << base
            << "_truth.csv" << std::endl;
}

// Pulls (dz - dz_expected) / dz_err of the free chips, per layer and overall.
void checkDzClosure(const char* solutionCsv, const char* truthCsv)
{
  std::vector<std::map<std::string, std::string>> solutionRows;
  std::vector<std::map<std::string, std::string>> truthRows;
  if(!read_csv(solutionCsv, solutionRows) || !read_csv(truthCsv, truthRows))
  {
    return;
  }
  std::map<std::string, double> expected;
  for(const auto& row : truthRows)
  {
    expected[row.at("layer") + "," + row.at("stave") + "," + row.at("chip")] =
      std::stod(row.at("dz_expected_mm"));
  }

  double sum[kMvtxLayers + 1] = {};
  double sum2[kMvtxLayers + 1] = {};
  double diff2[kMvtxLayers + 1] = {};
  double truth2[kMvtxLayers + 1] = {};
  double product[kMvtxLayers + 1] = {};
  int count[kMvtxLayers + 1] = {};
  for(const auto& row : solutionRows)
  {
    if(row.at("free") != "1")
    {
      continue;
    }
    const int layer = std::stoi(row.at("layer"));
    const auto truth = expected.find(row.at("layer") + "," + row.at("stave") + "," +
                                     row.at("chip"));
    const double error = std::stod(row.at("dz_err_mm"));
    if(truth == expected.end() || !(error > 0) || layer < 0 || layer >= kMvtxLayers)
    {
      continue;
    }
    const double dz = std::stod(row.at("dz_mm"));
    const double difference = dz - truth->second;
    const double pull = difference / error;
    for(const int bin : {layer, kMvtxLayers})
    {
      sum[bin] += pull;
      sum2[bin] += pull * pull;
      diff2[bin] += difference * difference;
      truth2[bin] += truth->second * truth->second;
      product[bin] += dz * truth->second;
      ++count[bin];
    }
  }

  std::cout << std::fixed << std::setprecision(3)
            << "Closure of " << solutionCsv << " against " << truthCsv << ":\n";
  bool pass = count[kMvtxLayers] > 0;
  for(int bin = 0; bin <= kMvtxLayers; ++bin)
  {
    if(count[bin] == 0)
    {
      continue;
    }
    const double mean = sum[bin] / count[bin];
    const double rms = std::sqrt(sum2[bin] / count[bin] - mean * mean);
    std::cout << "  " << (bin < kMvtxLayers ? "L" + std::to_string(bin) : std::string("all"))
              << ": " << count[bin] << " chips, pull mean " << mean << " RMS " << rms
              << ", RMS(dz - expected) " << std::setprecision(2)
              << 1e3 * std::sqrt(diff2[bin] / count[bin]) << " um, RMS(expected) "
              << 1e3 * std::sqrt(truth2[bin] / count[bin]) << " um"
              << std::setprecision(3);
    if(truth2[bin] > 0)
    {
      std::cout << ", slope dz/expected " << product[bin] / truth2[bin];
    }
    std::cout << '\n';
    if(bin == kMvtxLayers)
    {
      pass = std::fabs(mean) < 0.2 && rms > 0.8 && rms < 1.25;
    }
  }
  std::cout << (pass ? "PASS" : "FAIL")
            << " (overall pull mean within +-0.2 and RMS within 0.8-1.25)"
            << std::defaultfloat << std::endl;
}
