#include <TDecompBK.h>
#include <TFile.h>
#include <TMatrixDSym.h>
#include <TMatrixDSymEigen.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TVectorD.h>

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
#include <tuple>
#include <vector>

// Solves for one global z shift dz per MVTX chip that brings the DCA_z of the
// tracks to 0 while keeping their MVTX hits on straight lines, using every
// track of every vertex at once (a z-only Millepede with the vertex as a
// shared point). Shifting each chip by the mean DCA_z of its own tracks fails
// because a track's z at the vertex depends on all three MVTX chips it crosses
// with lever arms of opposite sign (about +2.4, +0.3 and -1.7 for L0-L2).
//
// Model, in global coordinates and mm, for MVTX hit k of track t of vertex v:
//   zglob_k = z_v + a_t + cot_t * s_k - dz(chip of k)
//   - s_k: transverse distance of the cluster from the event vertex (x, y);
//     z is linear in the transverse arc length, the chord differs by < 2 um.
//   - z_v: with useRecoVertex (default) the reconstructed vertex z of
//     track_ntp (event_zvtx, from all tracks of the vertex finder), a fixed
//     point, so DCA_z has the definition of residual_vertexY and single-track
//     vertices are used. Otherwise z_v is refitted from the selected tracks of
//     the vertex; with only 2-3 tracks in the ntuple that vertex absorbs a
//     large part of every track's DCA (half of it for 2 tracks).
//   - a_t: DCA_z of the track, with the prior a_t = 0 +- sigma_p,
//     sigma_p^2 = cot^2 sigma_xy^2 + sigma_bp^2 [+ sigma_vz^2] + (1 um)^2
//     (sigma_xy: error of the vertex x, y; sigma_bp: beam-pipe scattering,
//     which a free cot_t turns into an offset at the vertex; sigma_vz:
//     vertexZSigmaMm, the z error of the reconstructed vertex).
//   - Hit covariance: errY^2 + hitFloor^2, plus the scattering kinks at L0 and
//     L1 propagated to the outer hits (correlated, per track).
//   - a_t, cot_t (per track) and a refitted z_v are profiled out; the
//     remaining normal equations give dz of all chips at once.
//   The vertex finder refits the vertex from the shifted tracks, so a change
//   of a track's z0 changes its DCA_z only by f = 1 - track_in_vtx /
//   n_tracks_vtx (track_ntp). With vertexFollows (default) the model uses
//   DCA_z(after) = a0 + f (a - a0) (a0: the current hit-only a), which
//   turns the prior into a = -(1 - f) / f * a0 +- sigma_p / f; tracks with
//   f < 0.05 are dropped. Without it the shifts undershoot by about 1/N.
//   scatteringScale multiplies all scattering angles (beam pipe, L0, L1): a
//   larger value weights the inner hit more for z at the vertex, as a Kalman
//   fit with more material does, and lowers the lever arm of L2.
//
// Prediction only (predictDzCsv): no solve; the per-chip change of <-DCA_z>
//   that the dz of predictDzCsv (dz_mm of a _dz_summary.csv, i.e. the
//   applied dz, or of a dz_solution CSV) should cause, written to
//   dz_prediction_<label>.csv. With dcaFitsCsv (run with the base
//   alignment) and measuredAfterCsv (run with the shifted alignment) it
//   prints r = measured / predicted change per layer: choose hitFloorMm,
//   scatteringScale and vertexFollows so that r ~ 1 before the next solve.
//   Run it on the ntuples of the base alignment.
//   A chip that is really at geometry + m gives zglob = z_true - m, so the
//   solution dz = +m is the value to ADD to column 7 (global z translation,
//   mm) of the alignment file the ntuples were made with
//   (AlignmentTransformation applies it as a global translation).
// Patterns of dz that do not change any track (exactly or nearly): a common z
// shift, a shift proportional to the radius (shear) and one proportional to z
// (scale). With constrainLayerMeans they are fixed by sum(dz) = 0 per layer
// and sum(zbar_chip * dz) = 0 (zbar: mean z of the chip's hits), which keeps
// the base geometry wherever DCA_z cannot see; otherwise by sum(dz) = 0,
// sum(sbar * dz) = 0 and sum(zbar * dz) = 0. Chips with fewer than
// minTracksPerChip tracks are kept fixed (dz = 0, their hits still used).
// INTT clusters are not used (mm z resolution, no z information here).
//
// Inputs: one "list.txt:label" of MakeMilleFiles ntuples, read file by file.
//   ntp:       layer, stave, chip, xglob, yglob, zglob (cm), errY (mm),
//              event, track_id, track_crossing, vertex_id,
//              residual_vertexY (-DCA_z, mm), track_eta, track_pt
//   track_ntp: event, track_id, event_xvtx, event_yvtx, event_zvtx (mm)
//   `event` counts per job, so a decreasing event (or a track id repeated
//   within an event) starts a new segment of an hadd-merged file.
// Selection: known crossing, pT >= minimumPtGeV, finite |-DCA_z| <=
//   maxAbsMinusDcaZMm, >= 1 MVTX hit in each layer (<= 8 hits), a track_ntp
//   row, >= 2 such tracks in the vertex if it is refitted; eventParity 0/1 keeps even/odd
//   events only (split-sample checks). After a first solution, tracks with a
//   prior pull > 5 or a hit chi2 > 30 are dropped and the solve is repeated.
//
// Output: dz_solution_<label>.csv in csvOutputDirectory, one row per chip:
//   label,layer,stave,chip,stavechip,n_tracks,free,dz_mm,dz_err_mm,
//   mu_before_ntp_mm,mu_before_model_mm,mu_pred_after_mm,mu_err_mm
//   mu = <-DCA_z> of the chip's tracks (iterated +-2 sigma truncated mean, as
//   the core fit of plot_fits_overlaid_dcadistsfromstavechips.C): from the
//   ntuple (_ntp), from this model before the shift (_model) and predicted
//   after it (_pred_after). dz_err is scaled by sqrt(chi2/ndf) when > 1.
//   Apply it with applyDzSolution of makeDcaZShiftedAlignment.C.
// Check before applying (printed): mu_before_ntp agrees with the dca_fits mu
//   (dcaFitsCsv, same pT cut); mu_before_model follows mu_before_ntp (slope
//   ~ 1, RMS difference well below the chip-to-chip spread); chi2/ndf ~ 1;
//   only the expected near-zero eigenvalues; even/odd solutions agree.
//
//   root -l -b -q 'solveDcaZAlignment.C+("filelist1.txt:Iter1","dca_fits_Iter1.csv")'
// Runs locally only; nothing is sent to any external service.

namespace
{
  constexpr int kMvtxLayers = 3;
  constexpr int kChipsPerStave = 9;
  constexpr int kStavesPerLayer[kMvtxLayers] = {12, 16, 20};
  constexpr int kFirstChipOfLayer[kMvtxLayers] = {0, 108, 252};
  constexpr int kMvtxChips = 432;
  // Stave mid radii (rMid of PHG4MvtxDefs.h), used if a track has no hit to
  // place a scattering layer.
  constexpr double kMvtxRadiusMm[kMvtxLayers] = {25.23, 33.36, 41.48};
  // Material for the scattering terms (radiation lengths at normal
  // incidence): beryllium beam pipe and one MVTX layer.
  constexpr double kBeamPipeRadiusMm = 20.0;
  constexpr double kBeamPipeX0 = 0.0022;
  constexpr double kMvtxLayerX0 = 0.003;
  constexpr double kPionMassGeV = 0.13957;
  // SvtxTrack crossing of tracks whose bunch crossing is unknown (SHRT_MAX).
  constexpr double kUnknownCrossing = std::numeric_limits<short>::max();
  constexpr double kCmToMm = 10.0;
  constexpr int kMaxHitsPerTrack = 8;
  constexpr int kMinTracksPerVertex = 2;
  constexpr double kPriorFloorMm = 0.001;
  // Outlier cuts of the second pass.
  constexpr double kMaxPriorPull = 5.0;
  constexpr double kMaxTrackHitChi2 = 30.0;
  // Truncated mean of <-DCA_z> per chip: seed window +-0.1 mm around 0 (the
  // seed fit range of the dca_fits), then iterations of +-2 sigma.
  constexpr double kTruncatedMeanSeedMm = 0.1;
  constexpr double kTruncatedMeanWindow = 2.0;
  constexpr int kTruncatedMeanIterations = 4;
  constexpr int kMinimumValuesForMean = 10;
  constexpr int kEigenvaluesReported = 6;

  struct HitRecord
  {
    short chip = -1;
    short layer = -1;
    // Transverse distance from the event vertex and zglob relative to the
    // vertex reference line (mm); see read_sample.
    float s = 0;
    float y = 0;
    float sigma = 0;
  };

  struct TrackRecord
  {
    int firstHit = 0;
    int nHits = 0;
    float pt = 0;
    float eta = 0;
    float minusDcaZNtp = 0;
    // Fraction of a change of the track's z0 that remains in its DCA_z once
    // the vertex finder refits the vertex: 1 - track_in_vtx / n_tracks_vtx
    // with vertexFollows, else 1.
    float vertexFactor = 1;
    bool used = true;
  };

  struct VertexRecord
  {
    int firstTrack = 0;
    int nTracks = 0;
  };

  struct ReadCounts
  {
    long files = 0;
    long events = 0;
    long tracks = 0;
    long unknownCrossing = 0;
    long failedPt = 0;
    long failedDca = 0;
    long missingLayer = 0;
    long tooManyHits = 0;
    long noVertexRow = 0;
    long ownVertex = 0;
    long smallVertex = 0;
    long tracksKept = 0;
    long verticesKept = 0;
  };

  struct Sample
  {
    std::vector<HitRecord> hits;
    std::vector<TrackRecord> tracks;
    std::vector<VertexRecord> vertices;
    ReadCounts counts;
    // Per chip: tracks, and sums of the hit z (mm, absolute) and s.
    std::vector<int> tracksPerChip = std::vector<int>(kMvtxChips, 0);
    std::vector<double> sumZ = std::vector<double>(kMvtxChips, 0.0);
    std::vector<double> sumS = std::vector<double>(kMvtxChips, 0.0);
    std::vector<int> hitsPerChip = std::vector<int>(kMvtxChips, 0);
  };

  struct Settings
  {
    double minimumPtGeV = 0.65;
    double hitFloorMm = 0.005;
    double vertexXYSigmaMm = 0.03;
    bool includeScattering = true;
    double maxAbsMinusDcaZMm = 0.5;
    int eventParity = -1;
    Long64_t maxEvents = -1;
    bool useRecoVertex = true;
    double vertexZSigmaMm = 0.02;
    bool vertexFollows = true;
    double scatteringScale = 1.0;
  };

  // Below this vertexFactor the track is (almost) its own vertex and its
  // DCA_z cannot be moved by the alignment.
  constexpr double kMinVertexFactor = 0.05;

  // Tracks a vertex needs: with the reconstructed vertex as a fixed point
  // every track stands alone; with a refitted vertex one track carries no
  // DCA information.
  int min_tracks_per_vertex(const Settings& settings)
  {
    return settings.useRecoVertex ? 1 : kMinTracksPerVertex;
  }

  struct TruncatedMean
  {
    double mean = std::numeric_limits<double>::quiet_NaN();
    double error = std::numeric_limits<double>::quiet_NaN();
    int n = 0;
  };

  // ---------------------------------------------------------------------------
  // Chips and inputs
  // ---------------------------------------------------------------------------

  int chip_index(int layer, int stave, int chip)
  {
    if(layer < 0 || layer >= kMvtxLayers || stave < 0 ||
       stave >= kStavesPerLayer[layer] || chip < 0 || chip >= kChipsPerStave)
    {
      return -1;
    }
    return kFirstChipOfLayer[layer] + kChipsPerStave * stave + chip;
  }

  void chip_of_index(int index, int& layer, int& stave, int& chip)
  {
    layer = index < kFirstChipOfLayer[1] ? 0 : index < kFirstChipOfLayer[2] ? 1 : 2;
    const int inLayer = index - kFirstChipOfLayer[layer];
    stave = inLayer / kChipsPerStave;
    chip = inLayer % kChipsPerStave;
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

  // "list.txt:label"; a missing label defaults to the list filename without
  // directory and extension.
  bool parse_input_spec(const char* spec, std::string& listfile, std::string& label)
  {
    const std::string text = trim(spec ? spec : "");
    if(text.empty() || text.find(',') != std::string::npos)
    {
      std::cerr << "ERROR: give exactly one \"list.txt:label\" (no commas), not '"
                << text << "'" << std::endl;
      return false;
    }
    const std::string::size_type colon = text.find(':');
    listfile = trim(text.substr(0, colon));
    label = colon == std::string::npos ? "" : trim(text.substr(colon + 1));
    if(label.empty())
    {
      label = listfile.substr(listfile.find_last_of('/') + 1);
      label = label.substr(0, label.find_last_of('.'));
    }
    return !listfile.empty();
  }

  bool read_file_list(const std::string& listfile, std::vector<std::string>& files)
  {
    std::ifstream input(listfile);
    if(!input.is_open())
    {
      std::cerr << "ERROR: could not open " << listfile << std::endl;
      return false;
    }
    std::string filename;
    while(input >> filename)
    {
      files.push_back(filename);
    }
    if(files.empty())
    {
      std::cerr << "ERROR: no files listed in " << listfile << std::endl;
      return false;
    }
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

  bool has_branches(TTree* tree, const std::vector<std::string>& names)
  {
    for(const std::string& name : names)
    {
      if(!tree->GetBranch(name.c_str()))
      {
        std::cerr << "ERROR: branch '" << name << "' missing in tree "
                  << tree->GetName() << std::endl;
        return false;
      }
    }
    return true;
  }

  // (segment, event, track_id)
  using TrackKey = std::tuple<int, long, long>;

  struct VertexPosition
  {
    float x = 0;
    float y = 0;
    float z = 0;
    // Tracks of the vertex and whether this track is one of them (0 / 1);
    // read only for the vertexFollows factor.
    float nTracks = 0;
    float inVertex = 0;
  };

  // track_ntp rows (one per track, filled in the same order as ntp) ->
  // event vertex position (mm), and with needVertexTracks n_tracks_vtx and
  // track_in_vtx.
  bool read_vertex_positions(
    TFile& file,
    bool needVertexTracks,
    std::map<TrackKey, VertexPosition>& positions)
  {
    TTree* tree = dynamic_cast<TTree*>(file.Get("track_ntp"));
    if(!tree || !has_branches(tree, {"event", "track_id", "event_xvtx",
                                     "event_yvtx", "event_zvtx"}))
    {
      std::cerr << "ERROR: no usable track_ntp in " << file.GetName() << std::endl;
      return false;
    }
    if(needVertexTracks && !has_branches(tree, {"n_tracks_vtx", "track_in_vtx"}))
    {
      std::cerr << "ERROR: vertexFollows needs n_tracks_vtx and track_in_vtx in "
                   "track_ntp; rerun with vertexFollows = false." << std::endl;
      return false;
    }
    TTreeReader reader(tree);
    TTreeReaderValue<Float_t> event(reader, "event");
    TTreeReaderValue<Float_t> trackId(reader, "track_id");
    TTreeReaderValue<Float_t> x(reader, "event_xvtx");
    TTreeReaderValue<Float_t> y(reader, "event_yvtx");
    TTreeReaderValue<Float_t> z(reader, "event_zvtx");
    std::unique_ptr<TTreeReaderValue<Float_t>> nTracks;
    std::unique_ptr<TTreeReaderValue<Float_t>> inVertex;
    if(needVertexTracks)
    {
      nTracks = std::make_unique<TTreeReaderValue<Float_t>>(reader, "n_tracks_vtx");
      inVertex = std::make_unique<TTreeReaderValue<Float_t>>(reader, "track_in_vtx");
    }

    int segment = 0;
    long previousEvent = -1;
    std::set<long> idsInEvent;
    while(reader.Next())
    {
      const long eventNumber = std::lround(*event);
      const long id = std::lround(*trackId);
      if(eventNumber < previousEvent ||
         (eventNumber == previousEvent && idsInEvent.count(id)))
      {
        ++segment;
      }
      if(eventNumber != previousEvent || idsInEvent.count(id))
      {
        idsInEvent.clear();
      }
      previousEvent = eventNumber;
      idsInEvent.insert(id);
      VertexPosition position{*x, *y, *z};
      if(needVertexTracks)
      {
        position.nTracks = **nTracks;
        position.inVertex = **inVertex;
      }
      positions[TrackKey(segment, eventNumber, id)] = position;
    }
    return true;
  }

  struct RawHit
  {
    int chip;
    int layer;
    double x;
    double y;
    double z;
    double sigma;
  };

  struct PendingTrack
  {
    long id = -1;
    long vertexId = -1;
    float pt = 0;
    float eta = 0;
    float minusDcaZ = 0;
    float crossing = 0;
    std::vector<RawHit> hits;
  };

  // 1 - track_in_vtx / n_tracks_vtx: the vertex finder averages the tracks of
  // the vertex, so it follows a shift of this track's z0 by that fraction.
  // Only for the fixed reconstructed vertex with vertexFollows.
  double vertex_factor(const VertexPosition& position, const Settings& settings)
  {
    if(!settings.useRecoVertex || !settings.vertexFollows || !(position.nTracks > 0))
    {
      return 1.0;
    }
    return 1.0 - std::clamp(position.inVertex, 0.0f, 1.0f) / position.nTracks;
  }

  // Applies the track selection and appends the event's vertices with at
  // least min_tracks_per_vertex selected tracks to the sample.
  void flush_event(
    std::vector<PendingTrack>& pending,
    int segment,
    long eventNumber,
    const std::map<TrackKey, VertexPosition>& positions,
    const Settings& settings,
    Sample& sample)
  {
    std::map<long, std::vector<const PendingTrack*>> byVertex;
    std::map<const PendingTrack*, VertexPosition> vertexOf;
    for(const PendingTrack& track : pending)
    {
      ReadCounts& counts = sample.counts;
      ++counts.tracks;
      if(!std::isfinite(track.crossing) || track.crossing == kUnknownCrossing)
      {
        ++counts.unknownCrossing;
        continue;
      }
      if(!(track.pt >= settings.minimumPtGeV))
      {
        ++counts.failedPt;
        continue;
      }
      if(!std::isfinite(track.minusDcaZ) ||
         std::fabs(track.minusDcaZ) > settings.maxAbsMinusDcaZMm)
      {
        ++counts.failedDca;
        continue;
      }
      bool hasLayer[kMvtxLayers] = {false, false, false};
      for(const RawHit& hit : track.hits)
      {
        hasLayer[hit.layer] = true;
      }
      if(!hasLayer[0] || !hasLayer[1] || !hasLayer[2])
      {
        ++counts.missingLayer;
        continue;
      }
      if(static_cast<int>(track.hits.size()) > kMaxHitsPerTrack)
      {
        ++counts.tooManyHits;
        continue;
      }
      const auto position = positions.find(TrackKey(segment, eventNumber, track.id));
      if(position == positions.end())
      {
        ++counts.noVertexRow;
        continue;
      }
      if(vertex_factor(position->second, settings) < kMinVertexFactor)
      {
        ++counts.ownVertex;
        continue;
      }
      byVertex[track.vertexId].push_back(&track);
      vertexOf[&track] = position->second;
    }

    for(const auto& [vertexId, tracks] : byVertex)
    {
      if(static_cast<int>(tracks.size()) < min_tracks_per_vertex(settings))
      {
        sample.counts.smallVertex += static_cast<long>(tracks.size());
        continue;
      }
      // z reference: the reconstructed vertex z of the track (the fixed
      // vertex point with useRecoVertex), else one value per vertex (shifts
      // only the fitted z_v); one slope reference per track (shifts only
      // cot_t), so the stored y are small.
      const double vertexReference = vertexOf[tracks.front()].z;
      VertexRecord vertex;
      vertex.firstTrack = static_cast<int>(sample.tracks.size());
      for(const PendingTrack* track : tracks)
      {
        const VertexPosition& position = vertexOf[track];
        const double zReference = settings.useRecoVertex ? position.z : vertexReference;
        const double cotReference = std::sinh(track->eta);
        TrackRecord record;
        record.firstHit = static_cast<int>(sample.hits.size());
        record.nHits = static_cast<int>(track->hits.size());
        record.pt = track->pt;
        record.eta = track->eta;
        record.minusDcaZNtp = track->minusDcaZ;
        record.vertexFactor = static_cast<float>(vertex_factor(position, settings));
        std::set<int> chipsOfTrack;
        for(const RawHit& raw : track->hits)
        {
          HitRecord hit;
          hit.chip = static_cast<short>(raw.chip);
          hit.layer = static_cast<short>(raw.layer);
          const double s = std::hypot(raw.x - position.x, raw.y - position.y);
          hit.s = static_cast<float>(s);
          hit.y = static_cast<float>(raw.z - zReference - cotReference * s);
          hit.sigma = static_cast<float>(raw.sigma);
          sample.hits.push_back(hit);
          sample.sumZ[raw.chip] += raw.z;
          sample.sumS[raw.chip] += s;
          ++sample.hitsPerChip[raw.chip];
          chipsOfTrack.insert(raw.chip);
        }
        for(const int chip : chipsOfTrack)
        {
          ++sample.tracksPerChip[chip];
        }
        sample.tracks.push_back(record);
      }
      vertex.nTracks = static_cast<int>(tracks.size());
      sample.vertices.push_back(vertex);
      sample.counts.tracksKept += vertex.nTracks;
      ++sample.counts.verticesKept;
    }
    pending.clear();
  }

  // Reads one ntuple file into the sample. Returns false on a missing tree
  // or branch; stops early when maxEvents is reached.
  bool read_file(const std::string& filename, const Settings& settings, Sample& sample)
  {
    std::unique_ptr<TFile> file(TFile::Open(filename.c_str(), "READ"));
    if(!file || file->IsZombie())
    {
      std::cerr << "ERROR: could not open " << filename << std::endl;
      return false;
    }
    std::map<TrackKey, VertexPosition> positions;
    if(!read_vertex_positions(
         *file, settings.useRecoVertex && settings.vertexFollows, positions))
    {
      return false;
    }
    TTree* tree = dynamic_cast<TTree*>(file->Get("ntp"));
    if(!tree || !has_branches(tree, {"layer", "stave", "chip", "xglob", "yglob",
                                     "zglob", "errY", "event", "track_id",
                                     "track_crossing", "vertex_id",
                                     "residual_vertexY", "track_eta", "track_pt"}))
    {
      std::cerr << "ERROR: no usable ntp in " << filename << std::endl;
      return false;
    }
    ++sample.counts.files;

    TTreeReader reader(tree);
    TTreeReaderValue<Float_t> layer(reader, "layer");
    TTreeReaderValue<Float_t> stave(reader, "stave");
    TTreeReaderValue<Float_t> chip(reader, "chip");
    TTreeReaderValue<Float_t> xglob(reader, "xglob");
    TTreeReaderValue<Float_t> yglob(reader, "yglob");
    TTreeReaderValue<Float_t> zglob(reader, "zglob");
    TTreeReaderValue<Float_t> errY(reader, "errY");
    TTreeReaderValue<Float_t> event(reader, "event");
    TTreeReaderValue<Float_t> trackId(reader, "track_id");
    TTreeReaderValue<Float_t> crossing(reader, "track_crossing");
    TTreeReaderValue<Float_t> vertexId(reader, "vertex_id");
    TTreeReaderValue<Float_t> minusDcaZ(reader, "residual_vertexY");
    TTreeReaderValue<Float_t> eta(reader, "track_eta");
    TTreeReaderValue<Float_t> pt(reader, "track_pt");

    int segment = 0;
    long currentEvent = -1;
    std::vector<PendingTrack> pending;
    std::set<long> finishedIds;
    auto event_selected = [&](long eventNumber)
    {
      return settings.eventParity < 0 || eventNumber % 2 == settings.eventParity;
    };
    auto finish_event = [&]()
    {
      if(currentEvent >= 0 && event_selected(currentEvent))
      {
        flush_event(pending, segment, currentEvent, positions, settings, sample);
        ++sample.counts.events;
      }
      pending.clear();
      finishedIds.clear();
    };

    while(reader.Next())
    {
      const long eventNumber = std::lround(*event);
      const long id = std::lround(*trackId);
      const bool newTrack = pending.empty() || pending.back().id != id;
      if(eventNumber != currentEvent || (newTrack && finishedIds.count(id)))
      {
        finish_event();
        if(settings.maxEvents >= 0 && sample.counts.events >= settings.maxEvents)
        {
          return true;
        }
        if(eventNumber < currentEvent || eventNumber == currentEvent)
        {
          ++segment;
        }
        currentEvent = eventNumber;
      }
      if(pending.empty() || pending.back().id != id)
      {
        if(!pending.empty())
        {
          finishedIds.insert(pending.back().id);
        }
        PendingTrack track;
        track.id = id;
        track.vertexId = std::isfinite(*vertexId) ? std::llround(*vertexId) : -1;
        track.pt = *pt;
        track.eta = *eta;
        track.minusDcaZ = *minusDcaZ;
        track.crossing = *crossing;
        pending.push_back(track);
      }

      if(!std::isfinite(*layer) || !std::isfinite(*stave) || !std::isfinite(*chip))
      {
        continue;
      }
      const int layerNumber = static_cast<int>(std::lround(*layer));
      if(layerNumber < 0 || layerNumber >= kMvtxLayers)
      {
        continue;
      }
      const int index = chip_index(layerNumber, static_cast<int>(std::lround(*stave)),
                                   static_cast<int>(std::lround(*chip)));
      if(index < 0 || !std::isfinite(*zglob) || !std::isfinite(*xglob) ||
         !std::isfinite(*yglob) || !(*errY > 0) || !std::isfinite(*errY))
      {
        continue;
      }
      pending.back().hits.push_back(RawHit{
        index, layerNumber, *xglob * kCmToMm, *yglob * kCmToMm,
        *zglob * kCmToMm, *errY});
    }
    finish_event();
    return true;
  }

  // ---------------------------------------------------------------------------
  // Track model
  // ---------------------------------------------------------------------------

  double scattering_angle(double momentumGeV, double radiationLengths)
  {
    const double beta = momentumGeV /
                        std::sqrt(momentumGeV * momentumGeV + kPionMassGeV * kPionMassGeV);
    return 0.0136 / (beta * momentumGeV) * std::sqrt(radiationLengths) *
           (1.0 + 0.038 * std::log(radiationLengths));
  }

  // In-place inverse of a symmetric positive-definite n x n matrix (row-major,
  // n <= kMaxHitsPerTrack) by Cholesky decomposition.
  bool invert_spd(int n, double* matrix)
  {
    double lower[kMaxHitsPerTrack][kMaxHitsPerTrack] = {};
    for(int j = 0; j < n; ++j)
    {
      double sum = matrix[j * n + j];
      for(int k = 0; k < j; ++k)
      {
        sum -= lower[j][k] * lower[j][k];
      }
      if(!(sum > 0))
      {
        return false;
      }
      lower[j][j] = std::sqrt(sum);
      for(int i = j + 1; i < n; ++i)
      {
        double value = matrix[i * n + j];
        for(int k = 0; k < j; ++k)
        {
          value -= lower[i][k] * lower[j][k];
        }
        lower[i][j] = value / lower[j][j];
      }
    }
    double inverseLower[kMaxHitsPerTrack][kMaxHitsPerTrack] = {};
    for(int i = 0; i < n; ++i)
    {
      inverseLower[i][i] = 1.0 / lower[i][i];
      for(int j = 0; j < i; ++j)
      {
        double sum = 0;
        for(int k = j; k < i; ++k)
        {
          sum += lower[i][k] * inverseLower[k][j];
        }
        inverseLower[i][j] = -sum / lower[i][i];
      }
    }
    for(int i = 0; i < n; ++i)
    {
      for(int j = 0; j < n; ++j)
      {
        double sum = 0;
        for(int k = std::max(i, j); k < n; ++k)
        {
          sum += inverseLower[k][i] * inverseLower[k][j];
        }
        matrix[i * n + j] = sum;
      }
    }
    return true;
  }

  // One track with its line (a, cot) profiled out:
  //   Q = W - W A S A^T W,  S = (A^T W A + prior)^-1,  A_k = (1, s_k).
  // Vertex following (factor f < 1): after the shift the vertex moves by
  // (1 - f) of the change of a, so DCA_z = a0 + f (a - a0) with a0 the
  // current hit-only a. Requiring DCA_z = 0 +- sigma_p is the prior
  //   a = c +- sigma_p / f,  c = -(1 - f) / f * a0,
  // whose mean adds g = W A S (w_p c, 0) to the reduced right-hand side
  // (Q y is replaced by Q y - g).
  struct TrackSolve
  {
    int n = 0;
    double variance[kMaxHitsPerTrack] = {};
    double weight[kMaxHitsPerTrack * kMaxHitsPerTrack] = {};
    double q[kMaxHitsPerTrack * kMaxHitsPerTrack] = {};
    double weightA[kMaxHitsPerTrack][2] = {};
    double s[2][2] = {};
    // (A^T W A)^-1 without the prior: the hit-only line fit, whose z at the
    // vertex is the DCA_z a track fit without vertex constraint measures.
    double hitOnly[2][2] = {};
    // Effective prior width sigma_p / f, weight f^2 / sigma_p^2, mean c.
    double priorSigma = 0;
    double priorWeight = 0;
    double priorMean = 0;
    double g[kMaxHitsPerTrack] = {};
    // Current hit-only a (relative to the reconstructed vertex) and f.
    double a0 = 0;
    double vertexFactor = 1;
  };

  bool prepare_track(
    const Sample& sample,
    const TrackRecord& track,
    const Settings& settings,
    TrackSolve& solve)
  {
    const int n = track.nHits;
    const HitRecord* hits = &sample.hits[track.firstHit];
    solve.n = n;
    const double sinTheta = 1.0 / std::cosh(track.eta);
    const double cotTheta = std::sinh(track.eta);
    const double momentum = track.pt * std::cosh(track.eta);

    // Radii of the L0 and L1 scattering kinks along this track.
    double kinkRadius[2] = {kMvtxRadiusMm[0], kMvtxRadiusMm[1]};
    for(int layer = 0; layer < 2; ++layer)
    {
      double sum = 0;
      int count = 0;
      for(int k = 0; k < n; ++k)
      {
        if(hits[k].layer == layer)
        {
          sum += hits[k].s;
          ++count;
        }
      }
      if(count > 0)
      {
        kinkRadius[layer] = sum / count;
      }
    }
    double kinkSlope[2] = {0, 0};
    double beamPipeSlope = 0;
    if(settings.includeScattering)
    {
      const double toCot = settings.scatteringScale / (sinTheta * sinTheta);
      kinkSlope[0] = scattering_angle(momentum, kMvtxLayerX0 / sinTheta) * toCot;
      kinkSlope[1] = kinkSlope[0];
      beamPipeSlope = scattering_angle(momentum, kBeamPipeX0 / sinTheta) * toCot;
    }

    double* covariance = solve.weight;
    for(int k = 0; k < n; ++k)
    {
      for(int l = 0; l < n; ++l)
      {
        double value = 0;
        if(k == l)
        {
          value = hits[k].sigma * hits[k].sigma + settings.hitFloorMm * settings.hitFloorMm;
        }
        for(int kink = 0; kink < 2; ++kink)
        {
          const double armK = std::max(0.0, hits[k].s - kinkRadius[kink]);
          const double armL = std::max(0.0, hits[l].s - kinkRadius[kink]);
          value += kinkSlope[kink] * kinkSlope[kink] * armK * armL;
        }
        covariance[k * n + l] = value;
      }
      solve.variance[k] = covariance[k * n + k];
    }
    if(!invert_spd(n, solve.weight))
    {
      return false;
    }

    // With the reconstructed vertex as a fixed point its z error adds to the
    // prior (a refitted z_v has no such error).
    const double vertexZ2 =
      settings.useRecoVertex ? settings.vertexZSigmaMm * settings.vertexZSigmaMm : 0.0;
    const double priorSigma2 =
      cotTheta * cotTheta * settings.vertexXYSigmaMm * settings.vertexXYSigmaMm +
      std::pow(kBeamPipeRadiusMm * beamPipeSlope, 2) + vertexZ2 +
      kPriorFloorMm * kPriorFloorMm;
    const double f = track.vertexFactor;
    solve.vertexFactor = f;
    solve.priorSigma = std::sqrt(priorSigma2) / f;
    solve.priorWeight = f * f / priorSigma2;

    double normal[2][2] = {{solve.priorWeight, 0}, {0, 0}};
    for(int k = 0; k < n; ++k)
    {
      solve.weightA[k][0] = 0;
      solve.weightA[k][1] = 0;
      for(int l = 0; l < n; ++l)
      {
        solve.weightA[k][0] += solve.weight[k * n + l];
        solve.weightA[k][1] += solve.weight[k * n + l] * hits[l].s;
      }
    }
    for(int k = 0; k < n; ++k)
    {
      normal[0][0] += solve.weightA[k][0];
      normal[0][1] += solve.weightA[k][1];
      normal[1][1] += solve.weightA[k][1] * hits[k].s;
    }
    normal[1][0] = normal[0][1];
    const double hitNormal00 = normal[0][0] - solve.priorWeight;
    const double hitDeterminant = hitNormal00 * normal[1][1] - normal[0][1] * normal[1][0];
    const double determinant = normal[0][0] * normal[1][1] - normal[0][1] * normal[1][0];
    if(!(determinant > 0) || !(hitDeterminant > 0))
    {
      return false;
    }
    solve.hitOnly[0][0] = normal[1][1] / hitDeterminant;
    solve.hitOnly[1][1] = hitNormal00 / hitDeterminant;
    solve.hitOnly[0][1] = -normal[0][1] / hitDeterminant;
    solve.hitOnly[1][0] = solve.hitOnly[0][1];
    solve.s[0][0] = normal[1][1] / determinant;
    solve.s[1][1] = normal[0][0] / determinant;
    solve.s[0][1] = -normal[0][1] / determinant;
    solve.s[1][0] = solve.s[0][1];

    for(int k = 0; k < n; ++k)
    {
      for(int l = 0; l < n; ++l)
      {
        double correction = 0;
        for(int i = 0; i < 2; ++i)
        {
          for(int j = 0; j < 2; ++j)
          {
            correction += solve.weightA[k][i] * solve.s[i][j] * solve.weightA[l][j];
          }
        }
        solve.q[k * n + l] = solve.weight[k * n + l] - correction;
      }
    }

    // Current hit-only a and the prior mean of vertex following.
    double weightedY[2] = {0, 0};
    for(int k = 0; k < n; ++k)
    {
      weightedY[0] += solve.weightA[k][0] * hits[k].y;
      weightedY[1] += solve.weightA[k][1] * hits[k].y;
    }
    solve.a0 = solve.hitOnly[0][0] * weightedY[0] + solve.hitOnly[0][1] * weightedY[1];
    solve.priorMean = f < 1.0 ? -(1.0 - f) / f * solve.a0 : 0.0;
    for(int k = 0; k < n; ++k)
    {
      solve.g[k] = (solve.weightA[k][0] * solve.s[0][0] + solve.weightA[k][1] * solve.s[1][0]) *
                   solve.priorWeight * solve.priorMean;
    }
    return true;
  }

  // ---------------------------------------------------------------------------
  // Normal equations, solution and refit
  // ---------------------------------------------------------------------------

  // C dz = R for the free chips (freeIndex: chip -> compact index or -1),
  // with z_v profiled out per vertex:
  //   C += sum E^T Q E - v v^T / m,  R += -sum E^T Q y + v (sum 1^T Q y) / m,
  //   v = sum E^T Q 1, m = sum 1^T Q 1  (E: hit -> free chip incidence).
  // With useRecoVertex z_v is the fixed reconstructed vertex (y = 0 at the
  // vertex), so the v and m terms are dropped.
  void accumulate_normal_equations(
    const Sample& sample,
    const Settings& settings,
    const std::vector<int>& freeIndex,
    int nFree,
    std::vector<double>& normal,
    std::vector<double>& rhs)
  {
    normal.assign(static_cast<size_t>(nFree) * nFree, 0.0);
    rhs.assign(nFree, 0.0);
    std::vector<int> localOf(nFree, -1);
    std::vector<int> localFree;
    std::vector<double> localNormal;
    std::vector<double> localV;
    std::vector<double> localR;
    TrackSolve solve;

    for(const VertexRecord& vertex : sample.vertices)
    {
      localFree.clear();
      double m = 0;
      double rz = 0;
      // First pass: the free chips of the vertex.
      for(int t = vertex.firstTrack; t < vertex.firstTrack + vertex.nTracks; ++t)
      {
        const TrackRecord& track = sample.tracks[t];
        if(!track.used)
        {
          continue;
        }
        for(int k = 0; k < track.nHits; ++k)
        {
          const int free = freeIndex[sample.hits[track.firstHit + k].chip];
          if(free >= 0 && localOf[free] < 0)
          {
            localOf[free] = static_cast<int>(localFree.size());
            localFree.push_back(free);
          }
        }
      }
      const int nLocal = static_cast<int>(localFree.size());
      localNormal.assign(static_cast<size_t>(nLocal) * nLocal, 0.0);
      localV.assign(nLocal, 0.0);
      localR.assign(nLocal, 0.0);

      int tracksUsed = 0;
      for(int t = vertex.firstTrack; t < vertex.firstTrack + vertex.nTracks; ++t)
      {
        const TrackRecord& track = sample.tracks[t];
        if(!track.used || !prepare_track(sample, track, settings, solve))
        {
          continue;
        }
        ++tracksUsed;
        const int n = solve.n;
        const HitRecord* hits = &sample.hits[track.firstHit];
        double q1[kMaxHitsPerTrack] = {};
        double qy[kMaxHitsPerTrack] = {};
        for(int k = 0; k < n; ++k)
        {
          for(int l = 0; l < n; ++l)
          {
            q1[k] += solve.q[k * n + l];
            qy[k] += solve.q[k * n + l] * hits[l].y;
          }
          qy[k] -= solve.g[k];
          m += q1[k];
          rz += qy[k];
        }
        for(int k = 0; k < n; ++k)
        {
          const int free = freeIndex[hits[k].chip];
          if(free < 0)
          {
            continue;
          }
          const int a = localOf[free];
          localV[a] += q1[k];
          localR[a] -= qy[k];
          for(int l = 0; l < n; ++l)
          {
            const int freeL = freeIndex[hits[l].chip];
            if(freeL >= 0)
            {
              localNormal[a * nLocal + localOf[freeL]] += solve.q[k * n + l];
            }
          }
        }
      }

      const bool fixedVertex = settings.useRecoVertex;
      if(tracksUsed >= min_tracks_per_vertex(settings) && (fixedVertex || m > 0))
      {
        const double vertexTerm = fixedVertex ? 0.0 : 1.0 / m;
        for(int a = 0; a < nLocal; ++a)
        {
          const int ga = localFree[a];
          rhs[ga] += localR[a] + localV[a] * rz * vertexTerm;
          for(int b = 0; b < nLocal; ++b)
          {
            normal[static_cast<size_t>(ga) * nFree + localFree[b]] +=
              localNormal[a * nLocal + b] - localV[a] * localV[b] * vertexTerm;
          }
        }
      }
      for(const int free : localFree)
      {
        localOf[free] = -1;
      }
    }
  }

  struct RefitResult
  {
    // Per track: DCA_z of the hit-only line fit relative to the fitted
    // vertex (what the reconstruction measures), prior pull of the fitted
    // a_t, hit chi2.
    std::vector<float> dcaZ;
    std::vector<float> priorPull;
    std::vector<float> hitChi2;
    double chi2 = 0;
    long measurements = 0;
    long tracks = 0;
    long vertices = 0;
    // Per layer: sum of (hit residual / sqrt(V_kk))^2 and hits.
    double layerSum2[kMvtxLayers] = {0, 0, 0};
    long layerHits[kMvtxLayers] = {0, 0, 0};
  };

  // Refits every vertex (z_v, a_t, cot_t) with the hits moved by dz (mm per
  // chip, all 432 chips).
  void refit_sample(
    const Sample& sample,
    const Settings& settings,
    const std::vector<double>& dz,
    RefitResult& result)
  {
    const size_t nTracks = sample.tracks.size();
    result = RefitResult();
    result.dcaZ.assign(nTracks, std::numeric_limits<float>::quiet_NaN());
    result.priorPull.assign(nTracks, std::numeric_limits<float>::quiet_NaN());
    result.hitChi2.assign(nTracks, std::numeric_limits<float>::quiet_NaN());
    std::vector<TrackSolve> solves;
    std::vector<int> trackOf;

    for(const VertexRecord& vertex : sample.vertices)
    {
      solves.clear();
      trackOf.clear();
      double m = 0;
      double rz = 0;
      for(int t = vertex.firstTrack; t < vertex.firstTrack + vertex.nTracks; ++t)
      {
        const TrackRecord& track = sample.tracks[t];
        TrackSolve solve;
        if(!track.used || !prepare_track(sample, track, settings, solve))
        {
          continue;
        }
        const HitRecord* hits = &sample.hits[track.firstHit];
        for(int k = 0; k < solve.n; ++k)
        {
          for(int l = 0; l < solve.n; ++l)
          {
            m += solve.q[k * solve.n + l];
            rz += solve.q[k * solve.n + l] * (hits[l].y + dz[hits[l].chip]);
          }
          rz -= solve.g[k];
        }
        solves.push_back(solve);
        trackOf.push_back(t);
      }
      if(static_cast<int>(solves.size()) < min_tracks_per_vertex(settings) ||
         (!settings.useRecoVertex && !(m > 0)))
      {
        continue;
      }
      // y is relative to the reconstructed vertex with useRecoVertex.
      const double zVertex = settings.useRecoVertex ? 0.0 : rz / m;
      ++result.vertices;
      if(!settings.useRecoVertex)
      {
        result.measurements -= 1;
      }
      for(size_t i = 0; i < solves.size(); ++i)
      {
        const TrackSolve& solve = solves[i];
        const int t = trackOf[i];
        const TrackRecord& track = sample.tracks[t];
        const HitRecord* hits = &sample.hits[track.firstHit];
        const int n = solve.n;
        double residual[kMaxHitsPerTrack];
        for(int k = 0; k < n; ++k)
        {
          residual[k] = hits[k].y + dz[hits[k].chip] - zVertex;
        }
        double bu[2] = {0, 0};
        for(int k = 0; k < n; ++k)
        {
          bu[0] += solve.weightA[k][0] * residual[k];
          bu[1] += solve.weightA[k][1] * residual[k];
        }
        const double priorTerm = solve.priorWeight * solve.priorMean;
        const double a = solve.s[0][0] * (bu[0] + priorTerm) + solve.s[0][1] * bu[1];
        const double cot = solve.s[1][0] * (bu[0] + priorTerm) + solve.s[1][1] * bu[1];
        const double aHitOnly = solve.hitOnly[0][0] * bu[0] + solve.hitOnly[0][1] * bu[1];
        // What the reconstruction measures: the hit-only a, of whose change
        // the refitted vertex takes (1 - f).
        const double measuredDca = solve.a0 + solve.vertexFactor * (aHitOnly - solve.a0);
        for(int k = 0; k < n; ++k)
        {
          residual[k] -= a + cot * hits[k].s;
          result.layerSum2[hits[k].layer] += residual[k] * residual[k] / solve.variance[k];
          ++result.layerHits[hits[k].layer];
        }
        double chi2 = 0;
        for(int k = 0; k < n; ++k)
        {
          for(int l = 0; l < n; ++l)
          {
            chi2 += residual[k] * solve.weight[k * n + l] * residual[l];
          }
        }
        const double pull = (a - solve.priorMean) / solve.priorSigma;
        result.dcaZ[t] = static_cast<float>(measuredDca);
        result.priorPull[t] = static_cast<float>(pull);
        result.hitChi2[t] = static_cast<float>(chi2);
        result.chi2 += chi2 + pull * pull;
        // n hits + 1 prior - 2 line parameters per track.
        result.measurements += n + 1 - 2;
        ++result.tracks;
      }
    }
  }

  TruncatedMean truncated_mean(const std::vector<float>& values)
  {
    TruncatedMean result;
    double center = 0;
    double halfWidth = kTruncatedMeanSeedMm;
    for(int iteration = 0; iteration < kTruncatedMeanIterations; ++iteration)
    {
      double sum = 0;
      double sum2 = 0;
      int n = 0;
      for(const float value : values)
      {
        if(std::fabs(value - center) < halfWidth)
        {
          sum += value;
          sum2 += static_cast<double>(value) * value;
          ++n;
        }
      }
      if(n < kMinimumValuesForMean)
      {
        return TruncatedMean();
      }
      const double mean = sum / n;
      const double rms = std::sqrt(std::max(0.0, sum2 / n - mean * mean));
      result.mean = mean;
      result.error = rms / std::sqrt(static_cast<double>(n));
      result.n = n;
      if(!(rms > 0))
      {
        break;
      }
      center = mean;
      halfWidth = kTruncatedMeanWindow * rms;
    }
    return result;
  }

  // <-DCA_z> per chip from per-track -DCA_z values, one entry per hit (as
  // every ntp row fills its chip in the dca_fits).
  std::vector<TruncatedMean> per_chip_minus_dca(
    const Sample& sample, const std::vector<float>& minusDcaPerTrack)
  {
    std::vector<std::vector<float>> values(kMvtxChips);
    for(size_t t = 0; t < sample.tracks.size(); ++t)
    {
      const TrackRecord& track = sample.tracks[t];
      if(!track.used || !std::isfinite(minusDcaPerTrack[t]))
      {
        continue;
      }
      for(int k = 0; k < track.nHits; ++k)
      {
        values[sample.hits[track.firstHit + k].chip].push_back(minusDcaPerTrack[t]);
      }
    }
    std::vector<TruncatedMean> means(kMvtxChips);
    for(int chip = 0; chip < kMvtxChips; ++chip)
    {
      means[chip] = truncated_mean(values[chip]);
    }
    return means;
  }

  struct Solution
  {
    std::vector<double> dz = std::vector<double>(kMvtxChips, 0.0);
    std::vector<double> dzError = std::vector<double>(kMvtxChips, 0.0);
    std::vector<bool> free = std::vector<bool>(kMvtxChips, false);
  };

  // Solves [[C + P, H^T], [H, 0]] (dz, lambda) = (R, 0) with TDecompBK.
  // H rows are scaled to the size of C (does not change the solution or the
  // constrained covariance, which is the top-left block of the inverse).
  bool solve_constrained(
    const std::vector<double>& normal,
    const std::vector<double>& rhs,
    const std::vector<std::vector<double>>& constraints,
    const std::vector<int>& chipOfFree,
    double presigmaMm,
    Solution& solution)
  {
    const int nFree = static_cast<int>(rhs.size());
    const int nConstraints = static_cast<int>(constraints.size());
    const int size = nFree + nConstraints;
    double meanDiagonal = 0;
    for(int i = 0; i < nFree; ++i)
    {
      meanDiagonal += normal[static_cast<size_t>(i) * nFree + i];
    }
    meanDiagonal = nFree > 0 ? meanDiagonal / nFree : 1.0;
    const double constraintScale = std::sqrt(std::max(meanDiagonal, 1e-12));

    TMatrixDSym matrix(size);
    for(int i = 0; i < nFree; ++i)
    {
      for(int j = 0; j < nFree; ++j)
      {
        matrix(i, j) = normal[static_cast<size_t>(i) * nFree + j];
      }
      if(presigmaMm > 0)
      {
        matrix(i, i) += 1.0 / (presigmaMm * presigmaMm);
      }
    }
    for(int c = 0; c < nConstraints; ++c)
    {
      for(int i = 0; i < nFree; ++i)
      {
        matrix(nFree + c, i) = constraintScale * constraints[c][i];
        matrix(i, nFree + c) = constraintScale * constraints[c][i];
      }
    }

    TDecompBK decomposition(matrix);
    if(!decomposition.Decompose())
    {
      std::cerr << "ERROR: the constrained normal matrix could not be decomposed "
                   "(singular: disconnected chips or too few tracks?)." << std::endl;
      return false;
    }
    TVectorD vector(size);
    for(int i = 0; i < nFree; ++i)
    {
      vector(i) = rhs[i];
    }
    if(!decomposition.Solve(vector))
    {
      std::cerr << "ERROR: solving the constrained normal equations failed." << std::endl;
      return false;
    }
    TMatrixDSym inverse(size);
    decomposition.Invert(inverse);
    for(int i = 0; i < nFree; ++i)
    {
      const int chip = chipOfFree[i];
      solution.dz[chip] = vector(i);
      solution.dzError[chip] = std::sqrt(std::max(0.0, inverse(i, i)));
      solution.free[chip] = true;
    }
    return true;
  }

  // Constraint rows over the free chips, each normalised to unit length.
  std::vector<std::vector<double>> make_constraints(
    const Sample& sample,
    const std::vector<int>& chipOfFree,
    bool constrainLayerMeans)
  {
    const int nFree = static_cast<int>(chipOfFree.size());
    std::vector<std::vector<double>> rows;
    auto add_row = [&](std::vector<double> row)
    {
      double norm = 0;
      for(const double value : row)
      {
        norm += value * value;
      }
      if(norm > 0)
      {
        for(double& value : row)
        {
          value /= std::sqrt(norm);
        }
        rows.push_back(row);
      }
    };
    auto mean_of = [&](const std::vector<double>& sums, int chip)
    {
      return sample.hitsPerChip[chip] > 0 ? sums[chip] / sample.hitsPerChip[chip] : 0.0;
    };

    if(constrainLayerMeans)
    {
      for(int layer = 0; layer < kMvtxLayers; ++layer)
      {
        std::vector<double> row(nFree, 0.0);
        for(int i = 0; i < nFree; ++i)
        {
          int chipLayer;
          int stave;
          int chip;
          chip_of_index(chipOfFree[i], chipLayer, stave, chip);
          row[i] = chipLayer == layer ? 1.0 : 0.0;
        }
        add_row(row);
      }
    }
    else
    {
      add_row(std::vector<double>(nFree, 1.0));
      std::vector<double> shear(nFree, 0.0);
      double meanS = 0;
      for(int i = 0; i < nFree; ++i)
      {
        shear[i] = mean_of(sample.sumS, chipOfFree[i]);
        meanS += shear[i] / nFree;
      }
      for(double& value : shear)
      {
        value -= meanS;
      }
      add_row(shear);
    }
    std::vector<double> scale(nFree, 0.0);
    for(int i = 0; i < nFree; ++i)
    {
      scale[i] = mean_of(sample.sumZ, chipOfFree[i]);
    }
    add_row(scale);
    return rows;
  }

  // The smallest eigenvalues of C (free chips, without presigma) and the
  // fraction of each eigenvector inside the span of the constraints. A small
  // eigenvalue with a small fraction is an unconstrained weak mode.
  void report_weak_modes(
    const std::vector<double>& normal,
    const std::vector<std::vector<double>>& constraints)
  {
    const int nFree = static_cast<int>(constraints.empty() ? 0 : constraints[0].size());
    if(nFree == 0)
    {
      return;
    }
    TMatrixDSym matrix(nFree);
    for(int i = 0; i < nFree; ++i)
    {
      for(int j = 0; j < nFree; ++j)
      {
        matrix(i, j) = normal[static_cast<size_t>(i) * nFree + j];
      }
    }
    TMatrixDSymEigen eigen(matrix);
    const TVectorD& values = eigen.GetEigenValues();
    const TMatrixD& vectors = eigen.GetEigenVectors();

    // Orthonormal basis of the constraint span (Gram-Schmidt).
    std::vector<std::vector<double>> basis;
    for(const std::vector<double>& row : constraints)
    {
      std::vector<double> vector = row;
      for(const std::vector<double>& base : basis)
      {
        double dot = 0;
        for(int i = 0; i < nFree; ++i)
        {
          dot += vector[i] * base[i];
        }
        for(int i = 0; i < nFree; ++i)
        {
          vector[i] -= dot * base[i];
        }
      }
      double norm = 0;
      for(const double value : vector)
      {
        norm += value * value;
      }
      if(norm > 1e-12)
      {
        for(double& value : vector)
        {
          value /= std::sqrt(norm);
        }
        basis.push_back(vector);
      }
    }

    std::vector<int> order(nFree);
    for(int i = 0; i < nFree; ++i)
    {
      order[i] = i;
    }
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return values(a) < values(b); });
    const double largest = values(order.back());
    std::cout << "\nSmallest eigenvalues of the normal matrix (relative to the "
                 "largest) and the fraction of the mode fixed by the constraints:\n";
    for(int r = 0; r < std::min(kEigenvaluesReported, nFree); ++r)
    {
      const int index = order[r];
      double inside = 0;
      for(const std::vector<double>& base : basis)
      {
        double dot = 0;
        for(int i = 0; i < nFree; ++i)
        {
          dot += vectors(i, index) * base[i];
        }
        inside += dot * dot;
      }
      std::cout << "  " << std::scientific << std::setprecision(2)
                << values(index) / largest << std::fixed << "   constrained "
                << std::setprecision(3) << inside
                << (values(index) / largest < 1e-6 && inside < 0.5
                      ? "   <- WARNING: unconstrained weak mode"
                      : "")
                << '\n';
    }
    std::cout << std::defaultfloat;
  }

  // Least-squares slope and offset of y versus x and the RMS of y - x. The
  // comparison only means something if the chip spread is well above the
  // typical per-chip error (otherwise both are noise).
  void compare_values(
    const char* title,
    const std::vector<double>& x,
    const std::vector<double>& y,
    double typicalErrorMm)
  {
    const size_t n = x.size();
    if(n < 3)
    {
      std::cout << title << ": fewer than 3 chips to compare\n";
      return;
    }
    double mx = 0;
    double my = 0;
    for(size_t i = 0; i < n; ++i)
    {
      mx += x[i] / n;
      my += y[i] / n;
    }
    double sxx = 0;
    double sxy = 0;
    double sdd = 0;
    for(size_t i = 0; i < n; ++i)
    {
      sxx += (x[i] - mx) * (x[i] - mx);
      sxy += (x[i] - mx) * (y[i] - my);
      sdd += (y[i] - x[i]) * (y[i] - x[i]);
    }
    const double slope = sxx > 0 ? sxy / sxx : std::numeric_limits<double>::quiet_NaN();
    const double rmsDifference = std::sqrt(sdd / n);
    const double spread = std::sqrt(sxx / n);
    const bool meaningful = spread > 3.0 * typicalErrorMm;
    std::cout << std::fixed << std::setprecision(3) << title << ": " << n
              << " chips, slope " << slope << ", offset "
              << std::setprecision(2) << 1e3 * (my - slope * mx)
              << " um, RMS(difference) " << 1e3 * rmsDifference
              << " um, chip spread " << 1e3 * spread << " um, typical error "
              << 1e3 * typicalErrorMm << " um"
              << (!meaningful
                    ? "   (spread < 3x error: noise only, no test)"
                    : (slope < 0.8 || slope > 1.2 || rmsDifference > 0.5 * spread)
                        ? "   <- WARNING: check before applying"
                        : "   OK")
              << std::defaultfloat << '\n';
  }

  // MVTX dca_z mu_mm (fit_status ok) of a dca_fits_<label>.csv.
  bool read_dca_fits(const char* path, std::vector<double>& mu)
  {
    mu.assign(kMvtxChips, std::numeric_limits<double>::quiet_NaN());
    std::ifstream input(path);
    std::string text;
    if(!input.is_open() || !std::getline(input, text))
    {
      std::cerr << "ERROR: could not read " << path << std::endl;
      return false;
    }
    const std::vector<std::string> header = split_csv(text);
    std::map<std::string, size_t> column;
    for(size_t i = 0; i < header.size(); ++i)
    {
      column[header[i]] = i;
    }
    for(const char* name : {"detector", "layer", "stave", "chip", "coord", "fit_status", "mu_mm"})
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
      if(fields.size() != header.size() || fields[column["detector"]] != "MVTX" ||
         fields[column["coord"]] != "dca_z" || fields[column["fit_status"]] != "ok")
      {
        continue;
      }
      if(column.count("label"))
      {
        labels.insert(fields[column["label"]]);
      }
      const int index = chip_index(std::stoi(fields[column["layer"]]),
                                   std::stoi(fields[column["stave"]]),
                                   std::stoi(fields[column["chip"]]));
      if(index >= 0)
      {
        mu[index] = std::strtod(fields[column["mu_mm"]].c_str(), nullptr);
      }
    }
    if(labels.size() > 1)
    {
      std::cerr << "ERROR: " << path << " holds " << labels.size()
                << " labels; give a single-label dca_fits CSV." << std::endl;
      return false;
    }
    return true;
  }

  // chi2/ndf, normalised hit residuals per layer and prior pulls of a refit.
  void print_fit_quality(
    const char* title,
    const Sample& sample,
    const RefitResult& refit,
    long freeParameters)
  {
    const long ndf = refit.measurements - freeParameters;
    std::cout << std::fixed << std::setprecision(3) << "\n" << title
              << ": chi2/ndf = " << refit.chi2 << "/" << ndf << " = "
              << (ndf > 0 ? refit.chi2 / ndf : 0.0) << " (" << refit.tracks
              << " tracks, " << refit.vertices << " vertices)\n";

    const double ptEdges[] = {0, 1.0, 2.0, 1e9};
    const double cotEdges[] = {0, 0.4, 0.8, 1e9};
    double ptSum2[3] = {0, 0, 0};
    long ptCount[3] = {0, 0, 0};
    double cotSum2[3] = {0, 0, 0};
    long cotCount[3] = {0, 0, 0};
    for(size_t t = 0; t < sample.tracks.size(); ++t)
    {
      const TrackRecord& track = sample.tracks[t];
      if(!track.used || !std::isfinite(refit.priorPull[t]))
      {
        continue;
      }
      const double pull = refit.priorPull[t];
      const double cot = std::fabs(std::sinh(track.eta));
      for(int bin = 0; bin < 3; ++bin)
      {
        if(track.pt >= ptEdges[bin] && track.pt < ptEdges[bin + 1])
        {
          ptSum2[bin] += pull * pull;
          ++ptCount[bin];
        }
        if(cot >= cotEdges[bin] && cot < cotEdges[bin + 1])
        {
          cotSum2[bin] += pull * pull;
          ++cotCount[bin];
        }
      }
    }
    std::cout << "  RMS prior pull (DCA_z / sigma_p) in pT bins <1, 1-2, >2 GeV: ";
    for(int bin = 0; bin < 3; ++bin)
    {
      std::cout << (ptCount[bin] > 0 ? std::sqrt(ptSum2[bin] / ptCount[bin]) : 0.0) << " ";
    }
    std::cout << "\n  RMS prior pull in |cot theta| bins <0.4, 0.4-0.8, >0.8: ";
    for(int bin = 0; bin < 3; ++bin)
    {
      std::cout << (cotCount[bin] > 0 ? std::sqrt(cotSum2[bin] / cotCount[bin]) : 0.0) << " ";
    }
    // Residuals of fitted tracks have less than the hit variance, so these
    // are below 1 even for a correct error model; compare before and after.
    std::cout << "\n  RMS hit residual / sqrt(V_kk) in L0, L1, L2: ";
    for(int layer = 0; layer < kMvtxLayers; ++layer)
    {
      std::cout << (refit.layerHits[layer] > 0
                      ? std::sqrt(refit.layerSum2[layer] / refit.layerHits[layer])
                      : 0.0)
                << " ";
    }
    std::cout << std::defaultfloat << std::endl;
  }

  // dz per chip (column dz_mm, by layer, stave, chip) of a _dz_summary.csv
  // of makeDcaZShiftedAlignment.C (the applied, damped dz) or of a
  // dz_solution CSV; chips not listed get 0.
  bool read_dz_csv(const char* path, std::vector<double>& dz)
  {
    dz.assign(kMvtxChips, 0.0);
    std::ifstream input(path);
    std::string text;
    if(!input.is_open() || !std::getline(input, text))
    {
      std::cerr << "ERROR: could not read " << path << std::endl;
      return false;
    }
    const std::vector<std::string> header = split_csv(text);
    std::map<std::string, size_t> column;
    for(size_t i = 0; i < header.size(); ++i)
    {
      column[header[i]] = i;
    }
    for(const char* name : {"layer", "stave", "chip", "dz_mm"})
    {
      if(!column.count(name))
      {
        std::cerr << "ERROR: column " << name << " missing in " << path << std::endl;
        return false;
      }
    }
    int rows = 0;
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
      const int index = chip_index(std::stoi(fields[column["layer"]]),
                                   std::stoi(fields[column["stave"]]),
                                   std::stoi(fields[column["chip"]]));
      const double value = std::strtod(fields[column["dz_mm"]].c_str(), nullptr);
      if(index >= 0 && std::isfinite(value))
      {
        dz[index] = value;
        ++rows;
      }
    }
    std::cout << "Read dz of " << rows << " chips from " << path << std::endl;
    return rows > 0;
  }

  // Slope (r) and correlation of y versus x.
  void print_response(const char* title, const std::vector<double>& x, const std::vector<double>& y)
  {
    const size_t n = x.size();
    if(n < 3)
    {
      std::cout << "  " << title << ": fewer than 3 chips\n";
      return;
    }
    double mx = 0;
    double my = 0;
    for(size_t i = 0; i < n; ++i)
    {
      mx += x[i] / n;
      my += y[i] / n;
    }
    double sxx = 0;
    double syy = 0;
    double sxy = 0;
    for(size_t i = 0; i < n; ++i)
    {
      sxx += (x[i] - mx) * (x[i] - mx);
      syy += (y[i] - my) * (y[i] - my);
      sxy += (x[i] - mx) * (y[i] - my);
    }
    std::cout << std::fixed << std::setprecision(2) << "  " << std::left << std::setw(10)
              << title << std::right << std::setw(5) << n << " chips   r = "
              << (sxx > 0 ? sxy / sxx : 0.0) << "   corr "
              << (sxx > 0 && syy > 0 ? sxy / std::sqrt(sxx * syy) : 0.0)
              << std::setprecision(1) << "   RMS predicted " << 1e3 * std::sqrt(sxx / n)
              << " um, measured " << 1e3 * std::sqrt(syy / n) << " um"
              << std::defaultfloat << '\n';
  }

  // Prediction only: the change of <-DCA_z> per chip that the given dz
  // should cause, compared (with both CSVs) with the measured change between
  // the dca_fits of the run made with the base alignment (dcaFitsCsv) and of
  // the run made with the shifted one (measuredAfterCsv). r = measured /
  // predicted ~ 1 means the error model, vertexFollows and scatteringScale
  // describe the reconstruction's response.
  void run_prediction(
    Sample& sample,
    const Settings& settings,
    const std::string& label,
    const char* predictDzCsv,
    const char* dcaFitsCsv,
    const char* measuredAfterCsv,
    const char* csvOutputDirectory)
  {
    std::vector<double> dz;
    if(!read_dz_csv(predictDzCsv, dz))
    {
      return;
    }
    // Outliers are judged with the given dz applied, as in the solve (before
    // the shift the misalignment itself makes tracks look like outliers).
    const std::vector<double> noShift(kMvtxChips, 0.0);
    RefitResult after;
    refit_sample(sample, settings, dz, after);
    long dropped = 0;
    for(size_t t = 0; t < sample.tracks.size(); ++t)
    {
      TrackRecord& track = sample.tracks[t];
      if(track.used && (!std::isfinite(after.priorPull[t]) ||
                        std::fabs(after.priorPull[t]) > kMaxPriorPull ||
                        after.hitChi2[t] > kMaxTrackHitChi2))
      {
        track.used = false;
        ++dropped;
      }
    }
    std::cout << "Dropped " << dropped << " outlier tracks (with the given dz)" << std::endl;
    RefitResult before;
    refit_sample(sample, settings, noShift, before);
    refit_sample(sample, settings, dz, after);
    print_fit_quality("Before the shift", sample, before, 0);
    print_fit_quality("With the given dz", sample, after, 0);

    std::vector<float> minusBefore(sample.tracks.size());
    std::vector<float> minusAfter(sample.tracks.size());
    for(size_t t = 0; t < sample.tracks.size(); ++t)
    {
      minusBefore[t] = -before.dcaZ[t];
      minusAfter[t] = -after.dcaZ[t];
    }
    const std::vector<TruncatedMean> muBefore = per_chip_minus_dca(sample, minusBefore);
    const std::vector<TruncatedMean> muAfter = per_chip_minus_dca(sample, minusAfter);

    std::vector<double> measuredBefore(kMvtxChips, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> measuredAfter(kMvtxChips, std::numeric_limits<double>::quiet_NaN());
    const bool haveMeasured = dcaFitsCsv && *dcaFitsCsv && measuredAfterCsv && *measuredAfterCsv;
    if(haveMeasured && (!read_dca_fits(dcaFitsCsv, measuredBefore) ||
                        !read_dca_fits(measuredAfterCsv, measuredAfter)))
    {
      return;
    }
    if(haveMeasured)
    {
      std::cout << "\nMeasured change of <-DCA_z> (" << measuredAfterCsv << " - " << dcaFitsCsv
                << ") versus predicted change, chips with dz != 0:\n";
      std::vector<double> x[kMvtxLayers + 1];
      std::vector<double> y[kMvtxLayers + 1];
      for(int chip = 0; chip < kMvtxChips; ++chip)
      {
        const double predicted = muAfter[chip].mean - muBefore[chip].mean;
        const double measured = measuredAfter[chip] - measuredBefore[chip];
        if(dz[chip] == 0 || !std::isfinite(predicted) || !std::isfinite(measured))
        {
          continue;
        }
        int layer;
        int stave;
        int chipInStave;
        chip_of_index(chip, layer, stave, chipInStave);
        for(const int bin : {layer, kMvtxLayers})
        {
          x[bin].push_back(predicted);
          y[bin].push_back(measured);
        }
      }
      print_response("all", x[kMvtxLayers], y[kMvtxLayers]);
      for(int layer = 0; layer < kMvtxLayers; ++layer)
      {
        print_response(("L" + std::to_string(layer)).c_str(), x[layer], y[layer]);
      }
    }

    const std::string csvPath = std::string(csvOutputDirectory) + "/dz_prediction_" +
                                sanitize_for_filename(label) + ".csv";
    std::ofstream csv(csvPath);
    if(!csv.is_open())
    {
      std::cerr << "ERROR: could not write " << csvPath << std::endl;
      return;
    }
    csv << std::setprecision(10)
        << "label,layer,stave,chip,stavechip,n_tracks,dz_mm,mu_before_model_mm,"
           "mu_pred_after_mm,mu_pred_change_mm,mu_meas_before_mm,mu_meas_after_mm\n";
    for(int chip = 0; chip < kMvtxChips; ++chip)
    {
      int layer;
      int stave;
      int chipInStave;
      chip_of_index(chip, layer, stave, chipInStave);
      csv << label << ',' << layer << ',' << stave << ',' << chipInStave << ','
          << kChipsPerStave * stave + chipInStave << ',' << sample.tracksPerChip[chip] << ','
          << dz[chip] << ',' << muBefore[chip].mean << ',' << muAfter[chip].mean << ','
          << muAfter[chip].mean - muBefore[chip].mean << ',' << measuredBefore[chip] << ','
          << measuredAfter[chip] << '\n';
    }
    std::cout << "\nWrote " << csvPath << std::endl;
  }
}

// Writes dz_solution_<label>.csv (see the header comment for the model, the
// columns and the checks before applying the solution).
void solveDcaZAlignment(
  const char* listfileSpec = "filelist1.txt:Iter1",
  const char* dcaFitsCsv = "",
  double minimumPtGeV = 0.65,
  int minTracksPerChip = 50,
  bool constrainLayerMeans = true,
  double presigmaMm = 0.05,
  double hitFloorMm = 0.005,
  double vertexXYSigmaMm = 0.03,
  bool includeScattering = true,
  double maxAbsMinusDcaZMm = 0.5,
  int eventParity = -1,
  Long64_t maxEvents = -1,
  const char* csvOutputDirectory = ".",
  bool useRecoVertex = true,
  double vertexZSigmaMm = 0.02,
  bool vertexFollows = true,
  double scatteringScale = 1.0,
  const char* predictDzCsv = "",
  const char* measuredAfterCsv = "")
{
  std::string listfile;
  std::string label;
  if(!parse_input_spec(listfileSpec, listfile, label))
  {
    return;
  }
  std::vector<std::string> files;
  if(!read_file_list(listfile, files))
  {
    return;
  }

  Settings settings;
  settings.minimumPtGeV = minimumPtGeV;
  settings.hitFloorMm = hitFloorMm;
  settings.vertexXYSigmaMm = vertexXYSigmaMm;
  settings.includeScattering = includeScattering;
  settings.maxAbsMinusDcaZMm = maxAbsMinusDcaZMm;
  settings.eventParity = eventParity;
  settings.maxEvents = maxEvents;
  settings.useRecoVertex = useRecoVertex;
  settings.vertexZSigmaMm = vertexZSigmaMm;
  settings.vertexFollows = vertexFollows;
  settings.scatteringScale = scatteringScale;
  std::cout << "Vertex: "
            << (useRecoVertex
                  ? "reconstructed (track_ntp event_zvtx), sigma_z = " +
                      std::to_string(vertexZSigmaMm) + " mm" +
                      (vertexFollows ? ", follows the tracks (1 - track_in_vtx / n_tracks_vtx)"
                                     : ", fixed")
                  : std::string("refitted from the selected tracks"))
            << "; hit floor " << hitFloorMm << " mm; scattering "
            << (includeScattering ? "x " + std::to_string(scatteringScale) : std::string("off"))
            << std::endl;

  Sample sample;
  for(const std::string& filename : files)
  {
    std::cout << "Reading " << filename << std::endl;
    if(!read_file(filename, settings, sample))
    {
      return;
    }
    if(maxEvents >= 0 && sample.counts.events >= maxEvents)
    {
      break;
    }
  }
  const ReadCounts& counts = sample.counts;
  std::cout << label << ": " << counts.files << " files, " << counts.events
            << " events, " << counts.tracks << " tracks; rejected: "
            << counts.unknownCrossing << " unknown crossing, " << counts.failedPt
            << " pT < " << minimumPtGeV << " GeV, " << counts.failedDca
            << " |-DCA_z| > " << maxAbsMinusDcaZMm << " mm or NaN, "
            << counts.missingLayer << " without a hit in every MVTX layer, "
            << counts.tooManyHits << " > " << kMaxHitsPerTrack << " hits, "
            << counts.noVertexRow << " without track_ntp row, "
            << counts.ownVertex << " (almost) alone in their vertex, "
            << counts.smallVertex << " in vertices with < " << min_tracks_per_vertex(settings)
            << " tracks; kept " << counts.tracksKept << " tracks in "
            << counts.verticesKept << " vertices" << std::endl;
  if(sample.vertices.empty())
  {
    std::cerr << "ERROR: no vertex passed the selection." << std::endl;
    return;
  }
  if(predictDzCsv && *predictDzCsv)
  {
    run_prediction(sample, settings, label, predictDzCsv, dcaFitsCsv, measuredAfterCsv,
                   csvOutputDirectory);
    return;
  }

  std::vector<int> freeIndex(kMvtxChips, -1);
  std::vector<int> chipOfFree;
  for(int chip = 0; chip < kMvtxChips; ++chip)
  {
    if(sample.tracksPerChip[chip] >= minTracksPerChip)
    {
      freeIndex[chip] = static_cast<int>(chipOfFree.size());
      chipOfFree.push_back(chip);
    }
  }
  const int nFree = static_cast<int>(chipOfFree.size());
  if(nFree == 0)
  {
    std::cerr << "ERROR: no chip has " << minTracksPerChip << " tracks." << std::endl;
    return;
  }
  std::cout << nFree << " of " << kMvtxChips << " chips free (>= "
            << minTracksPerChip << " tracks)" << std::endl;
  const std::vector<std::vector<double>> constraints =
    make_constraints(sample, chipOfFree, constrainLayerMeans);

  // Pass 1, outlier removal with its solution, pass 2.
  Solution solution;
  std::vector<double> normal;
  std::vector<double> rhs;
  RefitResult refit;
  for(int pass = 1; pass <= 2; ++pass)
  {
    accumulate_normal_equations(sample, settings, freeIndex, nFree, normal, rhs);
    if(!solve_constrained(normal, rhs, constraints, chipOfFree, presigmaMm, solution))
    {
      return;
    }
    if(pass == 1)
    {
      refit_sample(sample, settings, solution.dz, refit);
      long dropped = 0;
      for(size_t t = 0; t < sample.tracks.size(); ++t)
      {
        TrackRecord& track = sample.tracks[t];
        if(track.used && (!std::isfinite(refit.priorPull[t]) ||
                          std::fabs(refit.priorPull[t]) > kMaxPriorPull ||
                          refit.hitChi2[t] > kMaxTrackHitChi2))
        {
          track.used = false;
          ++dropped;
        }
      }
      std::cout << "Pass 1: dropped " << dropped << " outlier tracks (prior pull > "
                << kMaxPriorPull << " or hit chi2 > " << kMaxTrackHitChi2 << ")"
                << std::endl;
    }
  }
  report_weak_modes(normal, constraints);

  const std::vector<double> noShift(kMvtxChips, 0.0);
  RefitResult before;
  RefitResult after;
  refit_sample(sample, settings, noShift, before);
  refit_sample(sample, settings, solution.dz, after);
  const long freeParameters = nFree - static_cast<long>(constraints.size());
  print_fit_quality("Before the shift", sample, before, 0);
  print_fit_quality("After the shift ", sample, after, freeParameters);
  const long ndf = after.measurements - freeParameters;
  const double errorScale = ndf > 0 ? std::sqrt(std::max(1.0, after.chi2 / ndf)) : 1.0;
  for(double& error : solution.dzError)
  {
    error *= errorScale;
  }

  std::vector<float> minusDcaNtp(sample.tracks.size());
  std::vector<float> minusDcaBefore(sample.tracks.size());
  std::vector<float> minusDcaAfter(sample.tracks.size());
  for(size_t t = 0; t < sample.tracks.size(); ++t)
  {
    minusDcaNtp[t] = sample.tracks[t].minusDcaZNtp;
    minusDcaBefore[t] = -before.dcaZ[t];
    minusDcaAfter[t] = -after.dcaZ[t];
  }
  const std::vector<TruncatedMean> muNtp = per_chip_minus_dca(sample, minusDcaNtp);
  const std::vector<TruncatedMean> muBefore = per_chip_minus_dca(sample, minusDcaBefore);
  const std::vector<TruncatedMean> muAfter = per_chip_minus_dca(sample, minusDcaAfter);

  // Checks before applying.
  std::cout << "\nChecks (per chip <-DCA_z>):\n";
  double typicalError = 0;
  {
    std::vector<double> errors;
    for(const TruncatedMean& mean : muNtp)
    {
      if(std::isfinite(mean.error))
      {
        errors.push_back(mean.error);
      }
    }
    if(!errors.empty())
    {
      std::nth_element(errors.begin(), errors.begin() + errors.size() / 2, errors.end());
      typicalError = errors[errors.size() / 2];
    }
  }
  {
    std::vector<double> x;
    std::vector<double> y;
    for(int chip = 0; chip < kMvtxChips; ++chip)
    {
      if(std::isfinite(muNtp[chip].mean) && std::isfinite(muBefore[chip].mean))
      {
        x.push_back(muNtp[chip].mean);
        y.push_back(muBefore[chip].mean);
      }
    }
    compare_values("  model before vs ntuple (must agree)", x, y, typicalError);
  }
  if(dcaFitsCsv && *dcaFitsCsv)
  {
    std::vector<double> csvMu;
    if(read_dca_fits(dcaFitsCsv, csvMu))
    {
      std::vector<double> x;
      std::vector<double> y;
      for(int chip = 0; chip < kMvtxChips; ++chip)
      {
        if(std::isfinite(csvMu[chip]) && std::isfinite(muNtp[chip].mean))
        {
          x.push_back(csvMu[chip]);
          y.push_back(muNtp[chip].mean);
        }
      }
      compare_values("  ntuple vs dca_fits CSV (same pT cut)", x, y, typicalError);
    }
  }
  {
    double before2 = 0;
    double after2 = 0;
    int n = 0;
    for(int chip = 0; chip < kMvtxChips; ++chip)
    {
      if(std::isfinite(muBefore[chip].mean) && std::isfinite(muAfter[chip].mean))
      {
        before2 += muBefore[chip].mean * muBefore[chip].mean;
        after2 += muAfter[chip].mean * muAfter[chip].mean;
        ++n;
      }
    }
    if(n > 0)
    {
      std::cout << std::fixed << std::setprecision(2)
                << "  RMS over chips of <-DCA_z> in the model: before "
                << 1e3 * std::sqrt(before2 / n) << " um, predicted after "
                << 1e3 * std::sqrt(after2 / n) << " um (" << n << " chips)"
                << std::defaultfloat << '\n';
    }
  }

  // Per-layer mean dz and the solution by stave-chip ID.
  const std::streamsize oldPrecision = std::cout.precision();
  for(int layer = 0; layer < kMvtxLayers; ++layer)
  {
    double sum = 0;
    int n = 0;
    for(int chip = kFirstChipOfLayer[layer];
        chip < kFirstChipOfLayer[layer] + kChipsPerStave * kStavesPerLayer[layer]; ++chip)
    {
      if(solution.free[chip])
      {
        sum += solution.dz[chip];
        ++n;
      }
    }
    std::cout << "\nMVTX layer " << layer << ": dz vs stave-chip ID = 9*stave + chip ("
              << n << " free chips, mean dz " << std::fixed << std::setprecision(5)
              << (n > 0 ? sum / n : 0.0) << " mm)\n"
              << "  stavechip  stave  chip  tracks       dz_mm   dz_err   mu_ntp  "
                 "mu_model  mu_after  note\n";
    for(int chip = kFirstChipOfLayer[layer];
        chip < kFirstChipOfLayer[layer] + kChipsPerStave * kStavesPerLayer[layer]; ++chip)
    {
      int chipLayer;
      int stave;
      int chipInStave;
      chip_of_index(chip, chipLayer, stave, chipInStave);
      std::cout << std::setw(11) << kChipsPerStave * stave + chipInStave << std::setw(7)
                << stave << std::setw(6) << chipInStave << std::setw(8)
                << sample.tracksPerChip[chip] << std::showpos << std::setprecision(5)
                << std::setw(12) << solution.dz[chip] << std::noshowpos
                << std::setw(9) << solution.dzError[chip] << std::showpos
                << std::setprecision(4) << std::setw(9) << muNtp[chip].mean
                << std::setw(10) << muBefore[chip].mean << std::setw(10)
                << muAfter[chip].mean << std::noshowpos << "  "
                << (solution.free[chip] ? "" : "fixed_low_stats") << '\n';
    }
  }
  std::cout << std::defaultfloat << std::setprecision(oldPrecision) << std::flush;

  const std::string csvPath = std::string(csvOutputDirectory) + "/dz_solution_" +
                              sanitize_for_filename(label) + ".csv";
  std::ofstream csv(csvPath);
  if(!csv.is_open())
  {
    std::cerr << "ERROR: could not write " << csvPath << std::endl;
    return;
  }
  csv << std::setprecision(10)
      << "label,layer,stave,chip,stavechip,n_tracks,free,dz_mm,dz_err_mm,"
         "mu_before_ntp_mm,mu_before_model_mm,mu_pred_after_mm,mu_err_mm\n";
  for(int chip = 0; chip < kMvtxChips; ++chip)
  {
    int layer;
    int stave;
    int chipInStave;
    chip_of_index(chip, layer, stave, chipInStave);
    csv << label << ',' << layer << ',' << stave << ',' << chipInStave << ','
        << kChipsPerStave * stave + chipInStave << ',' << sample.tracksPerChip[chip]
        << ',' << (solution.free[chip] ? 1 : 0) << ',' << solution.dz[chip] << ','
        << solution.dzError[chip] << ',' << muNtp[chip].mean << ','
        << muBefore[chip].mean << ',' << muAfter[chip].mean << ','
        << muNtp[chip].error << '\n';
  }
  std::cout << "\nWrote " << csvPath << ". dz is relative to the alignment file "
               "these ntuples were made with; apply it to that file with "
               "applyDzSolution (makeDcaZShiftedAlignment.C)." << std::endl;
}
