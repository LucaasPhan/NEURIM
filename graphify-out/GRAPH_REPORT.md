# Graph Report - NEURIM  (2026-07-11)

## Corpus Check
- 86 files · ~67,838 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 860 nodes · 1694 edges · 62 communities (55 shown, 7 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 125 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ed39db25`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29
- Community 30
- Community 31
- Community 32
- Community 33
- Community 34
- Community 35
- Community 37
- Community 38
- Community 39
- Community 40
- NEURIM
- faa.py
- run_real_eeg_optimizer.py
- MockEEGSource
- Plan: `scripts/run_stablediffusion.py` — real EEG driving the breed-mixture morph
- Known issues / open work — handoff summary
- test_faa_stream.py
- test_faa.py
- FAARewardComputer
- latestPR.md
- .read_reward
- AGENTS.md

## God Nodes (most connected - your core abstractions)
1. `Config` - 48 edges
2. `FAARewardComputer` - 33 edges
3. `EmotivCortexSource` - 32 edges
4. `OptimizerService` - 28 edges
5. `GeneratorService` - 25 edges
6. `LocalOrchestrator` - 25 edges
7. `NoiseAwareLatentTuRBO` - 24 edges
8. `PCAProjector` - 24 edges
9. `RewardMessage` - 23 edges
10. `Interpolator` - 23 edges

## Surprising Connections (you probably didn't know these)
- `_SessionSnapshot` --uses--> `Config`  [INFERRED]
  scripts/run_demo.py → src/common/config.py
- `_SessionSnapshot` --uses--> `GeneratorService`  [INFERRED]
  scripts/run_demo.py → src/generator/service.py
- `_SessionSnapshot` --uses--> `LocalOrchestrator`  [INFERRED]
  scripts/run_demo.py → src/orchestrator/orchestrator.py
- `_SessionSnapshot` --uses--> `WebSocketOrchestrator`  [INFERRED]
  scripts/run_demo.py → src/orchestrator/orchestrator.py
- `_SessionSnapshot` --uses--> `EmotivCortexSource`  [INFERRED]
  scripts/run_demo.py → src/signal_service/eeg_sources.py

## Import Cycles
- None detected.

## Communities (62 total, 7 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.18
Nodes (6): PairFAAMetrics, ndarray, Pair-level alpha power and raw FAA values for diagnostics., Compact EEG visualization payload for the frontend.          Uses alpha-band pow, Baseline-normalized r(t), or None if the buffer isn't full yet., _weighted_average()

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (36): _erf(), NoiseAwareLatentTuRBO, ndarray, Noise-Aware Latent TuRBO: a trust-region Bayesian optimizer built for the noisy,, Per-dim GP length scales (ARD), for shaping the trust region., TuRBO box: side length `self.length` scaled per-dim by the ARD length         sc, math.erf on a scalar (avoids importing scipy just for the normal CDF)., effective_sample_size() (+28 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (40): main(), EEGConfig, FAAConfig, GeneratorConfig, LoopConfig, OptimizerConfig, PresentationConfig, Path (+32 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (36): channelNames, ElectrodeNodes(), fallbackPositions, normalizeChannels(), BrainActivity3D, EEGFeatures, FaaRewardBar(), formatMs() (+28 more)

### Community 4 - "Community 4"
Cohesion: 0.14
Nodes (31): add_hud(), blend_noise_latents(), blend_prompt_embeds(), cosine_ease(), encode_breed_prompts(), FakeFAAReward, interpolate_z(), load_pipeline() (+23 more)

### Community 5 - "Community 5"
Cohesion: 0.10
Nodes (11): BrainFlowLSLSource, EmotivCortexSource, Any, Return Cortex EEG column labels from a subscribe response., Pulls EEG from an LSL stream (e.g. BrainFlow's LSL output). Lazy-imports     pyl, EMOTIV Cortex API client for the EPOC X headset (WebSocket JSON-RPC).      Flow:, FakeWebSocket, test_emotiv_extracts_eeg_cols_from_subscribe_result() (+3 more)

### Community 6 - "Community 6"
Cohesion: 0.06
Nodes (31): class-variance-authority, clsx, dependencies, class-variance-authority, clsx, lucide-react, next, openai (+23 more)

### Community 7 - "Community 7"
Cohesion: 0.07
Nodes (29): eslint, eslint-config-next, devDependencies, eslint, eslint-config-next, tailwindcss, @tailwindcss/postcss, @types/node (+21 more)

### Community 8 - "Community 8"
Cohesion: 0.07
Nodes (29): compilerOptions, allowJs, esModuleInterop, incremental, isolatedModules, jsx, lib, module (+21 more)

### Community 9 - "Community 9"
Cohesion: 0.13
Nodes (13): DiffusionRenderServer, _fit_projector(), main(), make_handler(), main(), DiffusionGenerator, ndarray, SDXL-Turbo / LCM wrapper: latent (well, prompt-embedding) in, frame out in ~100- (+5 more)

### Community 10 - "Community 10"
Cohesion: 0.23
Nodes (9): build_anchor_prompts(), _fit_projector(), LatentWalkRenderServer, load_pipeline(), main(), make_handler(), ndarray, Load a plain diffusers pipeline plus a fixed, seeded noise latent sized     for (+1 more)

### Community 11 - "Community 11"
Cohesion: 0.17
Nodes (24): add_hud(), cosine_ease(), encode_breed_prompts(), load_pipeline(), main(), open_video_writer(), parse_args(), device (+16 more)

### Community 12 - "Community 12"
Cohesion: 0.16
Nodes (14): LatentMorpher, morph_path(), ndarray, Real-time latent morphing between a jumpy stream of target latents and a smooth,, max_step:  max Euclidean distance z may move per step() call (per, Advance toward `target` by at most max_step; return the new z., Lower bound on frames to reach `target` at the current max_step -         useful, Fixed-endpoint linear path of `n` intermediate latents (inclusive of     z_new, (+6 more)

### Community 13 - "Community 13"
Cohesion: 0.14
Nodes (11): Wire format for the websocket messages passed between services.  Signal -> Orche, Signal service -> Orchestrator. One scalar reward reading., RewardMessage, KeyboardRewardSource, Fake reward sources with the exact same interface FAA reward has: a scalar in [-, Common interface: FAARewardComputer-backed or fake, doesn't matter., Up/down arrow keys nudge reward; it decays toward 0 between presses.      Uses `, Deterministic reward = similarity between a hidden target and whatever     z `ge (+3 more)

### Community 14 - "Community 14"
Cohesion: 0.25
Nodes (5): Protocol, Real FAA needs 30s of rest to fit the baseline; fake reward sources         (key, EEGSource, FAARewardSource, Wraps an EEGSource + FAARewardComputer behind the RewardSource interface.

### Community 15 - "Community 15"
Cohesion: 0.13
Nodes (9): AnchorInterpolationProjector, PCAProjector, ndarray, Reduce the search space from the raw latent/embedding dim down to 8-16 dims, per, Low-dim search vector <-> full embedding, via PCA fit on a prompt bank., embeddings: [n_prompts, embed_dim], z is a weight vector over `anchor_embeddings`; softmax-normalized so     the pro, test_anchor_projector_stays_in_convex_hull() (+1 more)

### Community 16 - "Community 16"
Cohesion: 0.20
Nodes (15): blend_noise_latents(), blend_prompt_embeds(), BreedMorphRenderServer, encode_breed_prompts(), load_pipeline(), main(), make_breed_latents(), make_handler() (+7 more)

### Community 17 - "Community 17"
Cohesion: 0.13
Nodes (7): GPBanditOptimizer, OnePlusOneES, ndarray, Upgrades over the plain hill-climb, for when there's time: a (1+1) evolution str, (1+1)-ES with Rechenberg's 1/5 success rule for adaptive sigma., GP-BO with a UCB acquisition, maximized by random search over the box     (cheap, test_one_plus_one_es_adapts_sigma_on_success_streak()

### Community 18 - "Community 18"
Cohesion: 0.12
Nodes (15): aliases, components, hooks, lib, ui, utils, iconLibrary, rsc (+7 more)

### Community 19 - "Community 19"
Cohesion: 0.23
Nodes (11): ProceduralRenderer, CPU-only fallback renderer: a deterministic function of z, no GPU or model weigh, ProceduralPseudo3D, Rotates the flat sprite to fake a 3D viewing angle - no mesh, no GPU., _FakeImages, _FakeOpenAIClient, test_mirrored_quadrants_composes_full_canvas(), test_openai_image_generator_decodes_and_caches_prompt() (+3 more)

### Community 20 - "Community 20"
Cohesion: 0.31
Nodes (7): main(), Saves the session's first and final frame to data/processed/, for the     OFFLIN, run_local(), run_served(), _save_frame(), _SessionSnapshot, emotiv_credentials()

### Community 21 - "Community 21"
Cohesion: 0.26
Nodes (9): BreedTargetRewardSource, main(), _post_anchors(), _post_render(), ndarray, Scripted reward for the breed-weight server: reward a selected breed., _softmax_weights(), Interpolator (+1 more)

### Community 22 - "Community 22"
Cohesion: 0.28
Nodes (4): LatentMessage, Optimizer service -> Orchestrator. Next point in the low-dim search space., Feed one reward reading. Returns a LatentMessage once a full         window has, Feed one fully-formed Observation (mean + variance + effective N +         artif

### Community 23 - "Community 23"
Cohesion: 0.27
Nodes (5): Any, Image, ndarray, Client for running diffusion on a separate machine (the GPU/SSH server).  The lo, RemoteDiffusionClient

### Community 24 - "Community 24"
Cohesion: 0.20
Nodes (7): The Generator service: z in, rendered pyramid frame out.  Backend is picked by c, mirrored_quadrants(), Image, Image -> pseudo-3D pyramid quadrants.  Real-time text-to-3D (TripoSR) is the par, Wraps TripoSR for fast image-to-3D. Lazy-imported; requires the `tsr`     packag, Compose 4 copies of `image`, each facing outward from center, for a     tabletop, TripoSRConverter

### Community 25 - "Community 25"
Cohesion: 0.16
Nodes (12): LatentWalkGeneratorAdapter, main(), FrameMessage, Same start/end capture as run_demo.py, for the optional offline     DiffMorpher, Adapts LatentWalkRenderServer (built for an HTTP response: z in,     PNG bytes o, run(), _save_frame(), _SessionSnapshot (+4 more)

### Community 26 - "Community 26"
Cohesion: 0.21
Nodes (4): ControlMessage, Orchestrator -> any service. Session control (start/stop/reset/calibrate)., Hub server: Signal service clients push RewardMessages; display clients     rece, WebSocketOrchestrator

### Community 27 - "Community 27"
Cohesion: 0.25
Nodes (6): _encode_jpeg(), _encode_png(), FrameMessage, Image, ndarray, JPEG is 5-10x smaller than PNG and decodes faster in the browser.     Forces RGB

### Community 28 - "Community 28"
Cohesion: 0.22
Nodes (5): main(), OpenAIImageGenerator, Any, Image, OpenAI Image API renderer.  This backend turns the optimizer's selected anchor p

### Community 29 - "Community 29"
Cohesion: 0.24
Nodes (13): run_keyboard(), run_scripted(), _save_frame(), Config, GeneratorService, _build_algorithm(), OptimizerService, The Optimizer service: reward in, latent stream out. ~150 lines including the st (+5 more)

### Community 30 - "Community 30"
Cohesion: 0.29
Nodes (3): _FakeHTTPResponse, _FakeHTTPSession, test_remote_diffusion_sends_optimizer_state_and_caches_step()

### Community 31 - "Community 31"
Cohesion: 0.53
Nodes (5): applyHub(), applyRemote(), ApplyRequest, normalizePrompts(), POST()

### Community 32 - "Community 32"
Cohesion: 0.60
Nodes (4): GenerateRequest, normalizeAxes(), normalizePrompts(), POST()

### Community 33 - "Community 33"
Cohesion: 0.40
Nodes (3): ndarray, The last *accepted* latent - what should be on screen at rest., The candidate currently being shown, awaiting a verdict.

### Community 34 - "Community 34"
Cohesion: 0.67
Nodes (3): _dim(), Image, ndarray

### Community 50 - "NEURIM"
Cohesion: 0.14
Nodes (13): Architecture: four services, Build order (non-negotiable), Generation, Layout, Milestone: real-time morph via StreamDiffusion (SD-Turbo), NEURIM, Risk register, Setup (+5 more)

### Community 51 - "faa.py"
Cohesion: 0.24
Nodes (8): main(), calibrate_baseline(), Per-session baseline calibration: 30s of rest before anything else runs., Consume samples from `sample_iter` for `duration_s`, fitting the baseline., Frontal alpha asymmetry: the entire "reward" signal.  FAA = ln(alpha_power(right, Mean/std of raw FAA collected during the rest period, for z-scoring., RunningBaseline, test_running_baseline_z_score()

### Community 52 - "run_real_eeg_optimizer.py"
Cohesion: 0.27
Nodes (8): _cue(), main(), _post_anchors(), _post_render(), ndarray, Same start/end capture as run_demo.py's _SessionSnapshot, for the     optional o, _save_frame(), _SessionSnapshot

### Community 53 - "MockEEGSource"
Cohesion: 0.23
Nodes (7): MockEEGSource, Synthetic 14-channel EEG for development without hardware.      Alpha-band power, _build_source(), Regression tests for FAARewardSource (KNOWN_ISSUES #1).  The bug: once the FAA w, test_raw_faa_varies_across_reads_after_warmup(), test_reward_is_not_constant_across_reads(), test_samples_per_read_is_floored_at_one()

### Community 54 - "Plan: `scripts/run_stablediffusion.py` — real EEG driving the breed-mixture morph"
Cohesion: 0.18
Nodes (10): Build steps, in order, Config changes needed (`config.yaml` / CLI overrides), Machine / terminal roles - explicit, since this trips people up every time, New file: `scripts/run_stablediffusion.py`, Open questions for you before I start writing code, Plan: `scripts/run_stablediffusion.py` — real EEG driving the breed-mixture morph, `scripts/run_real_eeg_optimizer.py` - does it need to change?, What already exists - confirmed by reading the code, not assumed (+2 more)

### Community 55 - "Known issues / open work — handoff summary"
Cohesion: 0.20
Nodes (9): 1. [CRITICAL, unfixed] FAA reward freezes solid a few seconds into every real-EEG session, 2. [Design gap] No script bridges real/mock EEG to the WebSocketOrchestrator hub, 3. [RESOLVED] `/anchors` endpoint gap - and the server it applied to no longer exists, 4. [Security - user-side action needed, status unconfirmed] Leaked OpenAI API key, 5. [Recurring operational gotcha, now documented] Config is read once at process startup, 6. [Resolved, but fragile] Anchor-prompt bank design is the dominant lever for search-space quality, 7. [RESOLVED - architecture change] `run_streamdiffusion_server.py` no longer uses StreamDiffusion, Known issues / open work — handoff summary (+1 more)

### Community 56 - "test_faa_stream.py"
Cohesion: 0.36
Nodes (8): _cue_label(), main(), _no_reward_reason(), _pair_label(), _pair_value(), EEG data sources. All of them yield (timestamp, {channel_name: value}) samples., Wrap a sample iterator to yield in real time (for mock sources that     would ot, wall_clock_pace()

### Community 57 - "test_faa.py"
Cohesion: 0.38
Nodes (9): band_power(), Welch PSD power in `band` (Hz) for a single-channel 1D signal., _alpha_signal(), _clean_alpha_signal(), _push_window(), test_band_power_higher_for_larger_amplitude(), test_faa_reward_computer_ready_and_clipped(), test_faa_reward_computer_uses_configured_pair_weights() (+1 more)

### Community 58 - "FAARewardComputer"
Cohesion: 0.27
Nodes (6): FAARewardComputer, Any, Sliding-window weighted FAA -> baseline z-score -> clip to [-1, 1] = r(t)., _safe_float(), test_faa_eeg_features_include_all_configured_channels(), test_faa_reward_computer_pair_metrics()

### Community 59 - "latestPR.md"
Cohesion: 0.29
Nodes (6): Note on the base, Planned fixes, Results, Test plan, The extra fix testing forced, What & why

## Knowledge Gaps
- **114 isolated node(s):** `ApplyRequest`, `GenerateRequest`, `metadata`, `$schema`, `style` (+109 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Config` connect `Community 29` to `Community 1`, `Community 2`, `Community 9`, `Community 10`, `Community 13`, `Community 14`, `Community 16`, `faa.py`, `Community 20`, `Community 21`, `run_real_eeg_optimizer.py`, `test_faa_stream.py`, `Community 25`, `Community 26`, `Community 28`, `Community 24`?**
  _High betweenness centrality (0.102) - this node is a cross-community bridge._
- **Why does `OptimizerService` connect `Community 29` to `Community 1`, `Community 2`, `Community 33`, `Community 17`, `run_real_eeg_optimizer.py`, `Community 21`, `Community 22`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Why does `NoiseAwareLatentTuRBO` connect `Community 1` to `Community 4`, `Community 29`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `Config` (e.g. with `_SessionSnapshot` and `DiffusionRenderServer`) actually correct?**
  _`Config` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `FAARewardComputer` (e.g. with `FAARewardSource` and `SignalService`) actually correct?**
  _`FAARewardComputer` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `EmotivCortexSource` (e.g. with `_SessionSnapshot` and `_SessionSnapshot`) actually correct?**
  _`EmotivCortexSource` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `OptimizerService` (e.g. with `Config` and `LatentMessage`) actually correct?**
  _`OptimizerService` has 7 INFERRED edges - model-reasoned connections that need verification._