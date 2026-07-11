# Graph Report - NEURIM  (2026-07-12)

## Corpus Check
- 155 files · ~56,050 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1405 nodes · 2880 edges · 137 communities (116 shown, 21 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 226 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ae6d35a6`
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
- test_api_server.py
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29
- Interpolator
- Community 31
- Community 32
- Config
- orchestrator.py
- Community 35
- Community 36
- run_poodle_turbo_morph.py
- Community 39
- record_reward_trials.py
- Community 41
- WebSocketOrchestrator
- Community 43
- Community 46
- _FakeHTTPResponse
- fake_reward.py
- .current_z
- _validate_server_url
- load
- __init__.py
- from_json
- _detect_device
- __init__.py
- AGENTS.md
- __init__.py
- EEG Preference-Reward Redesign
- __init__.py
- _is_eeg_sensor_col
- scoring_start
- MockPreferenceEEGSource
- three
- .process_sample
- button.tsx
- tabs.tsx
- .__init__
- _FakeHTTPResponse
- OpenAIImageGenerator
- procedural.py
- RemoteDiffusionClient
- service.py
- TripoSRConverter
- OnePlusOneES
- PCAProjector
- StateMachine
- __init__.py
- manager.py
- MockEEGSource
- EmotivCortexSource
- BrainFlowLSLSource
- test_faa.py
- EEGFeatureExtractor
- LearnedPreferenceReward
- Preprocessor
- StimulusPresenter
- test_service.py

## God Nodes (most connected - your core abstractions)
1. `Config` - 64 edges
2. `SessionManager` - 44 edges
3. `FrameStore` - 44 edges
4. `FAARewardComputer` - 38 edges
5. `EmotivCortexSource` - 37 edges
6. `cn()` - 35 edges
7. `EEGConnectionManager` - 30 edges
8. `PromptCurationManifest` - 29 edges
9. `NoiseAwareLatentTuRBO` - 29 edges
10. `OptimizerRenderLoop` - 29 edges

## Surprising Connections (you probably didn't know these)
- `_Writer` --uses--> `Config`  [INFERRED]
  scripts/record_reward_trials.py → src/common/config.py
- `_Writer` --uses--> `EmotivCortexSource`  [INFERRED]
  scripts/record_reward_trials.py → src/signal_service/eeg_sources.py
- `_Writer` --uses--> `EEGFeatureExtractor`  [INFERRED]
  scripts/record_reward_trials.py → src/signal_service/learned_reward.py
- `_Writer` --uses--> `FeatureBaseline`  [INFERRED]
  scripts/record_reward_trials.py → src/signal_service/learned_reward.py
- `_Writer` --uses--> `MockPreferenceEEGSource`  [INFERRED]
  scripts/record_reward_trials.py → src/signal_service/mock_preference.py

## Import Cycles
- None detected.

## Communities (137 total, 21 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.14
Nodes (26): _as_feature_tensor(), _build_presenter(), build_trials(), capture_window(), _clip_embed(), embed_images(), fit_session_baseline(), _images_in() (+18 more)

### Community 1 - "Community 1"
Cohesion: 0.13
Nodes (20): Noise-Aware Latent TuRBO: a trust-region Bayesian optimizer built for the noisy,, effective_sample_size(), Observation, ndarray, Observation model for the optimizer: a presentation window is NOT reduced to a s, One presentation window's reward estimate and its uncertainty., ESS accounting for autocorrelation of overlapping-window FAA samples.      Uses, Turn a window of FAA reward samples into an Observation.      artifact_fraction (+12 more)

### Community 2 - "Community 2"
Cohesion: 0.13
Nodes (11): PresentationSchedule, Interpolation fraction for the morph: reaches 1.0 by the end of the         tran, Return a list of human-readable warnings (empty if the schedule is sound)., Collects only the scoring-interval reward readings for one candidate,     then e, Start a fresh candidate (call when a new latent is presented)., ScoringGate, test_morph_alpha_completes_by_transition_then_holds(), test_schedule_phases() (+3 more)

### Community 3 - "Community 3"
Cohesion: 0.14
Nodes (18): _client(), FakeCurationService, FakeDiffusionClient, FakeRewardSource, _manifest(), test_duplicate_start_returns_conflict(), test_eeg_status_and_retry(), test_health() (+10 more)

### Community 4 - "Community 4"
Cohesion: 0.13
Nodes (12): FAARewardComputer, PairFAAMetrics, Any, ndarray, Sliding-window weighted FAA -> baseline z-score -> clip to [-1, 1] = r(t)., Pair-level alpha power and raw FAA values for diagnostics., Compact EEG visualization payload for the frontend.          Uses alpha-band pow, Baseline-normalized r(t), or None if the buffer isn't full yet. (+4 more)

### Community 5 - "Community 5"
Cohesion: 0.06
Nodes (40): DiffusionClient, Any, ndarray, HTTP client for a private manifest-driven diffusion server., FrameStore, Path, Live-frame and session snapshot storage., The finalized image the frontend displays (GET /api/target-frame). (+32 more)

### Community 6 - "Community 6"
Cohesion: 0.21
Nodes (11): ProceduralRenderer, CPU-only fallback renderer: a deterministic function of z, no GPU or model weigh, ProceduralPseudo3D, Rotates the flat sprite to fake a 3D viewing angle - no mesh, no GPU., _FakeHTTPResponse, _FakeHTTPSession, test_mirrored_quadrants_composes_full_canvas(), test_procedural_renderer_changes_with_z() (+3 more)

### Community 7 - "Community 7"
Cohesion: 0.17
Nodes (11): BaseModel, FastAPI, create_app(), FastAPI application factory for the local frontend bridge., main(), CLI for the local frontend API bridge., Local frontend API bridge., Request models for the local frontend API. (+3 more)

### Community 8 - "Community 8"
Cohesion: 0.10
Nodes (23): manifest_metadata(), Any, build_parser(), create_renderer(), main(), ArgumentParser, CLI composition root for the manifest-driven diffusion server., DiffusionServer (+15 more)

### Community 9 - "Community 9"
Cohesion: 0.17
Nodes (28): device, dtype, add_hud(), blend_noise_latents(), blend_prompt_embeds(), cosine_ease(), encode_breed_prompts(), interpolate_z() (+20 more)

### Community 10 - "Community 10"
Cohesion: 0.11
Nodes (19): clsx, dependencies, clsx, lucide-react, next, next-themes, openai, @radix-ui/react-slot (+11 more)

### Community 11 - "Community 11"
Cohesion: 0.09
Nodes (45): load_prompt_session_manifest(), Path, Helpers for manifest-backed anchor sessions used by the generalized server., _clean_string_list(), curate_prompt_manifest(), _extract_response_text(), format_manifest_summary(), _load_default_client() (+37 more)

### Community 12 - "Community 12"
Cohesion: 0.16
Nodes (14): LatentMorpher, morph_path(), ndarray, Real-time latent morphing between a jumpy stream of target latents and a smooth,, max_step:  max Euclidean distance z may move per step() call (per, Advance toward `target` by at most max_step; return the new z., Lower bound on frames to reach `target` at the current max_step -         useful, Fixed-endpoint linear path of `n` intermediate latents (inclusive of     z_new, (+6 more)

### Community 13 - "Community 13"
Cohesion: 0.08
Nodes (23): 1. Why the redesign, 2. Architecture, 3. Files added / changed, 4.0 Generate A/B candidate images (per target), 4.1 Record calibration trials (mock, offline), 4.2 Train + validate (the scientific gate), 4.3 Run the closed loop (mock EEG, headless-friendly reward core), 4.4 Tests (+15 more)

### Community 14 - "Community 14"
Cohesion: 0.06
Nodes (32): datetime, Protocol, Signal service -> Orchestrator. One scalar reward reading., RewardMessage, Real FAA needs 30s of rest to fit the baseline; fake reward sources         (key, EEGConnectionManager, _iso(), _log() (+24 more)

### Community 15 - "Community 15"
Cohesion: 0.36
Nodes (8): backendError(), BackendSession, cleanUrl(), POST(), requestBoolean(), requestNumber(), requestString(), SessionIntentRequest

### Community 16 - "Community 16"
Cohesion: 0.22
Nodes (11): ImageFinalizer, Any, OpenAI image finalize pass.  The anchor-morph diffusion server produces its last, Return a cleaned PNG for ``png_bytes`` (the last morphed frame).          Raises, FakeImages, FakeOpenAI, _png(), _png_b64() (+3 more)

### Community 17 - "Community 17"
Cohesion: 0.16
Nodes (9): main(), DiffusionGenerator, ndarray, SDXL-Turbo / LCM wrapper: latent (well, prompt-embedding) in, frame out in ~100-, Reconstruct (prompt_embeds, pooled_prompt_embeds) from a flat vector         pro, img2img pass against the previous frame, for smoother morphing         between o, A fixed-seed torch.Generator so nearby embeddings render to nearby         image, Straight text -> image, bypassing the embedding/projector path. (+1 more)

### Community 18 - "Community 18"
Cohesion: 0.33
Nodes (9): _make_reward(), _quiet(), Headless tests for the pairwise preference reward + optimizer core.  Exercises t, A near-target candidate must out-reward a far-from-target one (deterministic)., _run_loop(), _softmax(), _target_pref(), test_reward_gradient_points_toward_target() (+1 more)

### Community 19 - "Community 19"
Cohesion: 0.14
Nodes (16): Landing(), NeurimApp(), NeurimSession(), SessionView(), usePngFrame(), SessionPhase, useSession(), epocPositions (+8 more)

### Community 20 - "Community 20"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: retrieve context, Source Nodes

### Community 21 - "Community 21"
Cohesion: 0.29
Nodes (5): hanken, metadata, newsreader, plexMono, ThemeProvider()

### Community 22 - "Community 22"
Cohesion: 0.13
Nodes (16): ThemeToggle(), TopBar(), Badge(), BadgeProps, badgeVariants, Button, ButtonProps, buttonVariants (+8 more)

### Community 23 - "test_api_server.py"
Cohesion: 0.36
Nodes (3): ProcessingState(), PromptBubble(), SteerInput()

### Community 24 - "Community 24"
Cohesion: 0.10
Nodes (21): eslint, eslint-config-next, devDependencies, eslint, eslint-config-next, tailwindcss, @tailwindcss/postcss, @types/node (+13 more)

### Community 25 - "Community 25"
Cohesion: 0.29
Nodes (4): Event, Any, The OpenAI finalize pass, or None to skip it. A missing API key or an         un, SessionManager

### Community 26 - "Community 26"
Cohesion: 0.29
Nodes (6): BrainActivity3D(), channelNames, ElectrodeNodes(), fallbackPositions, normalizeChannels(), EEGFeatures

### Community 27 - "Community 27"
Cohesion: 0.12
Nodes (15): aliases, components, hooks, lib, ui, utils, iconLibrary, rsc (+7 more)

### Community 28 - "Community 28"
Cohesion: 0.22
Nodes (3): MockPreferenceEEGSource, Synthetic EEG carrying a controllable, decodable *preference* signal.  MockEEGSo, 14-channel synthetic EEG with a tunable-SNR preference signal.      SNR is gover

### Community 30 - "Interpolator"
Cohesion: 0.20
Nodes (17): OptimizerConfig, StateMachineConfig, CALIBRATE -> EXPLORE -> REFINE -> SETTLE, with a RECOVER escape hatch.  CALIBRAT, test_optimizer_converges_toward_hidden_target(), test_observe_observation_takes_one_step(), _sm(), test_explore_moves_to_refine_on_climbing_trend(), test_min_steps_before_settle_prevents_immediate_lock() (+9 more)

### Community 31 - "Community 31"
Cohesion: 0.23
Nodes (6): _erf(), NoiseAwareLatentTuRBO, ndarray, Per-dim GP length scales (ARD), for shaping the trust region., TuRBO box: side length `self.length` scaled per-dim by the ARD length         sc, math.erf on a scalar (avoids importing scipy just for the normal CDF).

### Community 32 - "Community 32"
Cohesion: 0.22
Nodes (8): name, private, scripts, build, dev, lint, start, version

### Community 33 - "Config"
Cohesion: 0.53
Nodes (5): applyHub(), applyRemote(), ApplyRequest, normalizePrompts(), POST()

### Community 34 - "orchestrator.py"
Cohesion: 0.17
Nodes (8): BrainOrb(), DisconnectedView(), EegGate(), EegStatusPill(), EegStatusHook, isEegStatus(), useEegStatus(), EegStatus

### Community 35 - "Community 35"
Cohesion: 0.33
Nodes (5): Card, CardContent, CardDescription, CardHeader, CardTitle

### Community 36 - "Community 36"
Cohesion: 0.60
Nodes (5): encode(), generate_for_target(), load_targets(), main(), Path

### Community 37 - "run_poodle_turbo_morph.py"
Cohesion: 0.60
Nodes (4): GenerateRequest, normalizeAxes(), normalizePrompts(), POST()

### Community 40 - "record_reward_trials.py"
Cohesion: 0.50
Nodes (3): Alert, AlertDescription, AlertTitle

### Community 41 - "Community 41"
Cohesion: 0.22
Nodes (11): HeroCandidate(), FrameStream, useFrameStream(), NeurimSession, decodeFrameSrc(), EegState, FrameMessage, FrameState (+3 more)

### Community 42 - "WebSocketOrchestrator"
Cohesion: 0.17
Nodes (5): FakeEEGManager, ReadyEEGManager, RecordingFinalizer, test_retry_finalization_refines_saved_raw_frame(), test_retry_finalization_requires_completed_raw_frame()

### Community 43 - "Community 43"
Cohesion: 0.27
Nodes (6): ApproachMeter(), PhaseChips(), RewardReadout(), BrainActivity3D, SignalRail(), Skeleton()

### Community 67 - "_is_eeg_sensor_col"
Cohesion: 0.07
Nodes (29): compilerOptions, allowJs, esModuleInterop, incremental, isolatedModules, jsx, lib, module (+21 more)

### Community 87 - "MockPreferenceEEGSource"
Cohesion: 0.29
Nodes (8): Namespace, _build_preference_reward(), build_reward_source(), LearnedEEGReward, Real EEG reward scored by a trained sklearn reward model., Pairwise-preference EEG reward (real headset or --mock-eeg)., build_preprocessor(), Construct a Preprocessor from a Config, or None if disabled.

### Community 99 - "three"
Cohesion: 0.19
Nodes (12): Calibrator, SourceFactory, EEGConfig, FAAConfig, GeneratorConfig, LoopConfig, PreprocessingConfig, PresentationConfig (+4 more)

### Community 103 - "button.tsx"
Cohesion: 0.16
Nodes (9): LatentMessage, Optimizer service -> Orchestrator. Next point in the low-dim search space., OptimizerService, ndarray, The last *accepted* latent - what should be on screen at rest., The candidate currently being shown, awaiting a verdict., Feed one reward reading. Returns a LatentMessage once a full         window has, Feed one fully-formed Observation (mean + variance + effective N +         artif (+1 more)

### Community 105 - ".__init__"
Cohesion: 0.11
Nodes (11): ControlMessage, FrameMessage, Wire format for the websocket messages passed between services.  Signal -> Orche, Generator service -> Orchestrator. One rendered frame, ready to display.      Ex, Orchestrator -> any service. Session control (start/stop/reset/calibrate)., Interpolator, -- Morphing Process (animation)     Linear interpolation between the last two ac, LocalOrchestrator (+3 more)

### Community 115 - "_FakeHTTPResponse"
Cohesion: 0.50
Nodes (4): ndarray, Fraction of samples in a cleaned window flagged as blink or EMG.          Blink:, Median/MAD z-score; robust to the very spikes we want to flag., _robust_z()

### Community 118 - "OpenAIImageGenerator"
Cohesion: 0.18
Nodes (7): OpenAIImageGenerator, Any, Image, OpenAI Image API renderer.  This backend turns the optimizer's selected anchor p, _FakeImages, _FakeOpenAIClient, test_openai_image_generator_decodes_and_caches_prompt()

### Community 119 - "procedural.py"
Cohesion: 0.67
Nodes (3): _dim(), Image, ndarray

### Community 122 - "RemoteDiffusionClient"
Cohesion: 0.27
Nodes (5): Any, Image, ndarray, Client for running diffusion on a separate machine (the GPU/SSH server).  The lo, RemoteDiffusionClient

### Community 123 - "service.py"
Cohesion: 0.15
Nodes (11): _encode_jpeg(), _encode_png(), GeneratorService, FrameMessage, Image, ndarray, The Generator service: z in, rendered pyramid frame out.  Backend is picked by c, JPEG is 5-10x smaller than PNG and decodes faster in the browser.     Forces RGB (+3 more)

### Community 127 - "TripoSRConverter"
Cohesion: 0.33
Nodes (3): Image, Wraps TripoSR for fast image-to-3D. Lazy-imported; requires the `tsr`     packag, TripoSRConverter

### Community 130 - "OnePlusOneES"
Cohesion: 0.08
Nodes (20): Config, GPBanditOptimizer, OnePlusOneES, ndarray, Upgrades over the plain hill-climb, for when there's time: a (1+1) evolution str, (1+1)-ES with Rechenberg's 1/5 success rule for adaptive sigma., GP-BO with a UCB acquisition, maximized by random search over the box     (cheap, MomentumHillClimb (+12 more)

### Community 144 - "PCAProjector"
Cohesion: 0.15
Nodes (9): AnchorInterpolationProjector, PCAProjector, ndarray, Reduce the search space from the raw latent/embedding dim down to 8-16 dims, per, Low-dim search vector <-> full embedding, via PCA fit on a prompt bank., embeddings: [n_prompts, embed_dim], z is a weight vector over `anchor_embeddings`; softmax-normalized so     the pro, test_anchor_projector_stays_in_convex_hull() (+1 more)

### Community 154 - "StateMachine"
Cohesion: 0.25
Nodes (4): Interpolate step size: wide in EXPLORE, shrinking in REFINE as the         rewar, Call once per optimizer step with the accepted/estimated reward and         the, StateMachine, State

### Community 165 - "manager.py"
Cohesion: 0.17
Nodes (6): DiffusionClientFactory, FinalizerFactory, FrameStoreFactory, ProcessLogStore, Path, _slug()

### Community 181 - "MockEEGSource"
Cohesion: 0.15
Nodes (15): main(), _cue_label(), main(), _no_reward_reason(), _pair_label(), _pair_value(), emotiv_credentials(), calibrate_baseline() (+7 more)

### Community 183 - "EmotivCortexSource"
Cohesion: 0.14
Nodes (9): EmotivCortexSource, Any, Return Cortex EEG column labels from a subscribe response., EMOTIV Cortex API client for the EPOC X headset (WebSocket JSON-RPC).      Flow:, FakeWebSocket, test_emotiv_extracts_eeg_cols_from_subscribe_result(), test_emotiv_formats_unknown_api_error_with_raw_payload(), test_emotiv_formats_unpublished_app_error_with_owner_hint() (+1 more)

### Community 188 - "test_faa.py"
Cohesion: 0.22
Nodes (13): band_power(), Frontal alpha asymmetry: the entire "reward" signal.  FAA = ln(alpha_power(right, Mean/std of raw FAA collected during the rest period, for z-scoring., Welch PSD power in `band` (Hz) for a single-channel 1D signal., RunningBaseline, _alpha_signal(), _clean_alpha_signal(), _push_window() (+5 more)

### Community 198 - "EEGFeatureExtractor"
Cohesion: 0.06
Nodes (44): augment_antisymmetric(), augment_jitter(), build_ensemble(), faa_feature_mask(), leave_session_out(), _load_one(), load_pairwise(), main() (+36 more)

### Community 209 - "LearnedPreferenceReward"
Cohesion: 0.33
Nodes (4): LearnedPreferenceReward, ndarray, One preference reward sample from the extractor's current window., Capture the fixed reference window A from a known reference image z.          Ca

### Community 212 - "Preprocessor"
Cohesion: 0.19
Nodes (11): PreprocessedSource, Preprocessor, Streaming EEG signal conditioning: the stage that was missing entirely.  Raw EPO, Wrap an EEGSource so downstream consumers get conditioned samples.      Same (ti, Stateful causal conditioning for streaming multi-channel EEG., _alpha(), ndarray, _stream_window() (+3 more)

### Community 230 - "StimulusPresenter"
Cohesion: 0.07
Nodes (18): Exception, AbortPresentation, ndarray, Path, OpenCV stimulus presentation for real EEG calibration sessions.  The pairwise re, Morph A -> B by alpha blend (the calibration analogue of the live latent, Raised when the operator presses q/ESC to stop the session early., One repaint - call this inside the EEG capture loop so the stable         image (+10 more)

### Community 236 - "test_service.py"
Cohesion: 0.53
Nodes (5): _build_source(), Regression tests for FAARewardSource (KNOWN_ISSUES #1).  The bug: once the FAA w, test_raw_faa_varies_across_reads_after_warmup(), test_reward_is_not_constant_across_reads(), test_samples_per_read_is_floored_at_one()

## Knowledge Gaps
- **128 isolated node(s):** `ApplyRequest`, `GenerateRequest`, `SessionIntentRequest`, `BackendSession`, `hanken` (+123 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **21 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Config` connect `OnePlusOneES` to `Community 0`, `Community 2`, `Community 3`, `Community 5`, `Community 8`, `Community 9`, `Community 11`, `Community 14`, `Community 25`, `Interpolator`, `manager.py`, `Community 39`, `WebSocketOrchestrator`, `MockEEGSource`, `MockPreferenceEEGSource`, `three`, `.process_sample`, `button.tsx`, `.__init__`, `service.py`?**
  _High betweenness centrality (0.218) - this node is a cross-community bridge._
- **Why does `DiffusionGenerator` connect `Community 17` to `.__init__`, `service.py`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Why does `FAARewardComputer` connect `Community 4` to `.process_sample`, `Community 39`, `Community 9`, `test_service.py`, `Community 14`, `MockEEGSource`, `MockPreferenceEEGSource`, `test_faa.py`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Are the 28 inferred relationships involving `Config` (e.g. with `_Writer` and `FakeFAAReward`) actually correct?**
  _`Config` has 28 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `SessionManager` (e.g. with `Config` and `ImageFinalizer`) actually correct?**
  _`SessionManager` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `FrameStore` (e.g. with `ProcessLogStore` and `SessionManager`) actually correct?**
  _`FrameStore` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `FAARewardComputer` (e.g. with `FakeFAAReward` and `LearnedEEGReward`) actually correct?**
  _`FAARewardComputer` has 5 INFERRED edges - model-reasoned connections that need verification._