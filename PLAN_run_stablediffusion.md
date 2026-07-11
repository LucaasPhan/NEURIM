# Plan: `scripts/run_stablediffusion.py` — real EEG driving the breed-mixture morph

## Why this replaces `run_streamdiffusion_server.py`'s mechanism

`run_streamdiffusion_server.py` (current) works, but its parameterization -
`z` → `PCAProjector.to_embedding(z)` fit from `config.yaml`'s `anchor_prompts` -
is a different, more fragile mechanism than what `run_poodle_turbo_morph.py`
and `continuous_dog_latent_morph.py` actually do, and it's shown real problems
(the `"Salmon"` stray-entry bug, `n_anchors - 1` rank limits, PCA directions
that don't cleanly correspond to one attribute each). The two reference
scripts don't use PCA at all - `run_poodle_turbo_morph.py` in particular
already proves a cleaner mechanism that also happens to already have a
production-quality noise-aware optimizer built for it. This plan builds the
missing piece: a real HTTP server for that mechanism.

## What already exists - confirmed by reading the code, not assumed

- **`NoiseAwareLatentTuRBO`** (`src/optimizer/latent_turbo.py`) - a trust-region
  Bayesian optimizer built specifically for noisy, delay-prone FAA reward
  (heteroscedastic GP, Thompson sampling, probabilistic success test, motion-
  limited candidates, checkpoint recovery). Registered as
  `config.optimizer.algorithm: "latent_turbo"` in `OptimizerService`'s
  `_ALGORITHMS` dict (`src/optimizer/service.py:25`).
- **`Observation`/`window_statistics`** (`src/optimizer/observation.py`) - turns
  a window of raw FAA samples into `(reward_mean, reward_variance,
  effective_sample_count, artifact_fraction)`, correcting for the
  autocorrelation of overlapping 2s FAA windows. `NoiseAwareLatentTuRBO.observe()`
  consumes this directly (`wants_observation = True` tells `OptimizerService` to
  hand it full `Observation`s, not a bare scalar).
- **`PresentationSchedule`/`ScoringGate`** (`src/signal_service/presentation.py`)
  - produces exactly these `Observation`s from real EEG at the right cadence
  (transition → stabilize → score), already wired into `LocalOrchestrator`.
- **The rendering mechanism, proven standalone** in `run_poodle_turbo_morph.py`:
  `softmax_weights(z)` → simultaneously blend **prompt embeddings**
  (`blend_prompt_embeds`) and **per-breed noise latents**
  (`blend_noise_latents`, norm-renormalized so the blend still looks like valid
  Gaussian noise) → one plain-diffusers `pipe(prompt_embeds=..., latents=...)`
  call, fresh every frame. Already optimizer-driven in that file (via
  `NoiseAwareLatentTuRBO` + a *fake* FAA reward) - so the mechanism-to-optimizer
  wiring is already proven; only the reward source needs to become real.

## What's missing - what this plan builds

An HTTP server implementing that exact rendering mechanism, so a real-EEG
client can drive it over the network the same way `run_real_eeg_optimizer.py`
already drives `run_streamdiffusion_server.py`.

### New file: `scripts/run_stablediffusion.py`

Runs on the **GPU server**. Responsibilities:

1. **CLI, modeled on `run_poodle_turbo_morph.py`/`continuous_dog_latent_morph.py`**:
   `--model` (default `stabilityai/sd-turbo`), `--breeds` (`nargs="+"`, replaces
   `config.yaml`'s `anchor_prompts` entirely - no PCA-fragility, no YAML hand-
   authoring risk), `--prompt-template` (defaults to the same
   `PROMPT_TEMPLATE` style, one `{breed}` placeholder), `--size`, `--steps`,
   `--seed`, `--host`, `--port`.
2. **At startup**: load a plain `diffusers.StableDiffusionPipeline` (reuse
   `load_pipeline`-style code from the current `run_streamdiffusion_server.py`),
   encode all `--breeds` prompts as **one batch** via `pipe.encode_prompt()`
   (matches `encode_breed_prompts()`), and generate **one fixed random latent
   per breed** (matches `make_breed_latents()`) - each breed gets its own seed-
   derived noise tensor, kept fixed for the whole session.
3. **`POST /render`**: same wire contract as today (`{"z": [...], "frame_size":
   ...}` → PNG bytes) - but `z` is now a **`len(breeds)`-dimensional breed-
   weight vector**, not an 8-dim PCA-projected point. Per request:
   `softmax_weights(z)` → `blend_prompt_embeds()` + `blend_noise_latents()` →
   one fresh `pipe(...)` call → PNG. No PCAProjector anywhere in this file.
4. **No `/anchors` endpoint** - there's no PCA projector to re-fit. If you want
   to change the breed list, restart with different `--breeds` (same
   restart-required rule as everything else in this codebase).

### Config changes needed (`config.yaml` / CLI overrides)

- **`optimizer.search_dims` must equal `len(breeds)`** - not the old PCA-space
  8. With this mechanism, every dimension of `z` directly *is* one breed's
  weight; there's no PCA compression, so dimensionality is exact, not
  "up to n_anchors - 1 directions."
- **`optimizer.algorithm: "latent_turbo"`** - selects `NoiseAwareLatentTuRBO`
  instead of hill-climb/ES/GP-BO.
- `generator.anchor_prompts` becomes unused for this path (no server reads it)
  - fine to leave in `config.yaml` for the other backends
  (`run_diffusion_server.py`, the old `run_streamdiffusion_server.py`), just
  irrelevant here.

### `scripts/run_real_eeg_optimizer.py` - does it need to change?

**No transport/code changes** - it already just does
`optimizer.observe_reward(msg.r)` / POSTs `z` to `/render` / saves frames to
`data/processed/live_frame.png`. `Interpolator` interpolates raw `z` vectors
regardless of what they mean downstream, so it's agnostic to "PCA space" vs
"breed-weight space." It only needs to be **run with the right config**:
`--server-url http://localhost:8766 --algorithm latent_turbo`, and
`config.optimizer.search_dims` set to `len(breeds)` before starting.

One caveat carried over from earlier in this project: this script still calls
`optimizer.observe_reward()` directly, not `optimizer.observe_observation()` -
so it does **not** yet get `PresentationSchedule`/`ScoringGate`'s
transition/stabilize/score gating, and therefore doesn't hand
`NoiseAwareLatentTuRBO` real `Observation`s with proper variance/artifact
info - it'll fall back through whatever scalar-reward path `OptimizerService`
has for non-`wants_observation`-aware callers. **This should probably be fixed
together with adopting `latent_turbo`**, since the optimizer is specifically
built to exploit `Observation`'s uncertainty - feeding it bare scalars wastes
most of what makes it noise-*aware*. Flagging as a likely immediate follow-up,
not doing it in this pass unless you want it folded in now.

## Machine / terminal roles - explicit, since this trips people up every time

```
┌─────────────────────────┐         ┌──────────────────────────────┐
│   YOUR LOCAL MACHINE     │         │        GPU SERVER             │
│                          │         │                                │
│  EMOTIV Launcher + │
│  headset (Cortex :6868)  │         │  Terminal A:                  │
│                          │         │  run_stablediffusion.py       │
│  Terminal B:             │  HTTP   │  (new file, this plan)        │
│  run_real_eeg_optimizer  │◄───────►│  listens :8766                │
│    --server-url          │  :8766  │                                │
│    http://localhost:8766 │ tunnel  │                                │
│    --algorithm           │         │                                │
│    latent_turbo          │         │                                │
│                          │         │                                │
│  writes                  │         │                                │
│  data/processed/         │         │                                │
│    live_frame.png ───────┼────┐    │                                │
│                          │    │    │                                │
│  Browser:                │    │    │                                │
│  frontend/live_view.html │◄───┘    │                                │
│  (opens the file above   │         │                                │
│   directly - same machine)│         │                                │
└─────────────────────────┘         └──────────────────────────────┘
```

**Recommended layout** (matches the split architecture already established
earlier in this project): run `run_real_eeg_optimizer.py` **locally**, near
the headset - Cortex (`wss://localhost:6868`) then just works natively, no
reverse tunnel needed. Only the render calls cross the network, via the
existing `LocalForward 8766 localhost:8766` SSH tunnel. Frames land in your
**local** `data/processed/live_frame.png`, so `frontend/live_view.html`
opened locally just works with zero extra port-forwarding for viewing.

| # | Where | Command | Role |
|---|---|---|---|
| 1 | GPU server, Terminal A | `python scripts/run_stablediffusion.py --breeds "Golden Retriever" "German Shepherd" ... --port 8766` | Render server: breed-mixture morph, HTTP `/render` |
| 2 | Local machine | `ssh vishc-server-1-vinuni-pinggy` (opens `LocalForward 8766`) | Tunnel so local `localhost:8766` reaches the GPU server |
| 3 | Local machine, Terminal B | `python scripts/run_real_eeg_optimizer.py --server-url http://localhost:8766 --algorithm latent_turbo` | Real EEG → FAA → `NoiseAwareLatentTuRBO` → POST z → save `live_frame.png` |
| 4 | Local machine, browser | open `frontend/live_view.html` | Polls local `live_frame.png`, displays the live morph |

**Alternative layout** (everything on the server, reverse tunnel for EEG only)
is still valid if you'd rather keep the GPU box as the single point of
control - same as documented for `run_streamdiffusion_demo.py` earlier - but
loses the "viewing just works, no extra tunnel" property, since frames would
then live on the server and `live_view.html` would need its own port-forwarded
HTTP server to view remotely.

## Build steps, in order

1. Write `scripts/run_stablediffusion.py`: `load_pipeline()` (reusable from
   current `run_streamdiffusion_server.py`), `encode_breed_prompts()` +
   `make_breed_latents()` + `softmax_weights()` + `blend_prompt_embeds()` +
   `blend_noise_latents()` (lifted near-verbatim from `run_poodle_turbo_morph.py`
   - these are pure functions, already correct, already tested standalone),
   `render_png()` wiring them together, `POST /render` handler.
2. Update `config.yaml`: set `optimizer.search_dims` to `len(breeds)` for
   whatever breed list you choose, `optimizer.algorithm: "latent_turbo"`.
3. Sanity-check standalone first (mirroring `test_streamdiffusion.py`'s
   discipline): a small script or REPL check that `POST /render` with a
   uniform `z` (all-zero, i.e. uniform breed mixture after softmax) produces a
   coherent blended image, before wiring real EEG in at all.
4. Run the 4-step table above.
5. Decide on the `observe_observation()` follow-up (see caveat above) - test
   first with the simpler `observe_reward()` path, since that's zero
   additional work, then decide if the `Observation`-aware path is worth
   wiring into `run_real_eeg_optimizer.py` based on how it feels.

## Open questions for you before I start writing code

1. **Breed/attribute list**: reuse dog breeds (matching the reference scripts
   exactly, easiest to validate against their proven behavior), or switch to
   your current cat-color theme (same mechanism, just a different
   `--prompt-template`/`--breeds` list)?
2. **Fold in the `observe_observation()`/`ScoringGate` fix now**, alongside
   building this server, or ship the simpler `observe_reward()` path first and
   treat the `Observation`-aware wiring as an immediate follow-up once this is
   proven working end to end?
