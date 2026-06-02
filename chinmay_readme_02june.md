# V-LEAD — No-Docker SAC Training Bring-Up (2 June 2026)

> **Purpose.** This is a blow-by-blow replication guide for getting **SAC RL fine-tuning**
> running end-to-end in the local (no-Docker) Python venv on a fresh machine. It captures
> every dependency conflict, missing package, runtime error, and the exact fix for each,
> in the order we hit them. If you are setting this up again, you can follow the
> **"Quick path"** at the top, or read the **"Full chronology"** to understand *why* each
> step is needed.
>
> **Machine this was done on:** Ubuntu 24.04, RTX 4090 (24 GB), driver 595, CUDA toolkit
> 12.9 at `/usr/local/cuda`, Python 3.10.20, PyTorch 2.7.0+cu128, venv at `<repo>/.venv`.
> Repo root: `/home/rohang73/Desktop/cs224r_yashadityachinmay/V-LEAD`. Branch: `chinmay/asl_setup`.
>
> **Context vs. the maven setup (`yash_readme.md`).** The original setup notes target the
> "maven" server (RTX 5090, 32 GB). Several things that "just work" inside the FiGS Docker
> image, or on a 32 GB GPU, do **not** work out of the box in the no-Docker venv on a 24 GB
> card. This doc fills those gaps.

---

## 0. TL;DR — what the working state looks like

After everything below, this command trains SAC end-to-end (collects FiGS rollouts with
live Gaussian-splat rendering + MPC, runs SAC updates, saves checkpoints):

```bash
cd /home/rohang73/Desktop/cs224r_yashadityachinmay/V-LEAD/nav_policy
source ../.venv/bin/activate
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export ACADOS_SOURCE_DIR=/home/rohang73/Desktop/cs224r_yashadityachinmay/V-LEAD/FiGS-Standalone/acados
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/train_rl.py --config configs/train_rl_flightroom_sac_24gb.yaml
```

Smoke-test it first (1 iteration, 1 rollout — collects an episode, does one SAC update):

```bash
python scripts/train_rl.py --config configs/train_rl_flightroom_sac_24gb.yaml \
  --n-iterations 1 --rollouts-per-iteration 1
```

> **Tip:** put the four `export` lines into `.venv/bin/activate` so you never forget them.
> Without them you will hit (respectively) gsplat/tcnn JIT-compile failures, ACADOS import
> errors, ACADOS shared-lib load errors, and possibly CUDA OOM/fragmentation.

---

## 1. Quick path (do these in order)

1. **Fix `requirements.txt` version conflicts** (§3.1) and `pip install -r requirements.txt`.
2. **Install the FiGS editable packages** that the no-Docker flow needs but doesn't auto-install:
   - `pip install -e FiGS-Standalone/gemsplat` (needs `setuptools<81` + a torchtyping shim — §3.4).
   - Build `tiny-cuda-nn` for your GPU arch (§3.5).
   - Build ACADOS + `pip install -e FiGS-Standalone/acados/interfaces/acados_template` + fetch `t_renderer` (§3.6).
3. **Add the `torch.load` weights-only monkey-patch** to `nav_policy/src/nav_policy/rl/train_rl.py` (§3.7).
4. **Download the minimal SAC data set** from the Google Drive via rclone (§2).
5. **Use the 24 GB config** `configs/train_rl_flightroom_sac_24gb.yaml` (§3.8), set the env vars (§0), run.

---

## 2. Data — where it lives and how to download it

### 2.1 The data is NOT in git or any downloadable repo

The training data, trained Gaussian-splat scenes, and checkpoints are **not** version-controlled.
The repo docs reference cluster/teammate paths that don't exist on a fresh box:
- `yash_readme.md`: `/home/yashr/CS231N/checkpoints/...`, `/home/yashr/CS231N/FiGS_data/...` (maven server)
- `V-LEAD_instructions.md`: `/data/kothari1/singer_figs_data` (legacy drive), `/project/kothari1/...` (project drive)

None of these are reachable from a new machine. The data was provided as a **Google Drive shared
folder**: `https://drive.google.com/drive/folders/14BB9ajONol2qFXe2o-25dkEJ_mrgUux9`
(folder id `14BB9ajONol2qFXe2o-25dkEJ_mrgUux9`).

### 2.2 Drive folder layout (top level)

```
3dgs/                  # Gaussian-splat workspaces (capture videos, COLMAP, trained outputs)
FiGS_data/             # SINGER rollout data (vlead_flightroom + backroom/packardpark)
GCP_VM_Data/           # a second 3dgs workspace
Rahul_Best_Weights/    # BC checkpoints (bc_best_balanced_dagger_r12_new.pt, bc_best_old.pt)
aditya_weights/        # sac_v6_2236.zip etc.
```

The full archive is large (multiple 3DGS workspaces + thousands of source training images +
rollout videos). **You do not need most of it for SAC.**

### 2.3 What SAC actually needs (minimal set, ~1 GB)

SAC **warm-starts from an existing BC checkpoint** (no dataset rebuild / BC retraining needed).
RL rollouts render the gsplat scene **live** and only read `setup_from` trajectories for the
initial drone state — the recorded RGB/depth videos and `imgdata` are for BC/offline and are
**not** used by RL. So the minimal set is:

| Purpose | Drive path | Local destination |
|---|---|---|
| Trained **gemsplat** scene | `3dgs/workspace/outputs/flightroom_ssv_exp/gemsplat/` | `FiGS-Standalone/3dgs/workspace/outputs/flightroom_ssv_exp/gemsplat/` |
| Source scene meta + images | `3dgs/workspace/flightroom_ssv_exp/{transforms.json, sparse_pc.ply, images/}` | same under `FiGS-Standalone/3dgs/workspace/flightroom_ssv_exp/` |
| Setup trajectories | `FiGS_data/vlead_flightroom/rollout_data/<DATE>/flightroom_ssv_exp_2026-05-22_071{718,353}/trajectories_val0000[01].pt` | `nav_policy/data/raw/flightroom_ssv_exp_2026-05-22_071{718,353}/` |
| BC warm-start | `Rahul_Best_Weights/bc_best_balanced_dagger_r12_new.pt` | `nav_policy/data/checkpoints_a2_da2_crossattn/bc_best.pt` |
| DA2 depth weights | (already on disk in `~/.cache`) | symlink → `nav_policy/data/weights/depth_anything_v2_vits.pth` |

**Two gotchas discovered the hard way:**
- **Pick exactly ONE gsplat variant.** Under `outputs/flightroom_ssv_exp/` there are both
  `gemsplat/` and `splatfacto/`. FiGS's scene loader does `rglob("*.yml")` and **errors if it
  finds more than one config** (`simulator.py:137`). Download only `gemsplat/`.
- **The real `flightroom_ssv_exp` source uses `images/` (150 frames `frame_00001..00150.png`),
  NOT `rgbs/`.** The Drive also contains a *different* scene with an `rgbs/` folder (300 frames,
  `0000.png`...). An early gdown attempt grabbed the wrong `transforms.json` (rgbs, 300 frames)
  and the scene load failed looking for `rgbs/0000.png`. The correct `transforms.json` has 150
  frames referencing `images/frame_*.png`. Always verify: `transforms.json` `frames[0].file_path`
  must be `images/frame_00001.png`.

### 2.4 rclone download commands (recommended)

rclone is already installed and a `gdrive:` remote already exists on this machine. Address the
folder by id and copy only the needed sub-paths. **`--tpslimit` matters** — the default rclone
client_id is a globally-shared Google project that gets HTTP 403 "Quota exceeded" without pacing.

```bash
FID=14BB9ajONol2qFXe2o-25dkEJ_mrgUux9
V=/home/rohang73/Desktop/cs224r_yashadityachinmay/V-LEAD
F="--drive-root-folder-id $FID --drive-acknowledge-abuse --tpslimit 6 --retries 8 --low-level-retries 20 --progress"

# 1. trained gemsplat scene (ONLY gemsplat, not splatfacto)
rclone copy "gdrive:3dgs/workspace/outputs/flightroom_ssv_exp/gemsplat" \
  "$V/FiGS-Standalone/3dgs/workspace/outputs/flightroom_ssv_exp/gemsplat" $F

# 2. source scene: transforms.json + sparse_pc.ply (skip images/ subfolders we don't need)
rclone copy "gdrive:3dgs/workspace/flightroom_ssv_exp" \
  "$V/FiGS-Standalone/3dgs/workspace/flightroom_ssv_exp" \
  --include "transforms.json" --include "sparse_pc.ply" $F
# 2b. the 150 training images (gemsplat eagerly loads them at startup — see §3.5 note)
rclone copy "gdrive:3dgs/workspace/flightroom_ssv_exp/images" \
  "$V/FiGS-Standalone/3dgs/workspace/flightroom_ssv_exp/images" $F

# 3. setup_from trajectories. NOTE the nested <DATE> folder — addressing by path 404s if you
#    flatten it. Easiest: copy the whole run folder (it's small once you exclude videos), or use
#    --drive-root-folder-id with the run folder's own id. Path form:
rclone copy "gdrive:FiGS_data/vlead_flightroom/rollout_data/2026-05-22_071718/flightroom_ssv_exp_2026-05-22_071718" \
  "$V/nav_policy/data/raw/flightroom_ssv_exp_2026-05-22_071718" \
  --include "trajectories_val0000[01].pt" $F
rclone copy "gdrive:FiGS_data/vlead_flightroom/rollout_data/2026-05-22_071353/flightroom_ssv_exp_2026-05-22_071353" \
  "$V/nav_policy/data/raw/flightroom_ssv_exp_2026-05-22_071353" \
  --include "trajectories_val0000[01].pt" $F

# 4. BC warm-start checkpoint
rclone copy "gdrive:Rahul_Best_Weights/bc_best_balanced_dagger_r12_new.pt" \
  "$V/nav_policy/data/checkpoints" $F

# 5. place the checkpoint where the SAC config's `checkpoint:` field expects it
mkdir -p "$V/nav_policy/data/checkpoints_a2_da2_crossattn"
cp "$V/nav_policy/data/checkpoints/bc_best_balanced_dagger_r12_new.pt" \
   "$V/nav_policy/data/checkpoints_a2_da2_crossattn/bc_best.pt"

# 6. DA2 depth weights (already cached on this machine from a prior run)
mkdir -p "$V/nav_policy/data/weights"
ln -sf ~/.cache/depth_anything_v2/depth_anything_v2_vits.pth \
   "$V/nav_policy/data/weights/depth_anything_v2_vits.pth"
```

> **gdown alternative.** `gdown --folder <id>` works for small folders but: (a) truncates folders
> with >50 files, (b) reconstructing which file is which from its flat listing is error-prone
> (this is how we grabbed the wrong `rgbs` `transforms.json`). Prefer rclone with path-addressing.

> **Disk warning.** On this machine `/` was at 97% (~64 GB free). The minimal set fits easily;
> the *full* archive may not. Don't `rclone copy` the whole folder unless you've checked `rclone size`.

---

## 3. Environment fixes (in the order we hit them)

### 3.1 `requirements.txt` dependency conflicts

**Symptom A — hard resolver failure:**
```
ERROR: Cannot install -r requirements.txt (line 23) and gsplat==1.5.3 because these package
versions have conflicting dependencies.
  The user requested gsplat==1.5.3
  nerfstudio 1.1.5 depends on gsplat==1.4.0
```
**Cause:** `requirements.txt` pinned `gsplat==1.5.3`, but `nerfstudio==1.1.5` hard-pins
`gsplat==1.4.0`. (The "no matching distributions" line in the error is a red herring from pip's
backtracking — gsplat 1.4.0 does exist on PyPI.)
**Fix:** change `gsplat==1.5.3` → `gsplat==1.4.0` in `requirements.txt`.

**Symptom B — numpy conflict surfaced after install:**
```
opencv-python 4.13.0.92 requires numpy>=2; ... but you have numpy 1.26.4
```
**Cause:** `grad-cam` (a real `nav_policy` dependency, used by `scripts/gradcam_saliency.py`) pulls
in the full `opencv-python` *by name*, which defaults to 4.13 and wants `numpy>=2`. That collides
with `numpy==1.26.4`, which gsplat/nerfstudio require. Also, `opencv-python` and
`opencv-python-headless` install to the **same `cv2/` directory**, so uninstalling one deletes the
shared files and breaks `import cv2`.
**Fix:** pin the full opencv to the numpy-1.x-compatible 4.10.x alongside headless. Added to
`requirements.txt`:
```
opencv-python-headless==4.10.0.84
opencv-python==4.10.0.84   # grad-cam needs full opencv by name; pin to 4.10 for numpy-1.x compat
```
If you already broke `cv2`, repair with:
```
pip install --force-reinstall --no-deps "opencv-python-headless==4.10.0.84" "opencv-python==4.10.0.84"
```
Verify: `pip check` → "No broken requirements found".

> Both fixes are already committed in `requirements.txt`. Just `pip install -r requirements.txt`.

### 3.2 CUDA toolkit must be on PATH (gsplat / tcnn JIT)

gsplat (and later tiny-cuda-nn) JIT-compile CUDA kernels **on first use**, not at install — they
need `nvcc` on `PATH`. On this box `nvcc` is at `/usr/local/cuda/bin/nvcc` (CUDA 12.9) but not on
`PATH` by default. `import gsplat` works without it; the first *render* fails without it.
**Fix:** `export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH`. (torch is cu128 / 12.8,
toolkit is 12.9 — the minor mismatch is fine for JIT.) This is documented in `nav_policy/nav_policy_no_docker.md`.

### 3.3 The scene config has no absolute paths (good news)

`outputs/.../gemsplat/2026-02-23_043616/config.yml` stores `data:` and `output_dir:` as **relative**
`PosixPath`s (`flightroom_ssv_exp`, `outputs`). FiGS `chdir`s into the workspace dir before loading,
so they resolve correctly. No path rewriting needed.

### 3.4 `gemsplat` not installed → install editable (+ two sub-fixes)

**Symptom:** scene load fails to deserialize `config.yml` — `ModuleNotFoundError: No module named 'gemsplat'`.
**Cause:** `gemsplat` (the nerfstudio CLIP-distilled-splat plugin) is vendored at
`FiGS-Standalone/gemsplat` and is editable-installed automatically inside Docker, but the no-Docker
flow skips it.
**Fix:** `pip install -e FiGS-Standalone/gemsplat`. This surfaced two more issues:

- **`ModuleNotFoundError: No module named 'pkg_resources'`** — gemsplat's vendored OpenAI-CLIP code
  does `from pkg_resources import packaging`, but `setuptools>=81` removed `pkg_resources`.
  **Fix:** `pip install "setuptools<81"` (we landed on 80.10.2).

- **`RuntimeError: Cannot subclass _TensorBase directly`** (from `torchtyping`) — `torchtyping 0.1.4`
  is unmaintained and incompatible with torch 2.x. gemsplat imports it only for type annotations in
  `gemsplat/data/utils/utils.py`. **Fix:** replaced the import with a subscriptable no-op shim
  (already applied in that file):
  ```python
  try:
      from torchtyping import TensorType
  except Exception:  # torchtyping breaks on torch>=2
      class _TensorTypeMeta(type):
          def __getitem__(cls, item):
              return cls
      class TensorType(metaclass=_TensorTypeMeta):
          pass
  ```

Verify: `python -c "import gemsplat"` and
`python -c "from nerfstudio.configs.method_configs import all_methods; print('gemsplat' in all_methods)"` → `True`.

> On first scene load, gemsplat downloads a **CLIP RN50x64 model (1.26 GB)** to `~/.cache`, and
> nerfstudio downloads **AlexNet (233 MB)** for LPIPS. One-time; just let them run.

### 3.5 `tiny-cuda-nn` not installed → RGB rendering crashes

**Symptom:** scene loads, but rendering raises `TypeError: 'NoneType' object is not callable` at
`gemsplat.py:1091` (`self.clip_field(...)`).
**Cause:** gemsplat builds its CLIP feature field as a `tcnn.NetworkWithInputEncoding`. The code is
`if 'tcnn' in globals() and ...: self.clip_field = tcnn... else: self.clip_field = None`. Without
`tinycudann`, `clip_field` is `None`. And RGB rendering hits it because FiGS's `render_rgb` calls
`get_outputs_for_camera(...)` with the default `compute_semantics=True`.
**Fix:** build tiny-cuda-nn from source for your GPU arch (RTX 4090 = sm_89):
```bash
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH TCNN_CUDA_ARCHITECTURES=89
pip install --no-build-isolation "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
```
This is a ~10–20 min CUDA compile. Verify: `python -c "import tinycudann as tcnn; print('NetworkWithInputEncoding' in dir(tcnn))"` → `True`.

> **Aside — why the source images are needed.** gemsplat's datamanager (`gemsplat_datamanager.py:96`)
> *eagerly loads all training images* even in inference mode, to compute CLIP embeddings and set
> `metadata["feature_dim"]` (=1024 for RN50x64), which the model reads in `populate_modules` to size
> `clip_field`. So you must have the 150 `images/frame_*.png`. (Vanilla nerfstudio inference skips
> image pixels — gemsplat does not.)

### 3.6 ACADOS not installed → FiGS rollouts fail

**Symptom:** SAC starts, loads the BC checkpoint, then every rollout fails:
`FAILED: No module named 'acados_template'`, so `iter 0: no episodes collected`.
**Cause:** ACADOS is FiGS's MPC / dynamics-integrator solver. It's vendored at
`FiGS-Standalone/acados` but neither the C library nor the Python interface was built.
**Fix:** build it (the submodules blasfeo/hpipm/qpoases are already vendored), install the Python
package, and download the Tera template renderer:
```bash
cd FiGS-Standalone/acados
mkdir -p build && cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4           # produces ../lib/libacados.so
cd ../..
pip install -e interfaces/acados_template
# t_renderer (code-gen binary) — fetch non-interactively (else it prompts input() -> EOFError headless):
ACADOS_SOURCE_DIR=$(pwd) python -c "from acados_template.utils import get_tera; print(get_tera(force_download=True))"
```
Then **at runtime** you must set:
```bash
export ACADOS_SOURCE_DIR=<repo>/FiGS-Standalone/acados
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
```

**Sub-issue we hit:** with ACADOS present but no `t_renderer`, the rollout failed with
`EOF when reading a line`. That's `acados_template` prompting
`"Do you wish to set up Tera renderer automatically? y/N"` via `input()` in a non-interactive
process. The `get_tera(force_download=True)` call above pre-installs it (we got `t_renderer` v0.2.0,
linux-amd64) so the prompt never fires.

### 3.7 `torch.load` weights-only patch (nerfstudio checkpoints)

**Symptom:** rollout loads the gsplat checkpoint and fails:
```
Weights only load failed ... WeightsUnpickler error:
Unsupported global: GLOBAL numpy.core.multiarray.scalar was not an allowed global by default.
```
**Cause:** PyTorch 2.6 changed `torch.load`'s default to `weights_only=True`, which rejects the
full-pickle nerfstudio/gsplat checkpoints. `yash_readme.md` notes this is handled by a `torch.load`
monkey-patch — but that patch was **missing** on the `chinmay/asl_setup` branch.
**Fix (already applied):** added a process-wide patch at the top of
`nav_policy/src/nav_policy/rl/train_rl.py` (right after `import torch`):
```python
_orig_torch_load = torch.load
def _torch_load_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_weights_only_false
```
(Rollouts run in-process, so this covers nerfstudio's internal `eval_load_checkpoint`.)

### 3.8 CUDA OOM during the SAC update → smaller batch (24 GB cards)

**Symptom:** a full episode is collected (e.g. `return=-34.89 steps=1170`), then the SAC update OOMs:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.53 GiB.
GPU 0 has total 23.50 GiB ... this process has ~20 GiB in use.
```
**Cause:** the canonical `train_rl_flightroom_sac.yaml` uses `sac.batch_size: 256`, tuned for the
maven 32 GB RTX 5090. The SAC update runs the ResNet-18 backbone several times per step (twin
critics + target + policy) on a `[B, T=4, 3, 224, 224]` batch; at B=256 that's ~1024 images of
gradient activations on top of the ~20 GB already resident (gsplat gaussians + tcnn CLIP field +
CLIP RN50x64 + DA2 + policy). `expandable_segments:True` alone is not enough.
**Fix:** use the 24 GB override config (already added):
`configs/train_rl_flightroom_sac_24gb.yaml`
```yaml
base_config: configs/train_rl_flightroom_sac.yaml
rl:
  sac:
    batch_size: 64
```
Also keep `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. With B=64 the update fits and
the loop completes (new-best logged, `rl_sac_a2_best.pt` / `rl_sac_a2_latest.pt` saved).

---

## 4. Verification ladder (how we confirmed each layer)

1. `pip check` → clean (no broken requirements).
2. `import torch, gsplat, nerfstudio` → OK; `torch.cuda.is_available()` → True.
3. `import gemsplat` + nerfstudio method registered → True.
4. `eval_setup(config.yml, test_mode="inference")` → scene loads (with the `torch.load` patch).
5. `model.get_outputs_for_camera(cam)` → renders `(1080,1920,3)` RGB (after tcnn) → confirms the
   full render path.
6. `train_rl.py ... --n-iterations 1 --rollouts-per-iteration 1` → collects an episode (1170 steps,
   live render + MPC) and completes one SAC update + checkpoint save → **full pipeline verified**.

The first verified smoke result:
```
[ep 00001] [flightroom_071353_q01|green_clock] return=-34.89 steps=1170 success=False
           final_dist=0.26m collision=False end=timeout
[rl] new best return=-34.89 -> rl_sac_a2_best.pt
[rl] done.   total_episodes=1
```
(`end=timeout`, `final_dist=0.26 m` = the BC policy flies but doesn't quite settle — which is what
SAC is meant to improve.)

---

## 5. Files changed / added in this session

- `requirements.txt` — `gsplat 1.5.3→1.4.0`; pinned `opencv-python==4.10.0.84`.
- `nav_policy/nav_policy_no_docker.md` — documented gsplat pins + the CUDA-toolkit-on-PATH requirement.
- `FiGS-Standalone/gemsplat/gemsplat/data/utils/utils.py` — torchtyping shim (torch-2.7 compat).
- `nav_policy/src/nav_policy/rl/train_rl.py` — `torch.load` weights-only monkey-patch.
- `nav_policy/configs/train_rl_flightroom_sac_24gb.yaml` — new, `sac.batch_size: 64` for 24 GB GPUs.
- Built/installed (not in git): `gemsplat` (editable), `tinycudann` (sm_89), ACADOS C lib +
  `acados_template` (editable) + `t_renderer` v0.2.0, `setuptools<81`.
- Data downloaded under `FiGS-Standalone/3dgs/workspace/...` and `nav_policy/data/...` (gitignored).

---

## 6. Gotchas checklist (for next time)

- [ ] `nvcc` on PATH (`CUDA_HOME=/usr/local/cuda`) — else gsplat/tcnn JIT fails at first render.
- [ ] Only ONE gsplat variant under `outputs/<scene>/` — else FiGS errors on multiple `*.yml`.
- [ ] Correct `flightroom_ssv_exp` source = `images/` (150 frames), not the decoy `rgbs/` (300 frames).
- [ ] `setuptools<81` (pkg_resources) + torchtyping shim before `import gemsplat`.
- [ ] tiny-cuda-nn built for your `TCNN_CUDA_ARCHITECTURES` (4090 = 89) — else `clip_field=None`.
- [ ] ACADOS built + `acados_template` installed + `t_renderer` pre-downloaded + env vars set.
- [ ] `torch.load` weights-only patch present in `train_rl.py`.
- [ ] 24 GB GPU → `sac.batch_size: 64` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- [ ] Watch disk (`/` was 97% full here) — `checkpoint_dir` is `data/checkpoints_rl_a2`.
