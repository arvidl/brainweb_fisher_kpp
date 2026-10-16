# Validation — 16 October 2026

- `uv sync --locked --offline`: passed with the installed Python 3.12 environment.
- `pytest -q`: **14 passed**.
- Jupyter notebook: all seven cells executed without errors.
- Widget callback check: short 3D simulation, time selection and arbitrary slice
  update passed (`scripts/check_notebook.py`).
- JupyterLab browser check: Run button completed the default 180-day, 2 mm run;
  all three anatomical views, common density colorbar, playback and view sliders
  fit in the notebook. Screenshot: `notebook_verified.png`.
- Portability check: copied source, data and lockfile into a separate folder
  under `/tmp`, recreated its environment offline from the uv cache, and
  independently generated a 3-day GIF using its own installed package.
- Replay check: re-rendered the saved 1 mm result without solving again.
- BrainWeb source alignment: identical native affines and 181 × 217 × 181 grids;
  checked official GM/WM tissue counts and file checksums.
- Native 1 mm example: 180 simulated days, 720 internal steps, maximum Δt=0.25
  day, 121 output frames. Solver runtime was 5.32 seconds on this host, excluding
  media rendering and initial font-cache setup.
- MP4 decoded successfully: H.264, yuv420p, 1920 × 1080, 12 fps, 139 frames,
  11.583 seconds including initial/final holds.
- GIF decoded successfully: 960 × 540, 121 distinct encoded frames, continuous
  loop, 11.12 seconds including holds. GIF timing rounds to centiseconds.
- First/final frames were visually inspected. The scientific solver tests are
  implementation checks, rather than patient-level biological validation.
