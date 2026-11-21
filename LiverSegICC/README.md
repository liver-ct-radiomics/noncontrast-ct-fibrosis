# LiverSegICC

A standalone Windows desktop application for blinded dual-reader whole-liver segmentation reproducibility analysis (`S1 Table`). `TotalSegmentator` generates the initial whole-liver mask (`TS`), two readers (`AUTH1` / Reader 1 and `AUTH2` / Reader 2) independently review and edit slices while blinded to each other's masks, and the analysis module compares all three masks.

**For each case it records:**
- Whole-liver volume (mL) and mean/SD parenchymal attenuation (HU) across `TS`, `AUTH1`, and `AUTH2`
- The four `K = 1–4` consensus radiomic features (`original_shape_Sphericity`, `wavelet-LLL_firstorder_10Percentile`, `wavelet-HLL_glszm_GrayLevelNonUniformityNormalized`, and `wavelet-LLH_glcm_ClusterProminence`)

**Across cases it calculates** (for `TS vs AUTH1`, `TS vs AUTH2`, `AUTH1 vs AUTH2`, and three-way agreement):
- Pairwise spatial overlap: `Dice`, `IoU`, `ASSD` (mm), `HD95` (mm), and percentage volume difference
- Two-way random-effects, absolute-agreement, single-measurement `ICC(2,1)` (and `ICC(3,1)`) with 95% confidence intervals, plus Bland-Altman bias and 95% limits of agreement

This is research software and is not intended for clinical diagnostic use.

## Requirements

- Windows 10 or 11
- Python 3.11 or 3.12 (`setup.bat` creates a local `.venv` and installs dependencies)
- Optional NVIDIA GPU for `TotalSegmentator` inference (not required if reviewing pre-segmented workspaces)

## Getting started

1. Place `LiverSegICC` in a local directory outside cloud-synced folders (do not use OneDrive when working with DICOM images).
2. Run `setup.bat` to create `.venv`, install `requirements.txt`, and run `src/selftest.py` against published benchmark values.
3. Run `run.bat` to launch the desktop interface.

When opened, the application searches for an existing `TotalSegmentator` installation (`PATH`, Python/conda environments, or 3D Slicer) or offers local installation into `.venv`.

## Workflow

### 1. Case import and initial segmentation

Organize per-patient DICOM studies in separate folders:

```text
Cases/
  Case_01/ ...
  Case_02/ ...
```

Click **Choose...**, select `Cases`, tick the cases to process, and click **Segment ticked cases**. Source DICOM folders are read-only and never modified.

### 2. Blinded slice-wise reader review

Select the active reader (`AUTH1` or `AUTH2`). Each reader starts from an independent copy of the raw `TS` mask in randomized case order and never sees the other reader's edits.

| Action | Mouse | Key |
| --- | --- | --- |
| Brush (add) | Left button | `B` |
| Eraser (remove) | Left button | `E` |
| Opposite tool (temporary) | Right button | |
| Brush diameter | `Shift` + wheel, or `-` / `+` buttons | `+` `-` |
| Change slice | Wheel or scrollbar | `Up` `Down`, `PgUp` `PgDn` |
| Zoom / reset | `Ctrl` + wheel | `0` |
| Pan | Middle button, or `Space` + left button | `H` |
| Undo, redo, save | | `Ctrl+Z`, `Ctrl+Y`, `Ctrl+S` |
| Window presets, mask, contour | | `1` `2` `3`, `M`, `C` |

Mark each reviewed examination as **DONE** when complete.

### 3. Reproducibility analysis

Click **Run analysis** once both readers have completed review. Outputs are written to `workspace/results/<timestamp>/`:

| File | Contents |
| --- | --- |
| `report.html` | Complete summary report |
| `measurements.csv` | Per-case volume, attenuation, and the four `K = 1–4` radiomic features across `TS`, `AUTH1`, and `AUTH2` |
| `overlap_per_case.csv` | Per-case pairwise `Dice`, `IoU`, `HD95`, `ASSD`, and volume difference |
| `overlap_summary.csv` | Summary statistics across pairs (`S1 Table` upper panel) |
| `icc_summary.csv` | Pairwise and three-way `ICC(2,1)` and `ICC(3,1)` with 95% CIs and Bland-Altman limits (`S1 Table` lower panel) |
| `case_details.csv` | Reader effort (active time, voxels added/removed, strokes) and acquisition metadata |
| `bland_altman_*.png` | Bland-Altman agreement plots |
| `excel/` | Locale-formatted CSV exports |

## How measurements and radiomic features are calculated

- **ICC:** Primary metric is `ICC(2,1)` (two-way random effects, absolute agreement, single measurement; Shrout & Fleiss 1979; Koo & Li 2016).
- **Spatial overlap:** `Dice`, `IoU`, `ASSD`, and `HD95` are computed directly on binary 3D voxel masks using physical voxel spacing.
- **Standalone radiomic feature implementation (`src/metrics.py`):** This tool runs in a standalone Python environment outside 3D Slicer and does not call `PyRadiomics`. The four `K = 1–4` consensus features are computed directly with `NumPy`/`SciPy` from the corresponding mathematical definitions at native voxel spacing (without resampling to `1 × 1 × 1 mm`): sphericity from a marching-cubes surface-area lookup table; undecimated 3D Coiflet-1 wavelet filtering; a fixed bin width of 25; 26-connected gray-level size zones; and 13-direction symmetric 3D gray-level co-occurrence. Feature values are therefore not numerically identical to the resampled `PyRadiomics` extraction used for model development.

To run the built-in verification suite directly:

```bash
python src/selftest.py
```

## Privacy

All processing runs locally. The `workspace/` directory (which holds NIfTI volumes and `case_source_map.csv`) and `local_settings.json` are excluded via `.gitignore` and must never be shared outside the study team.

## Reference

Wasserthal J, Breit HC, Meyer MT, Pradella M, Hinck D, Sauter AW, et al. TotalSegmentator: robust segmentation of 104 anatomic structures in CT images. *Radiol Artif Intell*. 2023;5(5):e230024.
