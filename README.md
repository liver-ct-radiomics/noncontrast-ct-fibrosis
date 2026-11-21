# Non-Contrast CT Radiomics and Clinical Markers for Liver Fibrosis

Public code repository accompanying the manuscript:

> *Complementing clinical markers with whole-liver radiomics for biopsy-defined liver fibrosis: A nested cross-validation study using routine non-contrast CT*

## Scripts

1. **`nested_cv_pipeline.py`** — Main 20-repeat 5-fold nested cross-validation pipeline. Evaluates a priori FIB-4, the selected clinical core (`FIB-4 + sex`), exploratory post-selection consensus hybrid models (Regime 1), and within-fold L1 selection models with cohort-level (Regime 2) and training-fold-only (Regime 3) unsupervised pre-filtering across `K = 1` to `5` radiomic features. Writes `nested_cv_summary.csv`, `oof_predictions.csv`, `oof_predictions_all_k1_to_5.csv`, and `incremental_value.csv`.
2. **`plot_figures.py`** — Generates main and supplementary evaluation figures from the out-of-fold probability tables produced by `nested_cv_pipeline.py` (both `.png` and 300-dpi LZW `.tif`, plus `dca_thresholds.csv`):
   - `Fig3`: Flexible LOESS calibration curves, 95% bootstrap bands, and predicted probability histograms at `K = 4`.
   - `Fig4`: Decision curve analysis at `K = 4`.
   - `Fig5`: Out-of-fold receiver operating characteristic (ROC) curves at `K = 4`.
   - `S1_Fig`: Supplementary `5 x 2` panel of flexible LOESS calibration curves and decision curves across `K = 1` to `5`.
3. **`inner_loop_sensitivity.py`** — Additional fully nested sensitivity analysis in which clinical expansion size (`m = 0` to `4` beyond FIB-4), radiomic subset size (`K = 0` to `5`), and the L2 penalty `C` (`0.01` to `10.0`) are jointly selected by 5-fold inner cross-validation under both the one-standard-error (1-SE) rule and the maximum inner-AUC rule. Writes `inner_loop_sensitivity_summary.csv`.
4. **`segmentation_reproducibility.py`** — Blinded dual-reader whole-liver segmentation reproducibility analysis (`TotalSegmentator` vs. Reader 1 vs. Reader 2 across 30 examinations), computing pairwise spatial overlap (`Dice`, `IoU`, `ASSD`, `HD95`) and `ICC(2,1)` with 95% confidence intervals for liver volume, mean parenchymal attenuation, and the four `K = 1` to `4` consensus radiomic features (`S1 Table`).

   Note: this script does not call PyRadiomics. The substudy tool runs in a standalone Python environment outside 3D Slicer, and the four features are computed with NumPy/SciPy from the corresponding definitions at the native voxel spacing (no resampling to 1 × 1 × 1 mm): sphericity from a marching-cubes surface area; undecimated 3D Coiflet-1 filtering; a fixed bin width of 25; 26-connected gray-level zones; and 13-direction symmetric 3D gray-level co-occurrence. Values are therefore not numerically identical to the main PyRadiomics extraction.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

1. Run the nested cross-validation pipeline on the tab-delimited feature table (`Label`: `0` = Ishak stages 0–1, `1` = Ishak stages 2–5):

```bash
python nested_cv_pipeline.py --input LBase.txt --outdir results
```

2. Generate main figures (`Fig3`, `Fig4`, `Fig5`) and the supplementary `K = 1–5` figure (`S1_Fig`):

```bash
python plot_figures.py --indir results --outdir figures
```

3. Run the additional inner-loop `(m, K, C)` sensitivity analysis:

```bash
python inner_loop_sensitivity.py --input LBase.txt --outdir results
```

4. Run the segmentation reproducibility analysis on a directory of per-case NIfTI folders (`ct.nii.gz`, `TS.nii.gz`, `R1.nii.gz`, `R2.nii.gz`):

```bash
python segmentation_reproducibility.py --cases_dir path/to/cases --outdir results
```

## License

Released under the GNU General Public License v3.0 (GPL-3.0). See https://www.gnu.org/licenses/gpl-3.0.html for license terms.
