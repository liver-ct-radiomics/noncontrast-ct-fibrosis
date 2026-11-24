# Non-Contrast CT Radiomics and Clinical Markers for Liver Fibrosis

Public code repository accompanying the manuscript:

> *Complementing clinical markers with whole-liver radiomics for biopsy-defined liver fibrosis: A nested cross-validation study using routine non-contrast CT*

## Repository Contents

1. **`nested_cv_pipeline.py`** - Unified modeling pipeline (5-fold stratified nested cross-validation repeated 20 times) reproducing all modeling tables and incremental-value analyses:
   - `Table 2`: A priori serum indices (`APRI`, `FIB-4`) and sequential clinical models (`Clinical models 2-5`, with `FIB-4 + sex` carried forward as the selected clinical core).
   - `Table 3`: Standalone radiomic-only models (`2-5` parameters; training-fold unsupervised pre-filtering and L1 path ranking).
   - `Table 4`: Unanchored joint clinical and radiomic models (`3-7` parameters; training-fold pre-filtering and cohort-prefiltered check).
   - `Table 5`: Clinically anchored hybrid models (`K = 1` to `5` radiomic features) across exploratory post-selection consensus evaluation (Regime 1) and dynamic within-fold L1 selection with cohort-level (Regime 2) and training-fold-only (Regime 3) unsupervised pre-filtering, plus top-5 cross-fold selection frequencies.
   - Descriptive full-cohort unpenalized likelihood-ratio test (`LRT`) and out-of-fold paired DeLong test, IDI, and continuous NRI comparisons at `K = 4`.
2. **`plot_figures.py`** - Generates main and supplementary evaluation figures from the out-of-fold probability tables produced by `nested_cv_pipeline.py` (both `.png` and 300-dpi LZW `.tif`, plus `dca_thresholds.csv`):
   - `Fig3`: Flexible LOESS calibration curves, 95% bootstrap bands, and predicted probability histograms at `K = 4`.
   - `Fig4`: Decision curve analysis at `K = 4`.
   - `Fig5`: Out-of-fold receiver operating characteristic (ROC) curves at `K = 4`.
   - `S1_Fig`: Supplementary `5 x 2` panel of flexible LOESS calibration curves and decision curves across `K = 1` to `5`.
3. **`inner_loop_sensitivity.py`** - Additional fully nested sensitivity analysis in which clinical expansion size (`m = 0` to `4` beyond FIB-4), radiomic subset size (`K = 0` to `5`), and the L2 penalty `C` (`0.01` to `10.0`) are jointly selected by 5-fold inner cross-validation under both the one-standard-error (1-SE) rule and the maximum inner-AUC rule.
4. **`LiverSegICC/`** - Standalone desktop workflow and analysis script used for the 30-patient blinded dual-reader whole-liver segmentation reproducibility substudy (`TotalSegmentator` vs. Reader 1 vs. Reader 2; `S1 Table`), including DICOM import, blinded slice-wise mask review, spatial overlap metrics (`Dice`, `IoU`, `ASSD`, `HD95`), and `ICC(2,1)` analysis. Full setup and usage instructions are provided in [`LiverSegICC/README.md`](LiverSegICC/README.md).

   Note: `LiverSegICC` runs in a standalone Python environment outside 3D Slicer and does not call PyRadiomics. The four `K = 1–4` consensus features are computed with NumPy/SciPy from the corresponding definitions at the native voxel spacing (no resampling to 1 × 1 × 1 mm): sphericity from a marching-cubes surface area; undecimated 3D Coiflet-1 filtering; a fixed bin width of 25; 26-connected gray-level zones; and 13-direction symmetric 3D gray-level co-occurrence. Values are therefore not numerically identical to the main PyRadiomics extraction.

## Usage

Install dependencies for the modeling and plotting scripts:

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

4. For the dual-reader segmentation reproducibility application (`LiverSegICC`), see [`LiverSegICC/README.md`](LiverSegICC/README.md) (`setup.bat` and `run.bat`, or `python LiverSegICC/src/selftest.py` to run the verification checks).

## License

Released under the GNU General Public License v3.0 (GPL-3.0). See https://www.gnu.org/licenses/gpl-3.0.html for license terms.
