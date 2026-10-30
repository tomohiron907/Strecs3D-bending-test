# Strecs3D three-point bending test data

Raw data, analysis script, and finite-element results for the mass-matched
three-point bending experiment in the paper *Strecs3D: Stress-Driven Infill
Optimization as a Slicing Preprocess for Material Extrusion 3D Printing*
(Taniguchi and Hayama). The software itself is at
<https://github.com/tomohiron907/Strecs3D> (release `v2.1.0`).

## Experiment

- Specimens: 180 × 50 × 7.5 mm beams, printed flat on a Bambu Lab P1S in
  Overture Matte PLA (1.75 mm), sliced in Bambu Studio (0.20 mm Standard
  profile, gyroid infill, no top or bottom layers).
- Conditions (three specimens each, both 15 g):
  - `normal1`–`normal3`: uniform infill at 14%.
  - `strecs1`–`strecs3`: optimized with Strecs3D for a 100 N central load,
    four subvolumes at 6, 10, 15, and 55%.
- Test: A&D MCT1150, edgewise (load along the 50 mm direction),
  span L = 110 mm, crosshead speed 10 mm/min.

## Contents

| Path | Description |
| --- | --- |
| `data/*.csv` | Raw machine output, one file per specimen. Shift_JIS encoded; columns are test number, time [s], displacement [mm], load [N]. |
| `analysis/analyze.py` | Analysis script (zero-load correction, toe correction, apparent flexural properties). Comments are in Japanese. |
| `analysis/summary.csv`, `analysis/summary_group.csv` | Results per specimen and per condition (mean, SD). |
| `analysis/processed/` | Corrected load–deflection and stress–strain data per specimen. |
| `analysis/figures_journal/` | Figures in English. `analysis/figures/` has the same figures in Japanese. |
| `fea/beam_mesh{5.0,2.5,1.25}mm.vtu` | Strecs3D FEA results for the beam at maximum element sizes of 5, 2.5, and 1.25 mm (mesh convergence check). The paper uses the 5 mm mesh. |

Flexural stress, strain, and modulus follow the formulas of ASTM D790
(b = 7.5 mm, h = 50 mm). Because L/h = 2.2, they are nominal values used only
to compare the two conditions.

## Reproduce

```sh
cd analysis
uv run --with numpy --with pandas --with matplotlib python analyze.py
```

## License

Data and figures: CC BY 4.0. Code: MIT.
