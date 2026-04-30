# AguaTrack-ARCO-SA Tutorial

Companion code release for the **AguaTrack-ARCO-SA** dataset — WAM2Layers
(v3.3.1) backward moisture-tracking output over South America at 0.25°
daily resolution, 1990–2019. The dataset is published on HuggingFace at
[`AguaTrack/AguaTrack-ARCO-SA`](https://huggingface.co/datasets/AguaTrack/AguaTrack-ARCO-SA)
and answers, for any tagged grid cell:
*"where did the rain that fell here today evaporate from?"*

This repository hosts six Colab-runnable Jupyter notebooks that reproduce
the science figures in the data paper. Each notebook is self-contained —
open the Colab link in its row, run all cells, and you get the figure.

## Notebook index

| # | Story | Region | Open in Colab |
|---|---|---|---|
| 00 | Setup and dataset primer | — | [open](https://colab.research.google.com/github/NTU-CompHydroMet-Lab/AguaTrack-ARCO-SA-Tutorial/blob/main/notebooks/00_setup_and_data.ipynb) |
| 01 | Serra do Mar 2011 flash-flood event | SE Brazil | [open](https://colab.research.google.com/github/NTU-CompHydroMet-Lab/AguaTrack-ARCO-SA-Tutorial/blob/main/notebooks/01_event_serra_do_mar_2011.ipynb) |
| 02 | The "aerial river" feeding Santa Cruz | Bolivia (Amazon source) | [open](https://colab.research.google.com/github/NTU-CompHydroMet-Lab/AguaTrack-ARCO-SA-Tutorial/blob/main/notebooks/02_aerial_river_santa_cruz.ipynb) |
| 03 | ENSO-phase composites of moisture sources | Central Chile | [open](https://colab.research.google.com/github/NTU-CompHydroMet-Lab/AguaTrack-ARCO-SA-Tutorial/blob/main/notebooks/03_enso_composites.ipynb) |
| 04 | The collapse of Aculeo Lake | Central Chile | [open](https://colab.research.google.com/github/NTU-CompHydroMet-Lab/AguaTrack-ARCO-SA-Tutorial/blob/main/notebooks/04_aculeo_lake_collapse.ipynb) |
| 05 | Central-Chile megadrought decade | Central Chile | [open](https://colab.research.google.com/github/NTU-CompHydroMet-Lab/AguaTrack-ARCO-SA-Tutorial/blob/main/notebooks/05_megadrought_central_chile.ipynb) |

## Getting started

### On Google Colab

Click any "open" link in the table above. Step 1 of each notebook is a
"Configuration" cell holding the HuggingFace zarr URLs and any region /
year constants you might want to edit. Step 2 installs the geo stack
(Colab only). Steps 3+ run the analysis.

### Locally with `uv`

```bash
git clone https://github.com/NTU-CompHydroMet-Lab/AguaTrack-ARCO-SA-Tutorial.git
cd <REPO>
uv sync
uv run jupyter lab
```

For local zarr archives on disk, replace the `hf://datasets/...` URLs
in each notebook's Step 1 cell with filesystem paths and drop the
`storage_options={"revision": ...}` argument.

## Repository layout

```
.
├── notebooks/                  # Colab-runnable tutorials (one per story)
│   ├── 00_setup_and_data.ipynb
│   ├── 01_event_serra_do_mar_2011.ipynb
│   ├── 02_aerial_river_santa_cruz.ipynb
│   ├── 03_enso_composites.ipynb
│   ├── 04_aculeo_lake_collapse.ipynb
│   └── 05_megadrought_central_chile.ipynb
├── docs/                       # dataset schema documentation
├── tools/                      # build helpers
├── pyproject.toml              # pinned dependency versions
├── CITATION.cff
├── LICENSE
└── README.md
```

## How to cite

> **TODO — paper in submission.** A BibTeX entry, DOI link, and Zenodo
> archive link will appear here once the data paper is accepted.

For now, please reference the AguaTrack dataset and this tutorial
repository by URL.

## License

- **Code** in this repository: [MIT](LICENSE)
- **Data** (the AguaTrack dataset on HuggingFace): CC-BY 4.0

## Dataset documentation

See [`docs/dataset_zarr_SA_0_25_deg_daily.md`](docs/dataset_zarr_SA_0_25_deg_daily.md)
for the daily zarr schema, chunk layout, and access-pattern guidance.
