# meerklass-ssins
Implementation of [SSINS](https://github.com/mwilensky768/SSINS) RFI flagging algorithm ([Wilensky et al. 2019](https://arxiv.org/abs/1906.01093)) to MeerKLASS autocorrelation data.

The current analysis have been developed primarily as Jupyter notebooks templates. We then use [papermill](https://papermill.readthedocs.io/en/latest/index.html) to execute the notebook given different parameters.

## Setup
Clone the repository. Then, to run the notebooks, create a python environment with the included `environment.yaml`

```
conda env create -f environment.yaml
```

This will create a environment named `katcali` with all the neccessary modules to run the notebook.

## Executing notebook templates
All templates for analysis notebooks are in the `notebook_templates` directory.
`mkssins.py` is a Python module file containing neccesary functions for the notebooks.

To execute the notebooks use the `execute_notebook.py` CLI script.

```bash
$ python execute_notebook.py -h
Usage: execute_notebook.py [OPTIONS]

  Execute SSINS notebook via papermill CLI.

  The executed notebook will be saved as:     `output_dir/notebook_name-block_number-pol.ipynb`

  When running with SLURM, katcali environment will be sourced from miniforge3 installation (see
  code or output sbatch script).

Options:
  -n, --notebook-file PATH  Path to the template notebook to run  [required]
  -o, --output-dir PATH     Output directory  [default: .]
  -b, --block-number TEXT   Block number of the observation (CB-ID), usually a 10-digit number.
  -p, --pol [h|v]           Polarisation to run, h or v.
  --local / --slurm         Whether to run papermill locally using the current Python environment,
                            or make and submit a SLURM script.  [default: slurm]
  --dry-run                 Make and save SLURM script and exit.
  --runtime TEXT            Time string for the slurm job  [default: 00:05:00]
  --mem INTEGER             Memory required in GB for the slurm job  [default: 4]
  --cpus-per-task INTEGER   cpus per task for the slurm job  [default: 1]
  -h, --help                Show this message and exit.
```

