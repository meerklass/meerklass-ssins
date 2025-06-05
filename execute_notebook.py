#!/usr/bin/env python3
import subprocess
from pathlib import Path
import click


CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 100}


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-n",
    "--notebook-file",
    required=True,
    type=click.Path(exists=True, resolve_path=True, path_type=Path),
    help="Path to the template notebook to run",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(exists=True, resolve_path=True, path_type=Path),
    default=Path("./"),
    show_default=True,
    help="Output directory",
)
@click.option(
    "-b",
    "--block-number",
    type=str,
    help="Block number of the observation (CB-ID), usually a 10-digit number. ",
)
@click.option(
    "-p", "--pol", type=click.Choice(["h", "v"]), help="Polarisation to run, h or v."
)
@click.option(
    "--local/--slurm",
    default=False,
    show_default=True,
    help="Whether to run papermill locally using the current Python environment, "
    "or make and submit a SLURM script.",
)
@click.option("--dry-run", is_flag=True, help="Make and save SLURM script and exit.")
@click.option(
    "--runtime",
    default="00:05:00",
    show_default=True,
    help="Time string for the slurm job",
)
@click.option(
    "--mem",
    default=4,
    show_default=True,
    help="Memory required in GB for the slurm job",
)
@click.option(
    "--cpus-per-task",
    default=1,
    show_default=True,
    help="cpus per task for the slurm job",
)
def run_notebook(
    notebook_file,
    output_dir,
    block_number,
    pol,
    local,
    dry_run,
    runtime,
    mem,
    cpus_per_task,
):
    """Execute SSINS notebook via papermill CLI.

    The executed notebook will be saved as:
        `output_dir/notebook_name-block_number-pol.ipynb`

    When running with SLURM, katcali environment will be sourced from miniforge3
    installation (see code or output sbatch script).
    """
    output_file = output_dir / f"{notebook_file.stem}-{block_number}-{pol}.ipynb"

    command = (
        f"papermill -k python3 -p block_number {block_number} -p pol {pol} "
        + f"{notebook_file.as_posix()} {output_file.as_posix()}"
    )

    if local:
        subprocess.run(command.split(), check=True)
    else:
        sbatch = f"""#!/bin/bash

    #SBATCH --job-name={notebook_file.stem}-{block_number}
    #SBATCH --output=logs/{notebook_file.stem}-{block_number}-%j.log
    #SBATCH --partition=Main
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task={cpus_per_task}
    #SBATCH --mem={mem}GB
    #SBATCH --time={runtime}

    # Set environment variables for Numpy threading
    export MKL_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
    export OPENBLAS_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
    export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}

    # Activate conda and the katcali Python environment
    source ~/miniforge3/bin/activate
    conda activate katcali
    echo "Using Python: $(which python)"

    # Executing papermill
    {command}
    """
        sbatch_file = Path("./_execute_notebook.sbatch").resolve()
        with open(sbatch_file, "w") as fl:
            print(f"Generating an sbatch script, saving it to {sbatch_file}:")
            print("-------BEGINING OF SBATCH-------")
            print(sbatch)
            print("-------END OF SBATCH-------")
            fl.write(sbatch)

        if not dry_run:
            subprocess.run(["sbatch", "_run_sanity_check.sbatch"], check=True)


if __name__ == "__main__":
    run_notebook()
