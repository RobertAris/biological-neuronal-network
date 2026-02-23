# SLURM Guide P&S

## Prerequisites

- You need your Snowflake account and password.
- Your account name follows this format: `bnn_[nn]fs26`, where `[nn]` is your assigned number (01-16).
- Ensure you are within the ETH network or connected via VPN.
- You must connect to the cluster through an intermediate PC: `tardis-b[mm]`, where `[mm]` matches your `[nn]`.
- Download the following scripts from Moodle:
  - `run_job.sh`
  - `install_conda.sh`

## Setup

1. Open a terminal.
2. Move the Conda installation script to your Snowflake home folder:

    ```bash
    scp -r /full/path/to/your/script bnn_[nn]fs26@tardis-b[mm].ee.ethz.ch:/home/bnn_[nn]fs26/
    ```

3. Enter your Snowflake password when prompted.
4. Connect to the server:

    ```bash
    ssh bnn_[nn]fs25@tardis-b[mm].ee.ethz.ch
    ```

5. Make the script executable:

    ```bash
    chmod +x ./install_conda.sh
    ```

6. Run the script:

    ```bash
    ./install_conda.sh /usr/itetnas04/data-scratch-01/bnn_[nn]fs26/data
    ```

7. Set up the Conda alias (adjust the path accordingly, as printed after installation):

    ```bash
    echo '[[ -f /usr/itetnas04/data-scratch-01/bnn_[nn]fs26/data/conda/bin/conda ]] && eval "$(/usr/itetnas04/data-scratch-01/bnn_[nn]fs26/data/conda/bin/conda shell.bash hook)"' >> /home/bnn_[nn]fs26/.bashrc
    ```

8. Now we link slurm to an alias:

    ```bash
    echo 'export SLURM_CONF=/home/sladmsnow/slurm/slurm.conf' >> ~/.bashrc
    ```

9. Enable the alias for both conda and slurm:

    ```bash
    source ~/.bashrc
    ```

10. Create a new Conda environment:

    ```bash
    conda create --name my_env -c "nvidia/label/cuda-11.6.2" cuda-toolkit cuda cudnn python=3.10
    ```

11. Confirm installation by typing `y` when prompted.

## Git Setup

1. Fork this repository
2. Clone your fork:

    ```bash
    git clone [paste_from_clone_https]
    ```

3. Navigate to the cloned repository:

    ```bash
    cd /path/to/the/new/folder
    ```

4. Activate the Conda environment:

    ```bash
    conda activate my_env
    ```

5. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

You will alsoways have to connect to the server:

```bash
ssh bnn_[nn]fs26@tardis-b[mm].ee.ethz.ch
```

### Submitting Jobs via SLURM

To use the GPU cluster, submit batch jobs. A pre-configured script is available for Jupyter Notebooks. Before submitting:

- Fill in the required fields (log file paths, error output paths, Jupyter Notebook port [5900-5999], and output print paths).
- Upload the script using the same method as the Conda installation script.

### Running SLURM Jobs

Start a job with a 24-hour limit, don't forget to activate your conda environment:

```bash
sbatch /path/to/the/script/run_job.sh
```

Check job status:

```bash
squeue
```

As soon as you no longer need the job, cancel it:

```bash
scancel [job_id]
```

To access the Jupyter Notebook, check the error output file (e.g., `id.err`):

```bash
tail -f [outputfile]
```

For more details on SLURM scheduling, see [ETH SLURM Guide](https://computing.ee.ethz.ch/Services/SLURM).