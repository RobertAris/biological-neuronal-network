# SLURM Guide P&S

## Prerequisites

- You need your Snowflake account and password which you received via mail from us.
- Your account name follows this format: `bnn_[nn]fs26`, where `[nn]` is your assigned number (01-16).
- Ensure you are within the ETH network or connected via VPN.
- You must connect to the cluster through an intermediate PC: `tardis-b[mm]`, where `[mm]` matches your `[nn]`.
- Download the following scripts from Moodle:
  - `run_job.sh`
  - `install_conda.sh`

## Setup

1. Open a terminal.
2. Move the Conda installation script to your Snowflake home folder. The command “scp“ stands for “secure copy protocol“. In a terminal, you can copy/paste using ctrl+shift+c/v on Linux and Windows machines. Note that in a shell, space is used as the delimiter, hence we use ““ to ensure that your path is read as one:

    ```bash
    scp "/full/path/to/your/script" "bnn_[nn]fs26@tardis-b[mm].ee.ethz.ch:/home/bnn_[nn]fs26/"
    ```

3. Enter your Snowflake password when prompted.
4. Connect to the server through “ssh“ (secure shell):

    ```bash
    ssh bnn_[nn]fs26@tardis-b[mm].ee.ethz.ch
    ```

5. Make the script executable. “chmod“ stands for “change mode“ and the “+x“ adds the execution rights:

    ```bash
    chmod +x ./install_conda.sh
    ```

6. Run the script. The path argument gives the save location where conda will be installed. We use a network scratch drive for this:

    ```bash
    ./install_conda.sh /usr/itetnas04/data-scratch-01/bnn_[nn]fs26/data
    ```

7. Set up the Conda alias (adjust the path accordingly, as printed after installation). This will enable you to call the command “conda“ in your terminal:

    ```bash
    echo '[[ -f /usr/itetnas04/data-scratch-01/bnn_[nn]fs26/data/conda/bin/conda ]] && eval "$(/usr/itetnas04/data-scratch-01/bnn_[nn]fs26/data/conda/bin/conda shell.bash hook)"' >> /home/bnn_[nn]fs26/.bashrc
    ```

8. Now we link SLURM (Simple Linux Utility for Resource Management) to an alias:

    ```bash
    echo 'export SLURM_CONF=/home/sladmsnow/slurm/slurm.conf' >> ~/.bashrc
    ```

9. Enable the alias for both conda and slurm by reloading the bashrc. Afterwards this will no longer be necessary, as with each new login, the bashrc gets activated automatically:

    ```bash
    source ~/.bashrc
    ```

10. Create a new Conda environment:

    ```bash
    conda create --name my_env python=3.10
    ```

11. Confirm installation by typing `y` when prompted.

12. Install the machine learning library with conda:
    ```bash
    conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 "mkl<2025.0.0" -c pytorch -c nvidia
    ```

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

You will always have to connect to the intermediate PC:

```bash
ssh bnn_[nn]fs26@tardis-b[mm].ee.ethz.ch
```

### Submitting Jobs via SLURM

To use the GPU cluster, submit batch jobs. A pre-configured script is available for Jupyter Notebooks. Before submitting:

- Fill in the required fields (log file paths, error output paths, Jupyter Notebook port [5900-5999], and output print paths).
- Upload the script using the same method as the Conda installation script.
- Pro-Trick: A Linux shell comes with the command line editor “nano“. You can modify your scripts in your homefolder by using “nano your_script.sh“

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
cat [outputfile]
```

or

```bash
tail -f [outputfile]
```

For more details on SLURM scheduling, see [ETH SLURM Guide](https://computing.ee.ethz.ch/Services/SLURM).

### Exercise Submssion Workflow

1. Update your fork on GitLab using the **"Update fork"** button.

2. Pull the latest changes:

   ~~~bash
   git pull origin main
   ~~~

3. Complete the notebook `NN/exercise.ipynb`  
   <!-- Run all cells before committing. Do NOT clear outputs. -->

4. Stage your changes:

   ~~~bash
   git add NN/exercise.ipynb
   ~~~

5. Commit:

   ~~~bash
   git commit -m "Submission for NN"
   ~~~

6. Push to your fork:

   ~~~bash
   git push origin main
   ~~~

7. Send us an email with the link to your repo and which exercise you solved.