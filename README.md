# UCL Cluster Example

This repository is a minimal template for training an **MNIST** classifier on the UCL High-Performance Computing (HPC) cluster using **PyTorch Lightning**.

---

## Contents

* **`main.py`** — training script that defines a simple LightningModule and DataModule for MNIST.  
* **`requirements.txt`** — Python dependencies to recreate the exact environment.  
* **`submit_job.sh`** — PBS/Torque submission script that launches `main.py` on the cluster.

---

## Prerequisites

* A recent **Linux** distribution (tested on Ubuntu 20.04 and RHEL 8).  
* An account on the **UCL HPC** system with access to `qsub` / PBS.  
* **Miniconda ≥ 4.12** for environment management (preferred over plain `pip` because it is more user-friendly and environments can live outside your limited home quota).

> **Why Miniconda instead of pip?**  
> Although the official HPC docs recommend `pip` to conserve space in users’ home directories, Conda environments can be created in your scratch or project space (via the `--prefix` flag), so quota limits are rarely an issue.

---

## Installation

1. **Install Miniconda**

   ~~~bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh      # accept licence, choose install dir
   ~~~

2. **Create and activate an isolated environment**

   ~~~bash
   conda create -y -p $HOME/conda/ucl-cluster-example python=3.10
   conda activate $HOME/conda/ucl-cluster-example
   ~~~

3. **Install required libraries**

   ~~~bash
   pip install -r requirements.txt
   ~~~

---

## Submitting a Job

1. **Edit `submit_job.sh`**  
   Adjust the PBS resource headers (CPUs, GPUs, memory, wall-time, log paths) and required `module load` lines to match your project.

2. **Submit the script**

   ~~~bash
   qsub submit_job.sh
   ~~~

3. **Monitor progress**

   ~~~bash
   qstat -u $USER
   ~~~

---

## Interactive Debugging

Request an interactive node, then run the script directly:

~~~bash
qsub -I -l select=1:ncpus=4:ngpus=1:mem=16gb:walltime=02:00:00
# Once the shell starts:
conda activate /path/to/your/env
python main.py
~~~

---

## Tips

* **Storage quotas** — create Conda environments in `/scratch` or a project directory if your home quota is tight:  
  `conda create -p /scratch/$USER/envs/ucl-cluster-example python=3.10`.
* **Extending to other datasets** — swap the Lightning `DataModule` in `main.py`; the training loop itself is dataset-agnostic.

---

## Author

Hyunmin Kim