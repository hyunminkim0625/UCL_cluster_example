# UCL Cluster Example

A minimal template for training an **MNIST** classifier on the UCL High-Performance Computing (HPC) cluster.

---

## Contents

- **`main.py`** — PyTorch Lightning training script  
- **`requirements.txt`** — Python dependencies  
- **`submit_job.sh`** — job-submission script  

---

## Installation

1. **Install Miniconda**  

        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash Miniconda3-latest-Linux-x86_64.sh   # accept licence, choose install dir

2. **Create and activate an environment**

        conda create -n ucl-mnist python=3.10
        conda activate ucl-mnist

3. **Install dependencies**

        pip install -r requirements.txt

---

## Submitting a Job

1. **Edit `submit_job.sh`** — adjust CPUs, GPUs, memory, wall-time, log paths...
2. **Submit**

        qsub submit_job.sh

---

## For Further Usage

For more advanced usage and examples, please refer to [Using the CMIC Cluster](https://www.notion.so/keane-ai-group/Using-the-CMIC-Cluster-851c9ae24f0249608d0d814be580800a?source=copy_link).