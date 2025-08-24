# 🧭 How Much Position Information Do Mix-FFN Layers Encode in Diffusion Transformers?

This repository contains the code and experiments for analyzing how **Mix-FFN layers encode positional information** in diffusion transformers.  
Our study investigates whether these layers carry positional cues beyond what is provided by attention mechanisms, using **probing** and **ablation** experiments.

---

## 🧪 Experiment 1 – Training Probes on Latent Activations

1. **Clone and set up the environment**:

    ```bash
    conda env create -f probing_env.yml
    conda activate sana  # or your chosen environment name
    ```

2. **If you are using a SLURM system**, please fill in the `MAIL_USER` and `CONDA_ENV` variables in `py-sbatch.sh`.

3. **Run the experiment commands**:  
   All commands used in this experiment are listed in `commands_probing.txt`.  
   Run them one by one.  
   If you are **not** using SLURM, replace each `./py-sbatch.sh` with `python`.

   You can distribute commands across jobs **if they belong to the same phase** (e.g., collecting activations or training probes).

---

## 🧪 Experiment 2 – Ablation Study

1. **Follow the instructions** [in this GitHub issue](https://github.com/djghosh13/geneval/issues/12) to create the environment required to run the GenEval benchmark.

2. **If you are using a SLURM system**, fill in the `MAIL_USER` and `CONDA_ENV` variables in `py-sbatch.sh`.

3. **Run the ablation experiment commands**:  
   All commands used in this experiment are listed in `commands_ablation.txt`.  
   If you are **not** using SLURM, replace each `./py-sbatch.sh` with `python`.

   You can distribute all commands **except the first part**, which involves collecting the three types of mean activation.

