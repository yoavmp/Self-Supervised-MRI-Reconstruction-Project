    # Self-Supervised MRI Reconstruction on fastMRI. (Replication and Extension of MICCAI 2021 Parallel Self-Supervised Framework)
    ### Abstract
    This repository contains a replication and comparative study of the MICCAI 2021 paper:

    `Self-Supervised Learning for MRI Reconstruction with a Parallel Network Training Framework. (MICCAI 2021)`

    The original implementation (designed for the IXI dataset) was adapted to the fastMRI single-coil knee dataset, and extended to include additional comparison baselines:

    - Parallel Self-Supervised (original MICCAI framework)

    - SSDU (Single-network self-supervised variant)

    - Supervised ISTA-Net+

    All models are trained and evaluated under identical data splits and masking conditions. Controlled slice-wise comparisons are implemented to ensure fair visual and quantitative evaluation.




    <p align="center">
    (a) Training phase
    </p>

    ![network_test](/figs/network_test.png)
    <p align="center">
    (b) Test phase
    </p>

    <p align="center">
    The pipeline of our proposed framework for self-supervised MRI reconstruction.
    </p>

    ## How to use
    This project is conducted on Ubuntu (64-bit) utilizing NVIDIA GPUs (tested with RTX series). All experiments were performed using single-GPU training due to our capabilities (no distributed data parallel).
    Below we explain how to use this code to perform self-supervised and supervised MRI reconstruction on fastMRI.

    ### Clone repository
    ```bash
    git clone https://github.com/yoavmp/Self-Supervised-MRI-Reconstruction-Project.git
    cd Self-Supervised-MRI-Reconstruction-Project
    ```

    ### Download dataset
    Download the [fastMRI single-coil knee dataset](https://fastmri.med.nyu.edu/) and divide them into three disjoint parts: training, validation, and test. Update the dataset paths in the command-line arguments when running **main.py**.

    ### Install dependencies
    This code is tested with Python 3.11. We recommend using a clean virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

    ### Training phase
    Enter the project directory and run the following commands.
    #### Parallel Self-Supervised (MICCAI 2021)
    ```bash
    python main.py \
    --mode train \
    --exp-name fastmri_parallel \
    --train-path <train_path> \
    --val-path <val_path> \
    --u-mask-path <mask> \
    --s-mask-up-path <mask> \
    --s-mask-down-path <mask>
    ```
    #### SSDU (Single Network Self-Supervised)
    ```bash
    python main.py \
    --mode train \
    --ssdu \
    --ssdu-theta-frac 0.5 \
    --exp-name fastmri_ssdu \
    --train-path <train_path> \
    --val-path <val_path>
    ```
    #### Supervised ISTA-Net+
    ```bash
    python main.py \
    --mode train \
    --supervised-ista \
    --exp-name fastmri_ista_sup \
    --train-path <train_path> \
    --val-path <val_path>
    ```

    ### Test phase
    Enter the path of the project and run the following scripts to test the saved model.
    ```bash
    python main.py \
    --mode test \
    --pretrained \
    --model-save-path <model_directory>
    ```

    ## Acknowledgments
    [1]. Original MICCAI 2021 implementation:
    https://github.com/chenhu96/Self-Supervised-MRI-Reconstruction

    [2]. fastMRI Dataset:
    https://fastmri.med.nyu.edu/

    [3]. Zhang, J., Ghanem, B.: ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing. CVPR 2018.
