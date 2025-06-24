## 🐳 Universal Domain Adaptation Benchmark (Time Series) – Docker Environment

This section describes how to use the pre-built Docker image to run the **Universal Domain Adaptation Benchmark for Time Series Data Representation** in a reproducible, GPU-enabled environment.

---

## 📚 Acknowledgments

This repository (`UniDABench`) is based on the original implementation of :octocat: [AdaTime](https://github.com/emadeldeen24/AdaTime). We sincerely thank the authors for their excellent work and publicly available implementation. 

We adapted their code specifically to address the **Universal Domain Adaptation** (UDA) task for time series data, including significant modifications and extensions.

Additionally, we implemented and included six baseline methods to benchmark UDA performance:

- **UAN** ([Universal Domain Adaptation](https://openaccess.thecvf.com/content_CVPR_2019/papers/You_Universal_Domain_Adaptation_CVPR_2019_paper.pdf))
- **OVANet** ([OVANet: One-vs-All Network for Universal Domain Adaptation
](https://arxiv.org/abs/2104.03344))
- **PPOT** ([Prototypical Partial Optimal Transport for Universal Domain Adaptation
](https://arxiv.org/abs/2408.01089))
- **DANCE** ([Universal Domain Adaptation through Self Supervision
](https://arxiv.org/abs/2002.07953))
- **UniJDOT** ([Deep Joint Distribution Optimal Transport for Universal Domain Adaptation on Time Series
](https://arxiv.org/abs/2503.11217))

We acknowledge the authors of each baseline method for their valuable contributions to domain adaptation research.

---

## 📁 Datasets and Running Experiments

We use three datasets in our experiments. They can be downloaded directly from the following link:

➡️ [Datasets for UniDABench](https://drive.google.com/file/d/1DcuNYsKyzKg_Vm7u-Lm2Y6BjDdV71QCJ/view?usp=sharing)

After downloading and setting up the datasets, you can reproduce all experiments reported in our paper by simply executing:

```bash
bash run.sh
```

Ensure that your environment is properly configured before running experiments.
All required packages are listed in the requirements.txt file. 
To enhance long-term reproducibility, we also provide an NVIDIA Docker image with all necessary packages 
pre-installed (for details see the section 🐳 **Docker Environment**).

## 🚀 Running Specific Experiments and Hyperparameter Sweeps

You can run specific experiments directly by specifying the dataset, domain adaptation method, backbone model, and the number of experiment repetitions:

```bash
python main.py --dataset "HAR" --da_method "DANCE" --backbone "CNN" --num_runs 10
```

This command runs the DANCE method on the HAR dataset using a CNN backbone for 10 repetitions.
The current hyperparameters have been selected through a thorough hyperparameter search using [**Weight & Biases**](https://wandb.ai/site/) framwork.

To perform a hyperparameter sweep to identify the optimal hyperparameters, you can use:

```bash
python main_sweep.py --dataset "HAR" --da_method "DANCE" --backbone "CNN" --num_sweeps 100 --num_runs 5
```

This command executes 100 hyperparameter search sweeps for the DANCE method on the HAR dataset with a CNN backbone, repeating each configuration 5 times to ensure robustness.

There is no automatic selection of scenarios during the hyperparameter search. If you conduct your own hyperparameter search, we recommend selecting scenarios that differ from those used for the final model evaluation to prevent overfitting.

---
## 🐳 Docker Environment

The provided Docker environment ensures consistent execution of benchmarks across different machines, streamlining the setup process for GPU-enabled experiments.

---
### 📦 Docker Image

- **Image Name:** `mussarom/uda_ts_bench`  
- **Base Image:** `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04`

---

### ✅ System Requirements

Ensure your system meets the following requirements:

- **NVIDIA GPU**
- **NVIDIA GPU driver** (version **≥ 515** recommended for CUDA 11.8)
- **Docker** ([Install Docker](https://docs.docker.com/get-docker/))
- **NVIDIA Container Toolkit**

Install the NVIDIA Container Toolkit:

```bash
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

Verify your GPU and driver installation:

```bash
nvidia-smi
```

> ⚠️ **Important:** The Docker image includes CUDA and cuDNN but **not the NVIDIA driver**, which must be installed separately on your host machine.

---

### 📥 Option 1: Pull the Docker Image

You can directly pull the image from Docker Hub:

```bash
docker pull mussarom/uda_ts_bench
```

---

### 🛠️ Option 2: Build the Docker Image from Dockerfile

Alternatively, you can build the Docker image locally using the provided Dockerfile. Navigate to the directory containing the Dockerfile and run:

```bash
docker build -t mussarom/uda_ts_bench .
```

This will create the image `mussarom/uda_ts_bench` on your machine.


---

### 🚀 Run the Docker Container

Launch the container with GPU support, mounting local folders for your data and temporary workspace:

```bash
docker run --gpus all -it --rm --shm-size=200g \
  -v /path/to/your/data:/workspace/data \
  -v /path/to/tmp/dir:/workspace/tmp \
  -e TMPDIR=/workspace/tmp \
  mussarom/uda_ts_bench
```

#### 🔧 Replace these paths:

- `/path/to/your/data`: Local path containing your project or dataset.
- `/path/to/tmp/dir`: Local path for temporary files or cache.

Inside the container, these paths are available as:

- `/workspace/data`
- `/workspace/tmp`

---

### 📂 Working Inside the Container

Once inside the Docker container, you'll have access to the following environment:

- ✅ Python environment with necessary packages
- ✅ PyTorch
- ✅ CUDA 11.8 Runtime
- ✅ cuDNN

Navigate to your mounted project directory:

```bash
cd /workspace/data
```

---

### 👤 Maintainer

- **Romain Mussard**  
  Github : [RomainMsrd](https://github.com/RomainMsrd)

  Docker Hub: [mussarom](https://hub.docker.com/u/mussarom)

  Mail : [romain.mussard@univ-rouen.fr](mailto:romain.mussard@univ-rouen.fr)

---

## 🙌 Citation

If you found our work useful for your research, please consider citing our papers:

```bibtex
@misc{mussard2025universaldomainadaptationbenchmark,
      title={Universal Domain Adaptation Benchmark for Time Series Data Representation}, 
      author={Romain Mussard and Fannia Pacheco and Maxime Berar and Gilles Gasso and Paul Honeine},
      year={2025},
      eprint={2505.17899},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.17899}, 
}
```
```bibtex
@misc{mussard2025deepjointdistributionoptimal,
      title={Deep Joint Distribution Optimal Transport for Universal Domain Adaptation on Time Series}, 
      author={Romain Mussard and Fannia Pacheco and Maxime Berar and Gilles Gasso and Paul Honeine},
      year={2025},
      eprint={2503.11217},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.11217}, 
}
```
