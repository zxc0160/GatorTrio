# GatorTrio: Topology-Refined Tri-View Graph Learning for Spatial Domain Identification in Spatial Transcriptomics
![model](https://github.com/Gator-Group/GatorTrio/blob/main/GatorTrio.png)

## Requirements
- python : 3.9.12
- scanpy : 1.10.3
- sklearn : 1.1.1
- scipy : 1.9.0
- torch : 1.11.2
- torch-geometric : 2.1.0
- numpy : 1.24.4
- pandas : 2.2.3


## Project Structure

```bash
.
├── main.py            # Main training and evaluation loop
├── GatorTrio.py         # Model architecture and loss functions
├── utils.py            # Utility functions (seed setup, metrics)
├── data/              # Folder for .h5ad input files
├── saved_models/      # Folder to save trained models
└── saved_results/     # Evaluation results output
```

## Usage

### **1. Prepare your input data**

Place your **.h5ad** datasets in the `./data/` directory.

Each `.h5ad` file should contain:

* **`adata.X`** – Gene expression matrix 
* **`adata.obs`** – Spot/Cell metadata.
  The loader automatically searches typical label fields:

  ```
  ['ground_truth']
  ```

Example dataset folder:

```bash
data/
 ├── ST-H1.h5ad
 ├── HLN.h5ad
 ├── 151507.h5ad
 ...
```

---

### **2. Run training and evaluation**

Execute the main script to train and evaluate across datasets:

```bash
python main.py
```

#### Optional configuration inside `main.py`:

* `epochs`: number of training epochs per run 
* `batch_size`: number of samples per batch
* `lr`: learning rate

You can modify these using `--epochs --batch_size --lr`

---

### **3. Output files**

After training completes:

* Trained model checkpoints: `saved_models/`

  ```
  saved_models/
   ├── ST-H1_model_1_dict
   ├── HLN_model_1_dict
   ├── 151507_model_1_dict
   ...
  ```
* Evaluation results (clustering/imputation/annotation): `saved_results/`

  ```
  saved_models/
   ├── ST-H1_result.json
   ├── HLN_result.json
   ├── 151507_result.json
   ...
  ```
