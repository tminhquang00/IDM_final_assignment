# Paper Recommendation Model

This project demonstrates the process of training a paper recommendation model using features such as Title, Abstract, Keywords, and Scopes (TAKS). It includes data preparation, feature selection, tokenization, and the creation of data loaders. The model architecture consists of a sentence embedder and a classifier that incorporates external features (Aims & Scopes). The training loop involves optimizing the model using AdamW optimizer and evaluating its performance on validation data. Finally, the notebook includes testing the model on a separate test dataset and reporting the final results.

**Presenation link:** https://rmiteduau-my.sharepoint.com/:v:/g/personal/s3409306_rmit_edu_vn/EZ89SnprCX9Ml5mFnvoVlloBZol5qWBU3hNK8veDv-GZmg 

## Project Structure
- `data/`: Contains the datasets required for training and testing the model.
- `notebooks/`: Includes Jupyter notebooks for fine-tuning the embedding model and training the recommendation model.
- `scripts/`: Contains scripts for preprocessing the data.
- `checkpoint/`: Stores model checkpoints during training.
- `requirements.txt`: Lists the dependencies needed to run the project.
- `README.md`: Provides an overview and instructions for the project.
```
├── data/
│   ├── train_pairs.csv
│   └── ...
├── notebooks/
│   ├── finetuning_embedding_model.ipynb
│   ├── training_recommendation_model_TAK.ipynb
│   └── training_recommendation_model_TAKS.ipynb
├── scripts/
│   └── preprocess.py
├── checkpoint/
│   └── ...
├── requirements.txt
└── README.md
```

## Notebooks

- `finetuning_embedding_model.ipynb`: Notebook for fine-tuning the embedding model.
- `training_recommendation_model_TAK.ipynb`: Notebook for training the recommendation model using TAK features.
- `training_recommendation_model_TAKS.ipynb`: Notebook for training the recommendation model using TAKS features.

## Scripts

- `preprocess.py`: Script for preprocessing the data.

## Data

- `data/`: Directory containing the datasets for different subjects.
- `train_pairs.csv`: CSV file containing training pairs.

## Checkpoints

- `checkpoint/`: Directory containing model checkpoints.

## Requirements

Install the required packages using:

```sh
pip install -r requirements.txt
```
## Usage

To use the paper recommendation model, follow these steps:

1. **Download the data and checkpoint from this link**:
    - Download link for pretrain model: https://drive.google.com/drive/folders/1AR4LosEjV8ygVNjcOwhch5QqdtyBxe29?usp=drive_link
    - Download link for dataset : https://drive.google.com/drive/folders/1-5iDfFrC63EQmMsGcJ248sAttWftD1_F?usp=drive_link 

2. **Preprocess the data**:
    Run the `preprocess.py` script to preprocess the data and generate the necessary files for training.

    ```sh
    python preprocess.py
    ```

3. **Fine-tune the embedding model**:
    Open and run the `finetuning_embedding_model.ipynb` notebook to fine-tune the embedding model.

4. **Train the recommendation model**:
    Open and run the `training_recommendation_model_TAK.ipynb` or `training_recommendation_model_TAKS.ipynb` notebook to train the recommendation model using the desired features.

5. **Evaluate the model**:
    Evaluate the model's performance on the validation data included in the notebooks.

6. **Test the model**:
    Test the model on a separate test dataset and report the final results as demonstrated in the notebooks.
