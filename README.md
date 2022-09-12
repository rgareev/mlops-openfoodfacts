# MLOps Project on Open Food Facts data with Ray AIR & MLFlow

## Dataset
[Open Food Facts](https://world.openfoodfacts.org/) is a food products database made by everyone, for everyone. It contains more than 2 000 000 product description records from all around the world.

### Schema Description
There are several data dump format supported by OpenFoodFacts. In this project CSV format is currently being used. Its schema can be found [here](https://static.openfoodfacts.org/data/data-fields.txt)

### Data Sources
https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv (raw TSV)

## ML Task
There are plenty of opportunities to build different kind of ML models on this dataset. For simplicity sake we will stop on the following:
* given data about product ingredients
* predict its [NOVA group](https://world.openfoodfacts.org/nova).

## Implementation
### Dataset preparation
See [notebooks/create-dataset.ipynb](notebooks/create-dataset.ipynb) that cleans the raw TSV dataset and fix the test subset for final evaluations. The current version retains only the following data attributes:
* `code` - as a record id (i.e, dataframe index)
* `product_name` - currently used only for developer curiosity (like description), not used in model input
* `ingredients_tags` - normalized (by OFF community and rules) list of ingredient identifiers. Note that there is a [rich taxonomy](https://wiki.openfoodfacts.org/Ingredients_taxonomy) over them, but in the baseline implementation we don't utilize that at all
* `nova_group` - our target variable.

### Baseline model
Ingredients are multihot encoded using Scikit-Learn's `DictVectorizer`. Resulting feature vectors are passed to Scikit-Learn's Naive Bayes classifier implementation for [multivariate Bernoulli models](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html).

## MLOps
### Ray AI Runtime
Ray AI Runtime (AIR) is a scalable and unified toolkit for ML applications. See an introduction [here](https://docs.ray.io/en/latest/ray-air/getting-started.html).

### Project Dependency Danagement
Poetry is used, see [pyproject.toml](pyproject.toml).

### Model Training Pipeline
Ray Workflows is used to define and execute the model training pipeline, see [src/training_pipeline.py](src/training_pipeline.py).

### Data Transformation
Ray `Dataset`s is used to work with data and apply transformations for feature generation and evalution.

### Model Training & Packaging
Ray `Trainer` implementation for Scikit-Learn is used to define model training task

### Experiment Tracking & Model Registry
MLFlow Tracking and Model Registry are used.

### Model Deployment
Currently Ray's `BatchPredictor` is used to deploy the model version from MLFlow model registry and apply it in batch mode to the input. See example in `predict_on_test` step of the [training pipeline](src/training_pipeline.py).

*Ray Serve can be used for online inference, but this is in progress...*

## How to Use
### Locally
1. Have MLFlow server running, for example:
```
mlflow server --backend-store-uri sqlite:///wrk/mlflow.db --serve-artifacts --artifacts-destination wrk/mlartifacts
```
2. Setup python environment for the project. In project directory run
```
poetry install
```
3. Download raw dataset using the link from the dataset description above.
4. Prepare dataset using [create-dataset.ipynb](notebooks/create-dataset.ipynb)
5. Set `MLFLOW_TRACKING_URI` environment variable
6. Run python script with the pipeline definition:
```
poetry run python src/training_pipeline.py ARGS
```

### In AWS Cloud
*This section is in progress...*

Instructions are the same but
1. Use `ray up -y config.yaml` to start Ray cluster in an AWS account
2. Use `RAY_ADDRESS` environment variable to point to the remote Ray cluster