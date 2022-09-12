import logging
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Optional

import mlflow

import ray
from ray.data.context import DatasetContext
from ray.train.sklearn import SklearnTrainer
from ray.data import Dataset
from ray.air.config import ScalingConfig
from ray.train.sklearn import SklearnCheckpoint, SklearnPredictor
from ray.train.batch_predictor import BatchPredictor
from ray import workflow

from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn import metrics


FEATURE_COLS = ['ingredients_list']
LABEL_COLUMN = 'nova_group'


def _set_mlflow_run(run_id: str):
    if not mlflow.active_run():
        mlflow.start_run(run_id)


@ray.remote
def start_tracking(experiment_name: str) -> str:
    """
    Start MLFlow run and returns its ID
    """
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run().info.run_id


@ray.remote
def train_model(train_data_path: str, run_id: str):
    log = logging.getLogger(__name__)
    # there is probably better (more clean) way to deal with MLFlow run
    # rather than passing run_id to each workflow step
    _set_mlflow_run(run_id)
    # read dataset
    ds = ray.data.read_parquet(train_data_path)
    ctx = DatasetContext.get_current()
    ctx.enable_tensor_extension_casting = False
    # TODO
    ds = ds.repartition(5)
    # define model and its preprocessing
    df_converter = FunctionTransformer(lambda X: X[FEATURE_COLS].to_dict(orient='records'))
    ingredient_encoder = DictVectorizer()
    nb_clf = BernoulliNB(binarize=None)
    sk_pipe = Pipeline([
        ('df_converter', df_converter),
        ('encoder', ingredient_encoder),
        ('clf', nb_clf)
    ])
    trainer = SklearnTrainer(
        estimator = sk_pipe,
        datasets = {
            'train' : ds,
        },
        label_column = LABEL_COLUMN,
        cv = 5,
        parallelize_cv = True,
        scaling_config = ScalingConfig(trainer_resources = {'CPU' : 5})
    )
    train_result = trainer.fit()
    log.info("Training metrics:\n%s", train_result.metrics)
    #
    checkpoint = SklearnCheckpoint.from_checkpoint(train_result.checkpoint)
    model_info = mlflow.sklearn.log_model(
        checkpoint.get_estimator(),
        artifact_path = 'model'
    )
    return model_info.model_uri


@ray.remote
def predict_on_test(
        model_uri: str,
        test_data_path: str,
        output_path_base: str,
        run_id: str):
    _set_mlflow_run(run_id)
    #
    input_ds = ray.data.read_parquet(test_data_path).repartition(10)
    ctx = DatasetContext.get_current()
    ctx.enable_tensor_extension_casting = False
    input_ds = input_ds.drop_columns([LABEL_COLUMN])
    # restore model
    sk_pipe = mlflow.sklearn.load_model(model_uri)
    with TemporaryDirectory() as tmpdir:
        model_checkpoint = SklearnCheckpoint.from_estimator(sk_pipe, path=tmpdir)
        predictor = BatchPredictor(model_checkpoint, SklearnPredictor)
    # apply model
    model_output_ds = predictor.predict(input_ds)
    # merge with input data
    result_ds = ray.data.from_arrow_refs(input_ds.to_arrow_refs()).zip(
        ray.data.from_arrow_refs(model_output_ds.to_arrow_refs()))
    output_path = str(Path(output_path_base) / run_id / 'test-predictions')
    result_ds.write_parquet(output_path)
    return output_path


@ray.remote
def evaluate(
        ground_truth_path: str,
        predictions_path: str,
        run_id: str):
    _set_mlflow_run(run_id)
    #
    truth_ds = ray.data.read_parquet(ground_truth_path).repartition(10)
    predicted_ds = ray.data.read_parquet(predictions_path).repartition(10)
    eval_ds = truth_ds.drop_columns(['product_name', 'ingredients_list'])\
        .zip(predicted_ds.drop_columns(['product_name', 'ingredients_list']))
    eval_df = eval_ds.to_pandas(eval_ds.count())
    #
    y_true = eval_df.nova_group
    y_pred = eval_df.predictions
    mlflow.log_metrics({
        'f1_' + t[0] : t[1]
        # get F1 for each class, TODO restore labels robustly
        for t in zip([str(l) for l in range(4)], metrics.f1_score(y_true, y_pred, average=None))
    })
    mlflow.log_metrics({
        'accuracy' : metrics.accuracy_score(y_true, y_pred),
        'f1_micro' : metrics.f1_score(y_true, y_pred, average='micro'),
        'f1_macro' : metrics.f1_score(y_true, y_pred, average='macro')
    })
    return run_id


@ray.remote
def register_model(model_uri, model_name, run_id):
    _set_mlflow_run(run_id)
    #
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    mlflow.end_run()
    return model_version


def build_pipeline(
        train_data_path: str, 
        test_data_path: str, 
        experiment_name: str,
        output_path_base: str):
    # set MLFlow tracking URI via environment variables
    run_id = start_tracking.bind(experiment_name)
    model_uri = train_model.bind(train_data_path, run_id)
    test_predictions_path = predict_on_test.bind(
        model_uri, test_data_path, output_path_base, run_id)
    run_id = evaluate.bind(test_data_path, test_predictions_path, run_id)
    # TODO experiment name is used as a model name
    model_version = register_model.bind(model_uri, experiment_name, run_id)
    return model_version


def run_pipeline(
        train_data_path: str, 
        test_data_path: str, 
        experiment_name: str,
        output_path_base: str,
        workflow_id: Optional[str] = None):
    model_version_node = build_pipeline(
        train_data_path, test_data_path, experiment_name, output_path_base)

    model_version = workflow.run(model_version_node, workflow_id=workflow_id)
    print(model_version)


if __name__ == '__main__':
    import fire

    fire.Fire(run_pipeline)