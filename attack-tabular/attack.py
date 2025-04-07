import json
import logging
from typing import Dict
import os

from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import numpy as np
from art.estimators.classification import PyTorchClassifier
from tqdm import tqdm

from src.attacks.cafa import CaFA
from src.constraints.constraint_projector import ConstraintProjector
from src.constraints.dcs.utilize_dcs import DCsConstrainer
from src.constraints.utils import evaluate_soundness_and_completeness
from src.models.utils import load_trained_model
from src.utils import evaluate_crafted_samples
from src.datasets.load_tabular_data import TabularDataset
from src.constraints.dcs.mine_dcs import mine_dcs
from src.models.mlp import grid_search_hyperparameters, train

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info(f"Used config: {OmegaConf.to_yaml(cfg)}")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # 1. Process data:
    tab_dataset = TabularDataset(**cfg.data.params)
    trainset, devset = tab_dataset.get_train_dev_sets(dev_set_proportion=0.15)

    # 2. Load model; optionally, re-train before:
    if cfg.ml_model.perform_training or cfg.ml_model.perform_grid_search_hparams:
        best_hparams = cfg.ml_model.default_hparams
        if cfg.ml_model.perform_grid_search_hparams:
            best_hparams = grid_search_hyperparameters(trainset=trainset,
                                                       testset=devset,
                                                       tab_dataset=tab_dataset)
        train(best_hparams, trainset=trainset, testset=devset, tab_dataset=tab_dataset,
              model_artifact_path=cfg.ml_model.model_artifact_path)
    model = load_trained_model(cfg.ml_model.model_artifact_path, model_type=cfg.ml_model.model_type)

    # 3. Wrap the model to ART classifier, for executing the attack:
    classifier = PyTorchClassifier(
        model=model,
        loss=lambda output, target: torch.functional.F.cross_entropy(output, target.long()),
        input_shape=tab_dataset.n_features,
        nb_classes=tab_dataset.n_classes,
    )
    eval_params = dict(classifier=classifier, tab_dataset=tab_dataset)

    # 4. Load constraints; Optionally, mine them before:
    if 'constraints' in cfg and cfg.constraints:
        mining_source_params = cfg.data.params.copy()
        mining_source_params['encoding_method'] = None  # we set default (label-) encoding for constraint mining
        tab_dcs_dataset = TabularDataset(**mining_source_params)
        mining_source = tab_dcs_dataset.X_train_df

        # [Optionally] Mine the DCs:
        mine_dcs(
            x_mine_source_df=mining_source,
            x_dcs_col_names=tab_dcs_dataset.x_dcs_col_names,
            **cfg.constraints.mining_params
        )

        logger.info("Initializing constraint set and projector for the attack.")
        # Initialize the DCs constrainer:
        constrainer = DCsConstrainer(
            x_tuples_df=mining_source,
            **tab_dcs_dataset.structure_constraints,
            **cfg.constraints.constrainer_params
        )

        # Initialize the generic constraints projector
        projector = ConstraintProjector(
            constrainer=constrainer,
            **cfg.constraints.projector_params
        )

        if cfg.perform_constraints_soundness_evaluation:
            # Evaluate the Soundness and Completeness of the DCs:
            logger.info("Evaluating the quality of the DCs.")
            evaluate_soundness_and_completeness(
                dataset_name=cfg.data.name,
                samples_to_eval=tab_dcs_dataset.X_test[:1500],
                idx_to_feature_name=tab_dcs_dataset.feature_names,
                constrainer=constrainer,
            )

        eval_params.update(dict(constrainer=constrainer, tab_dataset_constrainer=tab_dcs_dataset))

    # 5. Evaluate before the attack:
    X, y = tab_dataset.X_test[:cfg.n_samples_to_attack], tab_dataset.y_test[:cfg.n_samples_to_attack]
    if cfg.data_split_to_attack == 'train':
        X, y = tab_dataset.X_train[:cfg.n_samples_to_attack], tab_dataset.y_train[:cfg.n_samples_to_attack]
    evaluations: Dict[str, Dict[str, float]] = {}

    evaluations['before-attack'] = evaluate_crafted_samples(X_adv=X, X_orig=X, y=y, **eval_params)
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "Y.npy"), y)
    logger.info(f"before-attack: {evaluations['before-attack']}")

    # 4. Attack:
    X_adv = None
    if cfg.perform_attack:
        logger.info("Executing CaFA attack.")
        attack = CaFA(estimator=classifier,
                      **tab_dataset.structure_constraints,
                      **cfg.attack)
        X_adv = attack.generate(x=X, y=y)

        evaluations['after-cafa'] = evaluate_crafted_samples(X_adv=X_adv, X_orig=X, y=y, **eval_params)
        np.save(os.path.join(output_dir, "X_adv.npy"), X_adv)
        logger.info(f"after-cafa: {evaluations['after-cafa']}")

    # 5. Project
    if 'constraints' in cfg and cfg.constraints and cfg.perform_projection and X_adv is not None:
        logger.info("Executing projection of the crafted samples onto the constrained space.")
        # collect sample projected to numpy array
        X_adv_proj = []

        for x_orig, x_adv in tqdm(zip(X, X_adv), desc="Projecting crafted samples onto constraints."):  # for validation

            # 5.A. Transform sample to the format of the DCs dataset
            sample_orig = TabularDataset.cast_sample_format(x_orig, from_dataset=tab_dataset,
                                                            to_dataset=tab_dcs_dataset)
            sample_adv = TabularDataset.cast_sample_format(x_adv, from_dataset=tab_dataset, to_dataset=tab_dcs_dataset)

            # 5.B. Project
            is_succ, sample_projected = projector.project(sample_adv, sample_original=sample_orig)

            # 5.C. Transform back to the format of the model input
            x_adv_proj = TabularDataset.cast_sample_format(sample_projected, from_dataset=tab_dcs_dataset,
                                                           to_dataset=tab_dataset)
            X_adv_proj.append(x_adv_proj)

        X_adv_proj = np.array(X_adv_proj)

        evaluations['after-cafa-projection'] = evaluate_crafted_samples(X_adv=X_adv_proj, X_orig=X, y=y, **eval_params)
        np.save(os.path.join(output_dir, "X_adv_proj.npy"), X_adv_proj)
        logger.info(f"after-projection: {evaluations['after-cafa-projection']}")

    # 6. Log & save evaluations:
    logger.info(f"Evaluations: {evaluations}")
    with open(os.path.join(output_dir, "evaluations.json"), "w") as f:
        json.dump(evaluations, f, indent=4)
    logger.info(f"Finished run. results saved in {output_dir}")


if __name__ == "__main__":
    main()
