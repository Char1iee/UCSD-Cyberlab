from typing import Optional

import numpy as np
from art.estimators import NeuralNetworkMixin

from src.attacks.cafa import CaFA
from src.constraints.dcs.utilize_dcs import Constrainer
from src.datasets.load_tabular_data import TabularDataset


def evaluate_crafted_samples(
        X_adv: np.ndarray,
        X_orig: np.ndarray,
        y: np.ndarray,
        classifier: NeuralNetworkMixin,

        # Tabular data info:
        tab_dataset: TabularDataset,

        # Optional feasibility constraints:
        tab_dataset_constrainer: Optional[TabularDataset] = None,
        constrainer: Optional[Constrainer] = None,
):
    """
    Evaluates the crafted adversarial samples wrp to the given classifier.
    :param X_adv: Adversarial samples crafted by the attack, in format of model's input.
    :param X_orig: Samples on which the attack was applied, in format of model's input.
    :param y: labels of the samples.
    :param classifier: the targeted classifier.
    :param tab_dataset: the tabular dataset object of the model's input data.
    :param constrainer: constraint object, which checks the feasibility of the crafted samples.
    :param tab_dataset_constrainer: the tabular dataset object of the `constrainer` input data.
    :return:
    """
    # Evaluate misclassification
    is_misclassified = classifier.predict(X_adv).argmax(axis=1) != y

    # Evaluate compliance with constraints
    is_comp = np.ones(len(X_adv), dtype=bool)  # defaults to compliance of all samples
    if constrainer is not None:
        is_comp = np.zeros(len(X_adv), dtype=bool)
        for idx, (x_adv, x_orig) in enumerate(zip(X_adv, X_orig)):
            sample_adv = TabularDataset.cast_sample_format(x_adv,
                                                           from_dataset=tab_dataset,
                                                           to_dataset=tab_dataset_constrainer)
            sample_orig = TabularDataset.cast_sample_format(x_orig,
                                                            from_dataset=tab_dataset,
                                                            to_dataset=tab_dataset_constrainer)
            assert np.all(x_orig ==
                          TabularDataset.cast_sample_format(sample_orig, from_dataset=tab_dataset_constrainer,
                                                            to_dataset=tab_dataset))
            assert np.all(sample_orig ==
                          TabularDataset.cast_sample_format(x_orig, from_dataset=tab_dataset,
                                                            to_dataset=tab_dataset_constrainer))
            is_comp[idx] = constrainer.check_sat(sample_adv, sample_original=sample_orig)

    # Evaluate 'cost' metrics
    l0_costs = CaFA.calc_l0_cost(X_orig, X_adv)
    stand_linf_costs = CaFA.calc_standard_linf_cost(
        X_orig, X_adv,
        standard_factors=tab_dataset.standard_factors,
        relevant_indices=tab_dataset.ordinal_indices.tolist() + tab_dataset.cont_indices.tolist())

    assert len(is_misclassified) == len(is_comp) == len(l0_costs) == len(stand_linf_costs) == len(X_adv)

    return {
        # Attack success:
        'is_misclassified_rate': is_misclassified.mean(),
        'is_comp_rate': is_comp.mean(),
        'is_mis_and_comp_rate': (is_misclassified & is_comp).mean(),

        # Costs:
        #  - L0:
        'l0_costs_mean': l0_costs.mean(),
        'l0_costs_on_mis_mean': l0_costs[is_misclassified].mean(),
        'l0_costs_on_mis_and_comp_mean': l0_costs[is_misclassified & is_comp].mean(),

        #  - Standardized-linf
        'stand_linf_costs_mean': stand_linf_costs.mean(),
        'stand_linf_costs_on_mis_mean': stand_linf_costs[is_misclassified].mean(),
        'stand_linfcosts_costs_on_mis_and_comp_mean': stand_linf_costs[is_misclassified & is_comp].mean(),

    }

