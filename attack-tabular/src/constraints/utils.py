import logging
from typing import List
from collections import Counter

import numpy as np
from tqdm import tqdm

from src.constraints.dcs.utilize_dcs import Constrainer

logger = logging.getLogger(__name__)


def evaluate_soundness_and_completeness(
        samples_to_eval: np.ndarray,
        dataset_name: str,
        idx_to_feature_name: np.array,
        constrainer: Constrainer,
):
    """
    Algorithm for calculating the Empirical Soundness (see paper) of the given constraints.
    The Soundness is calculated by taking samples complied with the constraints, and injecting violations to them.
    We wish that these injected samples will `turn` to be not complied with the constraints, the more - the higher
    the soundness.
    :param dataset_name: The name of the dataset. Note that Soundness evaluation is dataset-dependent (since the generating the violation
                         requires golden constraints).
    :param samples_to_eval: Samples to use for evaluation. These should be in the format of the constrainer's input.
    :param idx_to_feature_name:  list of feature names, in the order of the sample.
    :param constrainer: The evaluated Constrainer object.
    :return:
    """
    total_count, turned_count = 0, 0
    total_samples_count, sat_samples_count = 0, 0  # for hard-completeness
    turned_per_violation = Counter()

    for sample in tqdm(samples_to_eval, desc="Evaluating Soundness and Completeness"):

        is_original_sample_sat = constrainer.check_sat(sample)
        sat_samples_count += int(is_original_sample_sat)
        total_samples_count += 1

        # Inject violation(s) to these samples (resulting in samples which are not in the feature-space):
        violated_samples = get_violated_samples_from_sample(sample, dataset_name=dataset_name,
                                                            idx_to_feature_name=idx_to_feature_name)
        for idx_violation, violated_sample in enumerate(violated_samples):
            is_violated_sample_sat = constrainer.check_sat(violated_sample)
            # We want to count those that were SAT and now, after adding the violation are NOT sat.
            turned_count += int(is_original_sample_sat and not is_violated_sample_sat)
            turned_per_violation[idx_violation] += int(is_original_sample_sat and not is_violated_sample_sat)
            total_count += int(is_original_sample_sat)

    soundness, completeness = turned_count / total_count, sat_samples_count / total_samples_count
    logger.info(f"Evaluated quality of constraints:")
    logger.info(f">> Soundness: {soundness * 100}%")
    logger.info(f">> Completeness: {completeness * 100}%")
    logger.info(f">> 'Turned' per violation: {turned_per_violation}")

    return soundness, completeness, turned_per_violation


def get_violated_samples_from_sample(sample: np.array,
                                     dataset_name: str,
                                     idx_to_feature_name: np.array,
                                     ) -> List[np.array]:
    """
    Generates samples that violates the golden constraints, based on the given sample.
    The golden-constraints are manually constructed constraints which are data-specific, thus the logic of generating
    the violations is also dataset specific.
    :param sample: the sample to generate violations upon.
    :param idx_to_feature_name: list of feature names, in the order of the sample.
    :param dataset_name: the name of the dataset, as it defines the violation created.
    :return: list of samples that violates the golden constraints.
    """
    def get_index_by_feature_name(feature_name: str) -> int:
        return (idx_to_feature_name == feature_name).argmax()

    violated_samples: List[np.array] = []

    if dataset_name == 'bank':
        # Craft samples that violates each of the golden constraints

        # Violation #1:
        violated_sample = sample.copy()
        violated_sample[get_index_by_feature_name('previous')] = 0
        violated_sample[get_index_by_feature_name('pdays')] = np.random.randint(300, 400)  # also out of Valiant's bin
        violated_samples.append(violated_sample)

        # Violation #2:
        violated_sample = sample.copy()
        violated_sample[get_index_by_feature_name('previous')] = np.random.randint(6, 30)  # also out of Valiant's bin
        violated_sample[get_index_by_feature_name('pdays')] = 999
        violated_samples.append(violated_sample)

        # Violation #3:
        violated_sample = sample.copy()
        # job = admin ⇒ education = secondary
        violated_sample[get_index_by_feature_name('job')] = 0  # admin.
        violated_sample[get_index_by_feature_name('education')] = 1  # primary
        violated_sample = sample.copy()
        violated_samples.append(violated_sample)

        # Violation #4:
        # job = Student ⇒ marital = Single ∧ age ≤ 35
        violated_sample[get_index_by_feature_name('job')] = 8  # student
        violated_sample[get_index_by_feature_name('marital')] = 1  # married
        violated_sample[get_index_by_feature_name('age')] = 50  # (old)  # also out of Valiant's bin
        violated_samples.append(violated_sample)

    elif dataset_name == 'phishing':
        # Violation #1 (based on df.NumNumericChars < df.UrlLength)
        violated_sample = sample.copy()
        violated_sample[get_index_by_feature_name('NumNumericChars')] = (
                violated_sample[get_index_by_feature_name('UrlLength')] + np.random.randint(0, 10)
        )
        violated_samples.append(violated_sample)

        # Violation #2 (Based on (df.NumSensitiveWords * 2)  < df.UrlLength)
        violated_sample = sample.copy()
        violated_sample[get_index_by_feature_name('NumSensitiveWords')] = (
                violated_sample[get_index_by_feature_name('UrlLength')] // 2
                + np.random.randint(0, 3)
        )
        violated_samples.append(violated_sample)

        # Violation #3 (Based on df.PctNullSelfRedirectHyperlinks == 1 --> df.PctExtHyperlinks == 0)
        violated_sample = sample.copy()
        violated_sample[get_index_by_feature_name('PctNullSelfRedirectHyperlinks')] = 1
        violated_sample[get_index_by_feature_name('PctExtHyperlinks')] = 0.1 + (np.random.rand() / 2)  # != 0
        violated_samples.append(violated_sample)

        # Violation #4 (aka #3') (df.PctNullSelfRedirectHyperlinks + df.PctExtHyperlinks) <= 1
        violated_sample = sample.copy()
        violated_sample[get_index_by_feature_name('PctNullSelfRedirectHyperlinks')] = 0.51 + (np.random.rand() / 5)
        violated_sample[get_index_by_feature_name('PctExtHyperlinks')] = 0.51 + (np.random.rand() / 5)
        violated_samples.append(violated_sample)

        # Violation #5_0 (PctExtNullSelfRedirectHyperlinksRT == 1 --> PctNullSelfRedirectHyperlinks+PctExtHyperlinks < 0.3)
        # violated_sample['PctExtNullSelfRedirectHyperlinksRT'] = 1
        # violated_sample['PctNullSelfRedirectHyperlinks'] = 0.2 + (np.random.rand() / 3)
        # violated_sample['PctExtHyperlinks'] = 0.2 + (np.random.rand() / 3)

        # Violation #5_1 (PctExtNullSelfRedirectHyperlinksRT == 1 --> PctNullSelfRedirectHyperlinks < 0.3):
        violated_sample = sample.copy()
        violated_sample[get_index_by_feature_name('PctExtNullSelfRedirectHyperlinksRT')] = 1
        violated_sample[get_index_by_feature_name('PctNullSelfRedirectHyperlinks')] = 0.4 + (np.random.rand() / 3)
        violated_samples.append(violated_sample)

        # Violation #5_2 (PctExtNullSelfRedirectHyperlinksRT == 1 --> PctExtHyperlinks < 0.3):
        violated_sample = sample.copy()
        violated_sample[get_index_by_feature_name('PctExtNullSelfRedirectHyperlinksRT')] = 1
        violated_sample[get_index_by_feature_name('PctExtHyperlinks')] = 0.4 + (np.random.rand() / 3)
        violated_samples.append(violated_sample)

    else:
        raise NotImplementedError(f"Dataset `{dataset_name}` is not supported.")

    return violated_samples
