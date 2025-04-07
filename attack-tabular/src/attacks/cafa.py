from typing import Union, Optional, List
import logging

import numpy as np
from tqdm import tqdm

from art.attacks import EvasionAttack
from art.estimators import BaseEstimator, LossGradientsMixin
from art.estimators.classification import ClassifierMixin
from art.summary_writer import SummaryWriter

logger = logging.getLogger(__name__)


class CaFA(EvasionAttack):
    """
    PGD attack variation on tabular data.
    """

    attack_params = EvasionAttack.attack_params + [
        'cat_indices',
        'ordinal_indices',
        'cont_indices',
        'feature_ranges',
        'standard_factors',
        'cat_encoding_method',
        'one_hot_groups',
        'random_init',
        'random_seed',
        'max_iter',
        'max_iter_tabpgd',
        'eps',
        'step_size',
        'perturb_categorical_each_steps',
        'summary_writer',
    ]
    # Requiring implementation of 'loss_gradient()' (i.e., white-box access), via `LossGradientsMixin`.
    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ClassifierMixin)

    def __init__(
            self,
            estimator: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",

            # Data-specific parameters:
            cat_indices: np.ndarray,
            ordinal_indices: np.ndarray,
            cont_indices: np.ndarray,
            feature_ranges: np.ndarray,
            standard_factors: np.ndarray = None,

            cat_encoding_method: str = 'one_hot_encoding',
            one_hot_groups: List[np.ndarray] = None,

            # TabPGD HPs:
            random_init: bool = True,
            random_seed: int = None,
            max_iter: int = 500,
            max_iter_tabpgd: int = 100,
            # batch_size: int = None,  # TODO [ADD-FEATURE] support for batching
            # targeted: bool = False, # TODO [ADD-FEATURE] support for targeted attack
            eps: float = 0.03,
            step_size: float = 0.0003,
            perturb_categorical_each_steps: int = 10,

            # Misc:  # TODO [ADD-FEATURE] integrate summary_writer in the code
            summary_writer: Union[str, bool, SummaryWriter] = False,
            **kwargs
    ):
        """

        :param estimator: Targeted NN model; should implement `loss_gradient()` (see ART's estimators). This
                          instantiates a white-box-accessed target model.

        ## Data properties:
        :param cat_indices: Indices of categorical features.
        :param ordinal_indices: Indices of ordinal features.
        :param cont_indices: Indices of continuous features.
        :param feature_ranges: The semantic range for each feature (e.g., feature that represents a rate should
                               have [0,1] range). Of shape (n_features, 2) with lower,upper for each feature.
        :param standard_factors: The standardization factor for each feature, used in the attack step size and cost. Of
                                 shape (n_features,) with the standardization factor for each feature.
        :param cat_encoding_method: The method used to encode categorical features. Currently only 'one_hot_encoding'
        :param one_hot_groups: A list of np.ndarray-s, where each array holds the indices of the one-hot-encoded

        ## Attack Hyperparameters: (see paper for more detailed description)
        :param random_init: Whether to start the attack from a random point inside the epsilon-ball.
        :param random_seed: Random seed to use for the attack. Defaults to no seed (i.e., random).
        :param max_iter: The maximum iterations to perform in the main loop (of TabCWL0), each runs TabPGD. Setting
                         the argument to `1` means running TabPGD alone.
        :param max_iter_tabpgd: The maximum iterations to perform in the TabPGD loop.
        :param eps: Proportion of the allowed l-inf-standardized ball to generate adversarial samples within.
        :param step_size: The step size taken in each TabPGD iteration.
        :param perturb_categorical_each_steps: The number of iterations to perform between perturbing categorical
                                               features.

        ## Misc:
        :param summary_writer:  # TODO [ADD-FEATURE] integrate summary_writer in the code
        """
        super().__init__(estimator=estimator,
                         summary_writer=summary_writer)

        self.eps = eps
        self.step_size = step_size
        self.max_iter = max_iter
        self.max_iter_tabpgd = max_iter_tabpgd
        self.perturb_categorical_each_steps = perturb_categorical_each_steps
        self.random_init = random_init
        self.random_seed = random_seed

        self.feature_ranges = feature_ranges
        self.standard_factors = standard_factors
        self.cat_indices = cat_indices
        self.ordinal_indices = ordinal_indices
        self.cont_indices = cont_indices

        self.cat_encoding_method = cat_encoding_method
        self.one_hot_groups = one_hot_groups

        # Validations:
        self._validate_input()

        # Fix a random seed (if required)
        np.random.seed(self.random_seed)

    def generate(self,
                 x: np.ndarray,
                 y: np.ndarray = None,
                 mask: np.ndarray = None,
                 ):
        """
        Generate adversarial samples with TabCW+TabPGD and return them in an array.
            - The adversarial samples enforce the structure-constraints.
            - The adversarial samples are crafted within an epsilon l-inf ball.
            - The algorithm minimizes the l0 of the adversarial perturbation.

        :param x: An array with the original inputs to be attacked.
        :param y: An array with the original labels of the inputs.
        :return: An array holding the adversarial examples.
        """

        if y is None:  # Use model predictions as correct outputs, if not provided
            y = self.estimator.predict(x).argmax(axis=1)

        x_adv = x.copy()
        mask = np.ones_like(x) if mask is None else mask
        mask = mask.astype(np.float32)
        new_mask = mask.copy()

        prev_attack_l0_vals = np.full(x.shape[0], np.inf, dtype=np.float32)
        i = 0

        # TODO [ENHANCE-PERFORMANCE] can add more stopping conditions (e.g., by time or success).
        while i < self.max_iter and mask.sum() > 0:
            if i > 0:
                # Prevent perturbation of 'least useful' feature
                least_imp_feature = self._get_least_important_feature(x=x, x_adv=x_adv, y=y, mask=mask)
                for sample_idx, least_imp_features in enumerate(least_imp_feature):
                    new_mask[sample_idx, least_imp_features] = 0

            # Perform TabPGD attack
            new_x_adv = self.generate_with_tabpgd(x=x, y=y, mask=new_mask)

            # Pick the best new adversarial samples and update with them
            is_attack_success = self.estimator.predict(new_x_adv).argmax(axis=1) != y
            attack_l0_vals = self.calc_l0_cost(new_x_adv, x)
            is_lower_l0 = attack_l0_vals < prev_attack_l0_vals
            update_samples = is_attack_success & is_lower_l0
            x_adv[update_samples] = new_x_adv[update_samples]
            mask[update_samples] = new_mask[update_samples]  # don't update mask for non-updated samples

            # Evaluate metrics
            logger.debug(f"[{i}] success(x_adv, y): {(self.estimator.predict(x_adv).argmax(axis=1) != y).mean()}")
            logger.debug(f"[{i}] l0(x_adv): {self.calc_l0_cost(x_adv, x).mean()}")
            logger.debug(f"[{i}] mean(mask): {mask.sum(axis=1).mean()}")
            logger.debug(f"[{i}] update rate: {update_samples.mean()}")

            # Perform updates
            i += 1  # decrease remaining iterations
            # prev_is_attack_success = is_attack_success
            prev_attack_l0_vals = attack_l0_vals

        return x_adv

    def generate_with_tabpgd(self,
                             x: np.ndarray,
                             y: np.ndarray,
                             mask: np.ndarray = None,
                             **kwargs) -> np.ndarray:
        """
        Utilize TabPGD to Generate adversarial samples enforcing the structural and standardized-l-inf
        constraints and return them in an array.
    
        :param x: An array with the original inputs.
        :param y: An array with the original labels to be predicted.
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Note that a mask should apply to One-Hot groups altogether.
        :return: An array holding the adversarial examples.
        """
        if y is None:
            y = self.estimator.predict(x).argmax(axis=1)
    
        if not np.issubdtype(y.dtype, np.integer):
            y = y.astype(np.int64)
    
        x, y = x.astype(np.float32), y.astype(np.int64)
        allow_updates = np.ones(x.shape[0], dtype=np.float32)
        accum_grads = np.zeros_like(x)
        mask = np.ones_like(x) if mask is None else mask
        mask = mask.astype(np.float32)
    
        epsilon_ball_lower, epsilon_ball_upper = self._init_epsilon_ball(x)
        x_adv = x.copy()
    
        for step in tqdm(range(self.max_iter_tabpgd)):
            x_adv_before_perturb = x_adv.copy()  # for validation purposes
    
            # Inject allow_updates-mask to 'mask'
            mask *= allow_updates[:, None]
    
            # Initialize the perturbation
            perturbation_temp = np.zeros_like(x_adv)
    
            if self.random_init and step == 0:
                perturbation_temp = np.random.uniform(-self.eps, self.eps, x.shape) * self.standard_factors
                x_adv += self._get_random_categorical_perturbation(x_adv) * mask
            else:
                grad = self.estimator.loss_gradient(x, y)
                grad *= mask
                accum_grads += grad
                perturbation_temp = self.step_size * self.standard_factors * np.sign(grad)
    
            # 4.1. Perturb continuous as is
            x_adv += self._get_perturbation_continuous(perturbation_temp) * mask
            # 4.2. Perturb integers with rounding
            x_adv += self._get_perturbation_ordinal(perturbation_temp) * mask
            # 4.3. Perturb categorical according to accumulated grads
            if step % self.perturb_categorical_each_steps == 0 and step != 0:
                x_adv += self._get_perturbation_categorical(x_adv, accum_grads) * mask
    
            # 5. Project back to standard-epsilon-ball
            x_adv = np.clip(x_adv, epsilon_ball_lower, epsilon_ball_upper)
    
            # 6.1. Clip to integer features
            x_adv[:, self.ordinal_indices] = np.round(x_adv[:, self.ordinal_indices])
            # 6.2. Clip to feature ranges
            x_adv = np.clip(x_adv, self.feature_ranges[:, 0], self.feature_ranges[:, 1])
    
            # 强制保持 mask=0 的位置不变
            x_adv[~mask.astype(bool)] = x_adv_before_perturb[~mask.astype(bool)]
    
            # Assert that non masked / early-stopped feature was perturbed (x_adv_before_perturb)
            assert np.allclose(x_adv[~mask.astype(bool)], x_adv_before_perturb[~mask.astype(bool)], atol=1e-5), \
                   "Masked features changed unexpectedly"
    
            # 8. Early stop who are already adversarial
            x_adv = x_adv.astype(np.float32)
            is_attack_success = self.estimator.predict(x_adv).argmax(axis=1) != y
            logger.debug(f"ASR: {is_attack_success.mean() * 100: .2f}%")
            allow_updates -= allow_updates * is_attack_success
    
            # 10. Early stop if all samples are already adversarial
            if allow_updates.sum() == 0:
                break
    
        return x_adv

    def _get_least_important_feature(self,
                                     x: np.ndarray,
                                     x_adv: np.ndarray,
                                     y: np.ndarray,
                                     mask: np.ndarray,
                                     ) -> List[List[int]]:
        """
        As part of TabCWL0 algorithm, applies a heuristic to identify tne least important features to each sample.
        :param x: The original samples
        :param x_adv: The perturbed samples.
        :param y: The original labels
        :param mask: The mask of features to consider.
        :return: A list of lists, where each inner list holds the indices of the least important features for a sample.
        """
        # TODO [ALGORITHM-ENHANCEMENT] insert randomness? specifically, for samples failed in previous iterations?
        delta = x_adv - x
        grad = self.estimator.loss_gradient(x_adv, y)
        cw_score = np.abs(grad * delta)

        if self.cat_encoding_method == 'one_hot_encoding':
            # Aggregate score for One-Hot-Encoded feature over all categories
            for oh_group in self.one_hot_groups:
                for sample_idx in range(cw_score.shape[0]):
                    cw_score[sample_idx, oh_group] = np.abs(cw_score[sample_idx, oh_group]).sum()

        # TODO [ADD-FEATURE] Implement TabNet support
        """ 
        # TABNET OPTION 
        if self.model_type == 'tabnet':
            embedding_grads = additional_grad_info['embedding_grad']
            unordered_indices = self.dataset.unordered_indices

            # for each categorical feature, get the gradient of the embedding
            for unordered_feat_idx, feat_idx in enumerate(unordered_indices):
                for sample_idx in range(cw_score.shape[0]):
                    # OPTION 1:  # cw_score[sample_idx, feat_idx] = embedding_grads[unordered_feat_idx][batch_x[:, feat_idx].long()].abs().sum()
                    cw_score[sample_idx, feat_idx] = embedding_grads[unordered_feat_idx].sum()
            cw_score[:, unordered_indices] *= (x_adv_delta[:, unordered_indices] != 0)
            # cw_score = cw_score.abs()  # currently disabled
        """

        cw_score = cw_score / self.standard_factors
        # Non perturbed feature should get maximal score (as they are meaningless)
        cw_score[~mask.astype(bool)] = np.inf

        # Get minimal score for each sample
        least_imp_feature_per_sample = [[score.item()] for score in cw_score.argmin(axis=1)]

        # Expand the list of features, per sample, to all the One-Hot involved coordinates
        for sample_idx in range(len(least_imp_feature_per_sample)):
            for oh_group in self.one_hot_groups:
                if least_imp_feature_per_sample[sample_idx][0] in oh_group:
                    least_imp_feature_per_sample[sample_idx] = oh_group
                    break  # can stop looking for the oh group

        return least_imp_feature_per_sample

    def _init_epsilon_ball(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        epsilon_ball_upper = x + (self.eps * self.standard_factors)
        # epsilon_ball_upper[:, self.ordinal_indices] = np.ceil(epsilon_ball_upper[:, self.ordinal_indices])

        epsilon_ball_lower = x - (self.eps * self.standard_factors)
        # epsilon_ball_lower[:, self.ordinal_indices] = np.floor(epsilon_ball_lower[:, self.ordinal_indices])

        if self.cat_encoding_method == 'one_hot_encoding':
            # in case of one-hot-encoding we simply allow perturbation of all categories
            epsilon_ball_upper[:, self.cat_indices] = 1.0
            epsilon_ball_lower[:, self.cat_indices] = 0.0

        return epsilon_ball_lower, epsilon_ball_upper

    def _get_perturbation_continuous(self, perturbation_temp: np.ndarray) -> np.ndarray:
        perturb_cont = np.zeros_like(perturbation_temp)
        perturb_cont[:, self.cont_indices] = perturbation_temp[:, self.cont_indices]
        return perturb_cont

    def _get_perturbation_ordinal(self, perturbation_temp):
        perturb_ord = np.zeros_like(perturbation_temp)
        perturb_ord[:, self.ordinal_indices] = (np.ceil(np.abs(perturbation_temp[:, self.ordinal_indices]))
                                                * np.sign(perturbation_temp[:, self.ordinal_indices]))
        return perturb_ord

    def _get_perturbation_categorical(self,
                                      x_adv: np.ndarray, accum_grads: np.ndarray,
                                      perturb_one_feature_only: bool = False) -> np.ndarray:
        perturb_cat = np.zeros_like(x_adv)

        if self.cat_encoding_method == 'one_hot_encoding':

            # get the max value of each OH group
            score_grads_per_group = np.zeros((x_adv.shape[0], len(self.one_hot_groups)))
            for id_oh_group, oh_group in enumerate(self.one_hot_groups):
                score_grads_per_group[:, id_oh_group] = accum_grads[:, oh_group].max(-1)

            # perturb groups with the largest max value
            for id_oh_group, oh_group in enumerate(self.one_hot_groups):
                samples_to_update_indices = np.arange(x_adv.shape[0])
                if perturb_one_feature_only:
                    # get indices of samples we want to update (samples with max grad in this group)
                    samples_to_update = (score_grads_per_group.argmax(axis=-1) == id_oh_group)
                    samples_to_update_indices = np.where(samples_to_update)[0]
                samples_to_update_indices = np.expand_dims(samples_to_update_indices, axis=1)  # to be used as an index
                # get the largest accum_grads of the group
                chosen_cats = oh_group[accum_grads[:, oh_group].argmax(axis=1)]
                chosen_cats = chosen_cats[samples_to_update_indices]
                # turn (only) these to 1 after cancelling previous category
                perturb_cat[samples_to_update_indices, oh_group] = -x_adv[samples_to_update_indices, oh_group]
                perturb_cat[samples_to_update_indices, chosen_cats] += 1
        else:
            # TODO [ADD-FEATURE] Implement TabNet support
            raise NotImplementedError
        return perturb_cat

    def _get_random_categorical_perturbation(self, x_adv: np.ndarray) -> np.ndarray:
        """
        Perturb a random feature of each sample, to a random category.
        Used for random initialization of the attack.
        """

        perturb_cat = np.zeros_like(x_adv)
        # 1. choose random feature to perturb, per sample
        chosen_oh_groups = np.random.randint(0, len(self.one_hot_groups), size=x_adv.shape[0])

        # 2. choose random category for the chosen feature, per sample
        for sample_idx, chosen_oh_group_idx in enumerate(chosen_oh_groups):
            chosen_oh_group = self.one_hot_groups[chosen_oh_group_idx]
            chosen_cat = np.random.randint(0, len(chosen_oh_group))
            chosen_cat_idx = self.one_hot_groups[chosen_oh_group_idx][chosen_cat]
            perturb_cat[sample_idx, chosen_oh_group] -= x_adv[sample_idx, chosen_oh_group]
            perturb_cat[sample_idx, chosen_cat_idx] = 1

        return perturb_cat

    def _validate_input(self):
        assert self.cat_encoding_method in ['one_hot_encoding'], 'only one-hot encoding is supported for now'

        # verify one-hot groups are cover the categorical features
        oh_indices = set()
        for oh_group in self.one_hot_groups:
            oh_indices.update(oh_group)
        assert oh_indices == set(self.cat_indices), 'one-hot groups should cover all categorical indices'

        assert (set(self.cat_indices) | set(self.ordinal_indices) | set(self.cont_indices)
                == set(range(self.feature_ranges.shape[0]))), 'indices should form all features'

        # verify feature indices are disjoint
        assert len(set(self.cat_indices) & set(self.ordinal_indices)) == 0, \
            'cat and ordinal indices should be disjoint'
        assert len(set(self.cat_indices) & set(self.cont_indices)) == 0, \
            'cat and cont indices should be disjoint'
        assert len(set(self.cont_indices) & set(self.ordinal_indices)) == 0, \
            'cont and ordinal indices should be disjoint'

    @staticmethod
    def calc_l0_cost(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return (np.abs(x1 - x2) > 1e-7).sum(axis=-1)

    @staticmethod
    def calc_standard_linf_cost(x1: np.ndarray, x2: np.ndarray,
                                standard_factors: np.ndarray,
                                relevant_indices: Optional[np.ndarray] = None) -> np.ndarray:
        if relevant_indices is None:
            relevant_indices = np.arange(x1.shape[1])
        delta = np.abs(x1 - x2)
        # delta[:, ordinal_indices] = np.floor(delta[:, ordinal_indices])  # currently disabled
        return (delta[:, relevant_indices] / standard_factors[relevant_indices]).max(axis=-1)
