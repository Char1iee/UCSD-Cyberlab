from typing import Dict, Tuple, List, Type

import numpy as np
import pandas as pd
from tqdm import tqdm
from z3 import *

from src.constraints.dcs.mine_dcs import load_evaluated_dcs
from src.constraints.dcs.model_dcs import DenialConstraint
from src.constraints import Constrainer


class DCsConstrainer(Constrainer):
    """
    Class for utilizing Denial Constraints (DCs) (e.g., satisfiability check, projection onto DCs, etc.).
    """
    def __init__(
            self,
            x_tuples_df: pd.DataFrame,  # must be the same as mining
            evaluated_dcs_out_path: str,

            # Data properties:
            feature_names: list,
            is_feature_ordinal: List[bool],
            is_feature_continuous: List[bool],
            is_feature_categorical: List[bool],
            feature_types: List[type],
            feature_ranges: List[Tuple[float, float]],
            standard_factors: List[float],

            # DCs config:
            n_dcs: int = 5000,
            n_tuples: int = 1,

            # Attack parameters:
            limit_cost_ball: bool = True,
            cost_ball_eps: float = 1 / 30,

            **kwargs
    ):
        # Attack parameters:
        self.limit_cost_ball = limit_cost_ball
        self.cost_ball_eps = cost_ball_eps

        # we define the top scores as the DCs and set them
        self.dc_constraints_eval = load_evaluated_dcs(evaluated_dcs_out_path)
        self.x_tuples_df, self.n_dcs, self.n_tuples = x_tuples_df, n_dcs, n_tuples
        self.dcs = None
        self._get_dcs()

        # Data properties:
        self.feature_names = feature_names
        self.is_feature_ordinal = is_feature_ordinal
        self.is_feature_continuous = is_feature_continuous
        self.is_feature_categorical = is_feature_categorical
        self.feature_ranges = feature_ranges
        self.feature_types = feature_types
        self.standard_factors = standard_factors

        # Build DCs in SAT solver
        solver, literals_dict = self._build_cnf_dcs()
        self.solver: Solver = solver
        self.literals_dict: Dict[int, ExprRef] = literals_dict

    def _get_dcs(self):
        # sort by 'weighted_score' and take the top `n_dcs`
        self.dc_constraints_eval = self.dc_constraints_eval.nlargest(self.n_dcs, 'weighted_score')

        self.dcs = []
        for i, row in self.dc_constraints_eval.iterrows():
            # Option I - sample the "other" tuples randomly
            # dc_tuples_data = self.x_tuples_df.sample(n=self.n_tuples, random_state=i)

            # Option II - use satisfiability of tuples over evaluated set
            assert self.n_tuples <= len(row.best_other_tuples), \
                "Tuples used for inference must be <= the 'best-other-tuples' given in the dc_constraints_eval file."
            dc_best_tuples_indices = row.best_other_tuples[:self.n_tuples]
            dc_tuples_data = self.x_tuples_df.loc[dc_best_tuples_indices]  # accessed by 'x_tuples_df.index'

            dc_tuples_data = dc_tuples_data.reset_index(drop=True)
            self.dcs.append(DenialConstraint(dc_string=row.dcs_repr, other_tuples_data=dc_tuples_data))

    def check_sat(self,
                  sample: np.array,
                  sample_original: np.array = None) -> bool:
        """
        :param sample: dict that maps `feature_name` (str) to its value (int/float), or no value.
        :param sample_original: original sample (usually 'sample' is perturbed one); used to limit the cost-ball.
        :return: whether the sample satisfies the constraints.
        """
        is_sat = self._check_sat_with_given_literals(sample, sample_original=sample_original, check_sat_only=True)
        return is_sat

    def _check_sat_with_given_literals(self,
                                       sample: np.array,
                                       sample_original: np.array = None,
                                       check_sat_only: bool = None):
        """
        :param sample: array of the sample's values, where `nan` means a literal to free.
        :param sample_original: If `self.limit_cost_ball` and `original_sample` is given, then the
                                projection also enforces the cost-ball.
        :param check_sat_only: whether to return only the satisfiability (without the satisfying model)
        """

        # Build the assignment to check
        def _get_partial_assignment(_literal_key, _assigned_val, _literals_dict):
            """ returns expression that forces the feature to have `assigned_val`"""
            return _literals_dict[_literal_key] == _assigned_val

        assignment = []
        for f_idx, f_val in enumerate(sample):
            if np.isnan(f_val):  # `nan` means a literal to free
                continue
            # asserts each feature to have the given sample's value
            assignment.append(_get_partial_assignment(f_idx, f_val, self.literals_dict))

        # Add any additional (e.g., cost) assertions
        additional_assertions = []
        if self.limit_cost_ball is not None and sample_original is not None:
            additional_assertions = self._get_cost_ball_assertions(sample_original)

        is_sat = self.solver.check(*assignment, *additional_assertions)  # returns satisfiability
        is_sat = is_sat.r == Z3_L_TRUE
        m = None
        if is_sat:
            m = self.solver.model()  # the 'model' used to satisfy

        if check_sat_only:
            return is_sat
        return is_sat, m

    def project_sample(self,
                       sample: np.ndarray,
                       literals_to_free: List[int],
                       sample_original: np.ndarray = None) -> Tuple[bool, np.ndarray]:
        """
        Project adv-samples that violate the constraints, by freeing literals.
        :return is_sat: whether the projection was successful
        :return projected_sample: the projected sample
        """

        # 0. Build the projection sample
        projected_sample = sample.copy()
        projected_sample[literals_to_free] = np.nan

        # 1. Attempt to satisfy sample after freeing the least-constrained features
        is_sat, sat_model = self._check_sat_with_given_literals(projected_sample, sample_original=sample_original)

        # 2. Fetch the sat-solver solution after the projection, and update the sample
        if is_sat:  # if projection successful
            for freed_literal_idx in literals_to_free:
                freed_literal = self.literals_dict[freed_literal_idx]
                freed_literal_type: Type = self.feature_types[freed_literal_idx]
                projected_sample[freed_literal_idx] = freed_literal_type(eval(sat_model[freed_literal].as_string()))
        else:
            projected_sample = sample  # unsuccessful projection, return the original sample

        return is_sat, projected_sample

    def get_literals_scores(self, sample: np.ndarray) -> np.array:
        """
        :return: a list of indices, order by their score, from lowest to highest.
            (the lower the score the less constrained the literal)
        """
        # 1. find the least-constrained features (intuition: should be the easiest to project)
        # the higher the sat --> the more constrained --> the less we would like to free it.
        features_sat = np.zeros_like(sample)
        for f_idx, f_val in enumerate(sample):
            # Option I: simply choose random features
            # features_sat[f_idx] = np.random.rand()  # disabled

            # TODO [ENHANCE-RUNTIME] can make it parallel to accelerate projection
            # Option II: rank feature by its ability to satisfy _alone_ DCs.
            for i, dc in enumerate(self.dcs):
                feature_name = self.feature_names[f_idx]
                features_sat[f_idx] += dc.does_given_feature_sat_dc(feature_name, f_val, dc_idx=i).mean()

        return features_sat

    def _build_cnf_dcs(self):
        """ Builds a z3 solver object, based on the valiant's constraints """
        # each feature-value pair is a literal, each literal's name is the DC-col name.
        literals_dict: Dict[int, ExprRef] = {}

        # 1. Extract the support values and initialize literals
        for idx, feature_name in enumerate(self.feature_names):
            if self.is_feature_continuous[idx]:
                literals_dict[idx] = Real(feature_name)  # add feature as literal
            else:  # if self.is_feature_ordinal[idx] or categorical:
                literals_dict[idx] = Int(feature_name)  # add feature as literal

        s = Solver()
        # 2. enforce support range
        for idx, feature_range in enumerate(self.feature_ranges):
            literal = literals_dict[idx]
            lower, upper = feature_range
            lower, upper = float(lower), float(upper)
            if lower == -np.inf: lower = -2 ** 32
            if upper == np.inf: upper = 2 ** 32
            s.add(lower <= literal, literal <= upper)

        # 3. enforce the constraints
        literals_dict_named = {f_name: literals_dict[f_idx] for f_idx, f_name in enumerate(self.feature_names)}
        for dc in tqdm(self.dcs, desc='Builds DCs CNF'):
            s.add(*dc.get_z3_formula(literals_dict_named))

        return s, literals_dict

    def _get_cost_ball_assertions(self, sample_original: np.ndarray) -> List[BoolRef]:
        """
        Generates the assertions for the cost-ball.
            - Should correspond to the cost-ball used in the attack.
        """
        cost_assertions = []
        for f_idx, f_name in enumerate(self.feature_names):
            if self.is_feature_categorical[f_idx]:  # we don't apply cost ball for categorical
                continue
            literal = self.literals_dict[f_idx]
            original_val = sample_original[f_idx]
            lower = original_val - (self.cost_ball_eps * self.standard_factors[f_idx])
            upper = original_val + (self.cost_ball_eps * self.standard_factors[f_idx])
            if self.is_feature_ordinal[f_idx]:  # we round ball for integers, to slightly relax this requirement
                lower, upper = math.floor(lower), math.ceil(upper)
            lower, upper = float(lower), float(upper)
            if lower == -np.inf: lower = -2 ** 32
            if upper == np.inf: upper = 2 ** 32
            cost_assertions += [lower <= literal, literal <= upper]
        return cost_assertions
