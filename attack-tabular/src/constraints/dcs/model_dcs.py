import pandas as pd
import numpy as np
from z3 import *

import logging
from functools import lru_cache
from typing import Dict, List, Tuple, Union

"""
Define the constraint model of the utilized constraints, Denial Constraints.
"""


logger = logging.getLogger(__name__)


class DenialConstraint:
    def __init__(self,
                 dc_string: str,  # the string representing the DC, assuming FastADC output format.
                 dc_file_idx: int = None,
                 other_tuples_data: pd.DataFrame = None):
        self.dc_string = dc_string
        self.dc_file_idx = dc_file_idx
        self.other_tuples_data = None
        self.dc_predicates: List[DCPredicate] = []

        self._parse_dc_from_string()
        self.set_other_tuples_data(other_tuples_data)

    def _parse_dc_from_string(self):
        """
        parses human-readable string of denial-constraint
        e.g.   `¬( t0.type(String) == t1.type(String) ∧ t0.local_code(String) <> t1.local_code(String) )`
        """
        # Split the string into predicates
        dc_string = self.dc_string.split(" ")
        self.dc_predicates: List[DCPredicate] = []

        for op_idx in range(2, len(dc_string), 4):
            col1_var, col1_name = dc_string[op_idx - 1][:2], dc_string[op_idx - 1][3:]
            col1_var = int(col1_var[-1])
            operator = dc_string[op_idx]
            col2_var, col2_name = dc_string[op_idx + 1][:2], dc_string[op_idx + 1][3:]
            # col2_var, col2_name = humanr_dc[op_idx + 1].split('.')
            col2_var = int(col2_var[-1])
            self.dc_predicates.append(DCPredicate(col1_name, col1_var, operator, col2_name, col2_var))

    def get_z3_formula(self, literals_dict: Dict[str, ExprRef]) -> List[BoolRef]:
        """
        Creates Z3 formula for the DC constraint, using the literals from `literals_dict`.
        """
        constraints_list = []

        # for each "other_tuple" we append the DC constraint.
        for _, other_tuple in self.other_tuples_data.iterrows():
            inner_constraint = []
            for pred in self.dc_predicates:
                inner_constraint.append(pred.get_z3_formula(literals_dict, other_tuple))
            constraints_list.append(Not(And(*inner_constraint)))  # append the full DC with other_tuple

        return constraints_list

    def check_satisfaction_of_pair(self, target_tuple: dict, other_tuple: dict):
        """
        Checks satisfaction of target_tuple as a pair with `other_tuple`, given as a dict.
        """
        is_sat = True
        sat_predicates_count = 0

        for predicate in self.dc_predicates:
            curr_pred_sat = predicate.check_pair_satisfaction((target_tuple, other_tuple))
            is_sat = is_sat and curr_pred_sat
            sat_predicates_count += int(curr_pred_sat)

        is_sat = not is_sat  # since this is *denial* constraint (describes what should not happen)
        return is_sat, sat_predicates_count

    def check_satisfaction_all_pairs(self, target_tuple: dict):
        """
        Returns array of satisfaction of `target_tuple` with all pairs (i.e. with `self.other_tuples_data`)

        Basically, an extended (array) version of `check_pair_satisfaction`
        """
        assert self.other_tuples_data is not None, "Must set `other_tuples_data` before calling this function"
        is_sat_arr = np.ones(len(self.other_tuples_data), dtype=bool)
        sat_predicates_count_arr = np.zeros(len(self.other_tuples_data), dtype=np.int8)

        for predicate in self.dc_predicates:
            curr_pred_sat_arr = predicate.check_pair_satisfaction((target_tuple, self.other_tuples_data))
            is_sat_arr = is_sat_arr & curr_pred_sat_arr
            sat_predicates_count_arr += curr_pred_sat_arr

        is_sat_arr = ~is_sat_arr  # since this is *denial* constraint (describes what should not happen)

        return is_sat_arr, sat_predicates_count_arr

    def get_predicate_count(self):
        # returns the number of predicates that compose the DC
        return len(self.dc_predicates)

    def set_other_tuples_data(self, other_tuples_data: pd.DataFrame):
        self.other_tuples_data = other_tuples_data

    def __repr__(self):
        return self.dc_string

    @lru_cache(maxsize=None)
    def does_given_feature_sat_dc(self, feature_name: str, feature_val, **kwargs) -> np.array:
        """
            Checks whether a single given feature, satisfies the DC (regardless to other features)
            - Observation: this happens IFF all the predicates that contain the feature are evaluated as False.
            - Note: We assume that each predicates addresses only a single feature (and does not mix features).
         """
        # 0. Define a target tuple, we don't add features that are not `feature_name`
        target_tuple = {feature_name: feature_val}

        # 1. Define the satisfaction tracker (per other-tuple)
        is_feature_sat_dc_arr = np.ones(len(self.other_tuples_data), dtype=int)

        # 2. Iterate on relevant predicates and check sat.
        for pred in self.dc_predicates:
            if pred.col1_name != pred.col1_name:
                logger.warning(f"[WARNING] Predicate {pred} includes two different variables; "
                               f"as opposed to this heuristics assumption.")
            if pred.col1_name != feature_name:
                continue  # we only care about predicates with the given feature_name.

            is_feature_sat_dc_arr += ~pred.check_pair_satisfaction((target_tuple, self.other_tuples_data))

        is_feature_sat_dc_arr = np.array(is_feature_sat_dc_arr)
        return is_feature_sat_dc_arr >= 1  # feature_sat_dc IFF _any_ relevant predicate was evaluated 'False'


class DCPredicate:
    def __init__(self, col1_name: str, col1_var: int, operator: str, col2_name: str, col2_var: int):
        assert operator in ["==", "<=", ">=", ">", "<", "<>"]
        self.col1_name, self.col1_var = col1_name.split('(')[0], col1_var
        self.col2_name, self.col2_var = col2_name.split('(')[0], col2_var
        self.operator = operator

    def check_pair_satisfaction(self, rows: Tuple[dict, Union[dict, pd.DataFrame, pd.Series]]) \
            -> Union[bool, pd.Series]:
        """ Also works for rows that are full pandas DataFrame. """
        row1, row2 = rows[self.col1_var], rows[self.col2_var]
        if self.operator == "==":
            return row1[self.col1_name] == row2[self.col2_name]
        if self.operator == ">":
            return row1[self.col1_name] > row2[self.col2_name]
        if self.operator == "<":
            return row1[self.col1_name] < row2[self.col2_name]
        if self.operator == "<=":
            return row1[self.col1_name] <= row2[self.col2_name]
        if self.operator == ">=":
            return row1[self.col1_name] >= row2[self.col2_name]
        if self.operator == "<>":
            return row1[self.col1_name] != row2[self.col2_name]

    def get_z3_formula(self, row1_literals: Dict[str, ExprRef], row2_values: pd.Series):
        return self.check_pair_satisfaction((row1_literals, row2_values))

    def __repr__(self):
        return f"t{self.col1_var}.{self.col1_name} {self.operator} t{self.col2_var}.{self.col2_name}"
