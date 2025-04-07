import json
import os
import sys
from typing import List
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
import logging

from src.constraints.dcs.model_dcs import DenialConstraint

logger = logging.getLogger(__name__)


def mine_dcs(x_mine_source_df: pd.DataFrame,
             raw_dcs_out_path: str,
             evaluated_dcs_out_path: str,
             path_to_fastadc_miner_jar: str,
             x_dcs_col_names: List[str],

             # DCs configuration
             approx_violation_threshold: float = 0.01,
             n_tuples_to_eval: int = 750,  # limit the recorded best-other-tuples to 0.75K
             n_dcs_to_eval: int = 10_000,  # limit the number of evaluated DCs to 10K
             n_other_tuples_to_eval: int = 5000,  # limit the number of evaluated DCs to 10K

             # Phases to execute:
             perform_constraints_mining: bool = True,
             perform_constraints_ranking: bool = True,
             ):
    x_dcs_df = x_mine_source_df.copy()
    x_dcs_df.columns = x_dcs_col_names
    if perform_constraints_mining:
        logger.info(">> Running DC Mining Algorithm")
        run_fast_adc(mine_source_df=x_dcs_df,
                     path_to_save_raw_dcs=raw_dcs_out_path,
                     approx_violation_threshold=approx_violation_threshold,
                     path_to_fastadc_miner_jar=path_to_fastadc_miner_jar)

    if perform_constraints_ranking:
        logger.info(">> Evaluating and Ranking DCs")
        dcs: List[DenialConstraint] = load_dcs_from_txt(raw_dcs_out_path)
        # Evaluate DCs metrics and Rank DCs by these metrics (via manually-crafted linear combination)
        evaluated_dcs = eval_and_rank_dcs(
            x_tuples_df=x_mine_source_df,
            dcs=dcs,
            n_tuples_to_eval=n_tuples_to_eval,
            n_dcs_to_eval=n_dcs_to_eval,
            n_other_tuples_to_eval=n_other_tuples_to_eval,
        )
        evaluated_dcs.to_csv(evaluated_dcs_out_path, index=False)


def run_fast_adc(mine_source_df: pd.DataFrame,
                 path_to_save_raw_dcs: str,
                 path_to_fastadc_miner_jar: str,
                 approx_violation_threshold: float = 0.01):
    curr_package_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    path_to_jar = os.path.join(curr_package_dir, path_to_fastadc_miner_jar)

    # Save mine_source_df to 'input_processed_data_csv_name'
    input_processed_data_csv_name = os.path.join(curr_package_dir, "temp___input_processed_data.csv")
    mine_source_df.to_csv(input_processed_data_csv_name, index=False)

    # Run:
    logger.info(">> Running DC Mining Algorithm")
    res = subprocess.run(["java", "-jar", path_to_jar, input_processed_data_csv_name, path_to_save_raw_dcs,
                          str(approx_violation_threshold)])
    # raise exception if the mining failed
    res.check_returncode()

    logger.info(f"{res} \n {res.stdout} \n {res.stderr}")


def load_dcs_from_txt(dcs_txt_path: str) -> List[DenialConstraint]:
    dcs = []
    # Loads constraints from txt
    with open(dcs_txt_path, "r") as f:
        for dc_file_idx, dc_string in enumerate(f):
            dcs.append(DenialConstraint(dc_string=dc_string,
                                        dc_file_idx=dc_file_idx))
    return dcs


def eval_and_rank_dcs(x_tuples_df: pd.DataFrame,
                      dcs: List[DenialConstraint],
                      n_dcs_to_eval: int = 10_000,
                      n_other_tuples_to_eval: int = 5000,
                      n_tuples_to_eval: int = 1000):
    """
    :param x_tuples_df: DataFrame with data tuples to evaluate on. These tuples will play the role of main tuple
                        (in the pair).
                        Should be trimmed BEFORE this function.
    :param dcs: list with DC objects to evaluate.
                We assume that all DCs have the same size of `dc.data_tuple` (denoted `n_other_data_tuples`)
    :param n_dcs_to_eval: amount of DC to evaluate (evaluate the `n_dcs` most succinctness DCs)
    :param n_other_tuples_to_eval: amount of 'other' tuples to consider as candidates for 'best' tuples of each DC.
    :param n_tuples_to_eval: Amount of samples on which we perform the evaluation on
    :return: prints and saves evaluation CSV.
    """
    # Trim DC as requested, by _Succinctness_ - measure the length of each DC  (the closer it is to 1,
    # the more compact the DC)
    _dc_sizes = [dc.get_predicate_count() for dc in dcs]
    _min_dc_size = min(_dc_sizes)
    succinctness_per_dc = _min_dc_size / np.array(_dc_sizes)
    if n_dcs_to_eval:
        # trim by succinctness
        top_succinct_dcs = succinctness_per_dc.argsort()[-n_dcs_to_eval:]
        old_dcs, dcs = dcs, []
        for old_dc_idx in top_succinct_dcs:
            dcs.append(old_dcs[old_dc_idx])

        # calculate succinctness again
        _dc_sizes = [dc.get_predicate_count() for dc in dcs]
        succinctness_per_dc = _min_dc_size / np.array(_dc_sizes)

    # Set the 'other-tuples' data for each DC
    x_other_tuples_df = x_tuples_df[:n_other_tuples_to_eval].copy()  # takes the tuples for `others` from the beginning
    for dc in dcs:
        dc.set_other_tuples_data(x_other_tuples_df)

    x_tuples_to_eval_df = x_tuples_df[-n_tuples_to_eval:].copy()  # takes the tuples to evaluate with from the end
    # Note: each operation on x_tuples_df should keep the original indices of the DataFrame (as they are used later).

    logger.info(f"Evaluating {len(dcs)=}, `t` from {len(x_tuples_to_eval_df)=} and `t'` from {len(x_other_tuples_df)=}")

    # Metric I (g_1): for each DC we calculate the violation rate (over all the possible pairs)
    ##      [Analogue to f_1 from Livshits et al. 2021 / g_1 in FastADC]
    pairs_violation_rate_per_dc = np.zeros(len(dcs))
    # Metric II (g_2): proportion of tuples with _any_ violation of the DC
    tuple_violation_rate_per_dc = np.zeros(len(dcs))
    """ # disabled at the moment
    # Metric III: proportion of tuples that perfectly satisfy the dc
    # # the entry [t_idx, dc_idx] records the violation count the tuple `t_idx`  the DC `dc_idx`.
    tuple_violation_count_per_dc_per_tuple = np.zeros((len(dcs), len(df)))
    """
    # Coverage - rates the amount of predicates-sat in a certain DC
    #       entry [dc_idx, sat_pred_count_idx] = how many pairs satisfied `sat_pred_count_idx` predicates in `dc_idx`
    sat_pred_count_per_dc = np.zeros((len(dcs), max(_dc_sizes) + 1))
    best_other_tuples_per_dc = np.zeros((len(dcs), n_tuples_to_eval), dtype=int)

    for dc_idx, dc in tqdm(enumerate(dcs), desc="Evaluating DCs..."):
        dc_sat_per_other_tuples = np.zeros(n_other_tuples_to_eval, dtype=int)
        for idx1, row in x_tuples_to_eval_df.iterrows():  # iterate on t (main tuple)
            is_sat_arr, sat_predicates_count_arr = dc.check_satisfaction_all_pairs(row.to_dict())

            # Track the sat of tuples playing the "other-tuple" role
            dc_sat_per_other_tuples += is_sat_arr.values

            # Metrics:
            pairs_violation_rate_per_dc[dc_idx] += (~is_sat_arr).sum()  # Metric I
            tuple_violation_rate_per_dc[dc_idx] += (~is_sat_arr).any()  # Metric II
            # Coverage:
            for dc_pred_count_idx, dc_pred_count_val in \
                    zip(*np.unique(sat_predicates_count_arr, return_counts=True)):
                # aggregates the satisfied-predicates spotted in the `dc` with `idx1`.
                sat_pred_count_per_dc[dc_idx, int(dc_pred_count_idx)] += dc_pred_count_val

        best_other_tuples_per_dc[dc_idx] = dc.other_tuples_data.index[dc_sat_per_other_tuples.argsort()[-n_tuples_to_eval:][::-1]].values
        logger.info(f"{dc_idx} >> {dc_sat_per_other_tuples.min()} {dc_sat_per_other_tuples.max()}")

    # Normalize metrics
    pairs_violation_rate_per_dc /= n_other_tuples_to_eval * n_tuples_to_eval  # normalize by number of evaluated pairs
    tuple_violation_rate_per_dc /= n_tuples_to_eval
    # Calculate coverage
    w = (np.arange(sat_pred_count_per_dc.shape[-1]) + 1) / sat_pred_count_per_dc.shape[-1]
    coverage_per_dc = (sat_pred_count_per_dc * w).sum(axis=-1) / sat_pred_count_per_dc.sum(axis=-1)

    logger.info(f">> Mean DC violation rate (lower->better, rate is over all pairs): "
          f"{pairs_violation_rate_per_dc.mean() * 100}%")
    logger.info(f">> Mean DC violation rate (lower->better, rate over tuples, for each tuples there exist a pair violating): "
          f"{tuple_violation_rate_per_dc.mean() * 100}%")
    logger.info(f">> Mean Succinctness (higher->better, correlates to predicate size, higher the closer "
          f"DCs to the min-sized DC): {succinctness_per_dc.mean() * 100}%")
    logger.info(f">> Mean Coverage (higher->better, correlates to amount of predicates being sat in DCs): "
          f"{coverage_per_dc.mean() * 100}%")

    logger.info(f">> {(pairs_violation_rate_per_dc == 1).mean() * 100}% of the DC are perfectly satisfied :) ")
    logger.info(f">> Worst DC, with {pairs_violation_rate_per_dc.max() * 100}% sat-rate was : "
          f"{dcs[pairs_violation_rate_per_dc.argmax()]}")

    # Transform metrics to a DF
    dc_constraints_eval = pd.DataFrame({
        'dcs_file_idx': [dc.dc_file_idx for dc in dcs],
        'dcs_repr': [str(dc) for dc in dcs],  # from which the DC can be reproduced
        'pairs_violation_rate_per_dc': pairs_violation_rate_per_dc,
        'tuple_violation_rate_per_dc': tuple_violation_rate_per_dc,
        'succinctness_per_dc': succinctness_per_dc,
        'coverage_per_dc': coverage_per_dc,

        # list of the tuples with highest sat-rate in their role as 'other tuple' in the DC.
        'best_other_tuples': [json.dumps(lst) for lst in best_other_tuples_per_dc.tolist()],
    })

    # filter DCs by their 'interesting-ness' ranking form a weighted-score of the metrics
    # adds score column, and set the order accordingly.
    # calculate additional factors
    dc_constraints_eval['tuple_violation_rate_per_dc__below_1_pct'] = dc_constraints_eval[
                                                                          'tuple_violation_rate_per_dc'] <= 0.015
    dc_constraints_eval['tuple_violation_rate_per_dc__below_5_pct'] = dc_constraints_eval[
                                                                          'tuple_violation_rate_per_dc'] <= 0.05
    dc_constraints_eval['tuple_violation_rate_per_dc__below_10_pct'] = dc_constraints_eval[
                                                                           'tuple_violation_rate_per_dc'] <= 0.10
    col_to_weight = {
        'tuple_violation_rate_per_dc__below_1_pct': 5.0,
        'tuple_violation_rate_per_dc__below_5_pct': 1.2,
        'tuple_violation_rate_per_dc__below_10_pct': 1.2,
        'pairs_violation_rate_per_dc': -1.5,
        'tuple_violation_rate_per_dc': -1.5,
        'succinctness_per_dc': 1.5,
        'coverage_per_dc': 4.0,
    }
    dc_constraints_eval['weighted_score'] = 0
    for col_name, weight in col_to_weight.items():
        dc_constraints_eval['weighted_score'] += weight * dc_constraints_eval[col_name]

    return dc_constraints_eval


def load_evaluated_dcs(eval_csv_out_path: str):
    dc_constraints_eval = pd.read_csv(eval_csv_out_path, converters={'best_other_tuples': literal_eval})

    return dc_constraints_eval
