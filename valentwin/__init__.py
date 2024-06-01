# Ported from Valentine: https://github.com/delftdata/valentine
# Added multiprocessing and pairwise matching

from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
from typing import Iterable, List, Sequence, Union

import pandas as pd

import valentwin.algorithms
import valentwin.data_sources


class NotAValentineMatcher(Exception):
    pass


def validate_matcher(matcher):
    if not isinstance(matcher, valentwin.algorithms.BaseMatcher):
        raise NotAValentineMatcher('The method that you selected is not supported by Valentine')


def valentine_match(df1: pd.DataFrame,
                    df2: pd.DataFrame,
                    matcher: valentwin.algorithms.BaseMatcher,
                    df1_name: str = 'table_1',
                    df2_name: str = 'table_2'):

    validate_matcher(matcher)

    table_1 = valentwin.data_sources.DataframeTable(df1, name=df1_name)
    table_2 = valentwin.data_sources.DataframeTable(df2, name=df2_name)
    matches = dict(sorted(matcher.get_matches(table_1, table_2).items(),
                          key=lambda item: item[1], reverse=True))

    return matches


def process_pair_matching(args):
    df1, df2, table_1_name, table_2_name, matcher = args
    table_1 = valentwin.data_sources.DataframeTable(df1, name=table_1_name)
    table_2 = valentwin.data_sources.DataframeTable(df2, name=table_2_name)

    # Perform the matching operation and return the result along with the pair names
    match_result = matcher.get_matches(table_1, table_2)
    return ((table_1_name, table_2_name), match_result)


def valentine_match_batch(df_iter_1: Iterable[pd.DataFrame],
                          df_iter_2: Iterable[pd.DataFrame],
                          matcher: valentwin.algorithms.BaseMatcher,
                          df_iter_1_names: Union[List[str], None] = None,
                          df_iter_2_names: Union[List[str], None] = None,
                          parallelize: bool = False):

    validate_matcher(matcher)

    matches = {}

    seen_pairs = set()

    if parallelize:
        task_inputs = []
        seen_pairs = set()

        for df1_idx, df1 in enumerate(df_iter_1):
            table_1_name = df_iter_1_names[df1_idx] if df_iter_1_names is not None else f'table_1_{df1_idx}'
            for df2_idx, df2 in enumerate(df_iter_2):
                table_2_name = df_iter_2_names[df2_idx] if df_iter_2_names is not None else f'table_2_{df2_idx}'
                if (table_1_name, table_2_name) in seen_pairs or table_1_name == table_2_name:
                    continue
                seen_pairs.add((table_1_name, table_2_name))
                seen_pairs.add((table_2_name, table_1_name))
                task_inputs.append((df1, df2, table_1_name, table_2_name, matcher))

        matches = {}
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_pair_matching, args) for args in task_inputs]
            for future in tqdm(futures):
                pair, result = future.result()
                matches.update(result)

    else:
        for df1_idx, df1 in enumerate(tqdm(df_iter_1)):
            for df2_idx, df2 in enumerate(tqdm(df_iter_2)):
                table_1_name = df_iter_1_names[df1_idx] if df_iter_1_names is not None else f'table_1_{df1_idx}'
                table_2_name = df_iter_2_names[df2_idx] if df_iter_2_names is not None else f'table_2_{df2_idx}'
                if (table_1_name, table_2_name) in seen_pairs:
                    continue

                table_1 = valentwin.data_sources.DataframeTable(df1, name=table_1_name)
                table_2 = valentwin.data_sources.DataframeTable(df2, name=table_2_name)
                if table_1_name == table_2_name:
                    continue
                matches.update(matcher.get_matches(table_1, table_2))
                seen_pairs.add((table_1_name, table_2_name))
                seen_pairs.add((table_2_name, table_1_name))

    matches = dict(sorted(matches.items(), key=lambda item: item[1], reverse=True))

    return matches


def valentine_match_pairwise(df_iter: Sequence[pd.DataFrame],
                             matcher: valentwin.algorithms.BaseMatcher,
                             df_iter_names: Union[List[str], None] = None,
                             train_df_iter: Sequence[pd.DataFrame] = None,
                             parallelize: bool = False,
                             holistic: bool = False):

    if isinstance(matcher, valentwin.algorithms.alite.alite.ALITE):
        matches = matcher.get_matches_from_batch(df_iter, df_iter_names)
    elif isinstance(matcher, valentwin.algorithms.sima.sima.SiMa):
        matches = matcher.get_matches_from_batch(df_iter, df_iter_names)
    elif isinstance(matcher, valentwin.algorithms.omnimatch.omnimatch.OmniMatch):
        matches = matcher.get_matches_from_batch(df_iter, table_names=df_iter_names, train_tables=train_df_iter)
    elif isinstance(matcher, valentwin.algorithms.valentwin.valentwin.ValenTwin) and holistic:
        matches = matcher.get_matches_from_batch(df_iter, df_iter_names)
    else:
        matches = valentine_match_batch(df_iter_1=df_iter, df_iter_2=df_iter, matcher=matcher,
                                        df_iter_1_names=df_iter_names, df_iter_2_names=df_iter_names,
                                        parallelize=parallelize)

    # filter matches to only include matches between different tables
    matches = {k: v for k, v in matches.items() if k[0][0] != k[1][0]}

    # Only keep unique pairs: (A, B) would already be in the matches, remove (B, A)
    unique_matches_dict = {}
    for pair, similarity in matches.items():
        # Check if the reverse of the current pair is not in the new dictionary
        if (pair[1], pair[0]) + pair[2:] not in unique_matches_dict:
            # Add the current pair to the new dictionary
            unique_matches_dict[pair] = similarity

    return unique_matches_dict

