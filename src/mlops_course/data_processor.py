"""Provding functions or classes to pre-process data before model training."""

import datetime

import pandas as pd

from mlops_course.config import SelectionConfig

# 1 if not initial input exists
#   bootstrap (use pre_processor)
#
# don't want to pickle -> read max day from a config file -> this is just fro reproducability
# extract current max date from data
# generate matches for date +1 (append) to boostrapped table
# generate results for all matches not yet having a result and older than date+1 update table
# split now into train val and test, based on max_date
# update train set
# remove and attach from validation set
# remove and attach from test set


def pre_processor(df: pd.DataFrame, selection_config: SelectionConfig) -> pd.DataFrame:
    """Pre-processes the parsed data frame.

    Comments:
    - It's unclear whether we need to include the LabelEncoder already here.
    - TODO: more sophisticated feature computation
    """
    features = [selection_config.date_column] + selection_config.features + [selection_config.target]
    return df.filter(items=features)


def basic_temporal_train_test_split(
    df: pd.DataFrame, last_training_day: datetime.date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a basic temporal train test split.

    We exploit that dates can be considered as normalized timestamps
    """
    return df.query("date <= @last_training_day"), df.query("date > @last_training_day")


def extract_distributions(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Extract distribution from cs go data columns."""
    # this is difficult to make config dependent, so we hard code it
    team_dist = df[["team_1", "team_2"]].stack().value_counts(normalize=True).rename_axis(index="teams")
    map_dist = df[["map_name"]].value_counts(normalize=True).rename_axis(index="map_name")
    starting_ct_dist = df[["starting_ct"]].value_counts(normalize=True).rename_axis(index="starting_ct")
    ranks_dist = df[["rank_1", "rank_2"]].stack().value_counts(normalize=True).rename_axis(index="rank")
    map_winners_dist = df[["map_winner"]].value_counts(normalize=True).rename_axis(index="map_winner")
    return {
        "team": team_dist,
        "map": map_dist,
        "starting_ct": starting_ct_dist,
        "ranks": ranks_dist,
        "winners": map_winners_dist,
    }


def sample_match(dists: dict[str, pd.Series]) -> pd.Series:
    """Sample single match from distirbutions."""
    teams = list(dists["team"].sample(2, weights=dists["team"]).index.tolist())
    map_name = [elem[0] for elem in dists["map"].sample(1, weights=dists["map"]).index.tolist()]
    starting_ct = [elem[0] for elem in dists["starting_ct"].sample(1, weights=dists["starting_ct"]).index.tolist()]
    ranks = list(dists["ranks"].sample(2, weights=dists["ranks"]).index.tolist())
    return pd.Series(
        teams + map_name + starting_ct + ranks,
        index=["team_1", "team_2", "map_name", "starting_ct", "rank_1", "rank_2"],
    )


def sample_matches(n: int, dists: dict[str, pd.Series]) -> pd.DataFrame:
    """Sample multiple matches."""
    matches = [sample_match(dists).to_frame().T for _ in range(n)]
    return pd.concat(matches, axis=0)


def sample_outcomes(n: int, dists: dict[str, pd.Series]) -> list[int]:
    """Sample outcomes of matches."""
    return [elem[0] for elem in dists["winners"].sample(n, replace=True).index.tolist()]


def drift_dists(dists: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Introduce drift in distributions."""
    # we just switch the probabilities of a team, map and rank to depend on their lexicographical order
    drifted_teams = (
        dists["team"]
        .to_frame()
        .reset_index()
        .assign(teams=lambda df: df["teams"].sort_values().to_numpy())
        .set_index("teams")["proportion"]
    )
    drifted_map = (
        dists["map"]
        .to_frame()
        .reset_index()
        .assign(map_name=lambda df: df["map_name"].sort_values().to_numpy())
        .set_index("map_name")["proportion"]
    )
    drifted_ranks = (
        dists["ranks"]
        .to_frame()
        .reset_index()
        .assign(map_name=lambda df: df["rank"].sort_values().to_numpy())
        .set_index("rank")["proportion"]
    )

    # Change probabilities of starting cts and winning classes
    drifted_starting_ct = dists["starting_ct"].copy()
    drifted_starting_ct.update(pd.Series([0.8, 0.2], index=[1, 2]))

    drifted_winners = dists["winners"].copy()
    drifted_winners.update(pd.Series([0.65, 0.35], index=[1, 2]))

    return {
        "team": drifted_teams,
        "map": drifted_map,
        "starting_ct": drifted_starting_ct,
        "ranks": drifted_ranks,
        "winners": drifted_winners,
    }
