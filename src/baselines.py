"""Baseline aggregation rules: Solo Agent and Unweighted Majority Voting (UMV)."""

from collections import Counter
from typing import Iterable, List, TypeVar

T = TypeVar("T")


def unweighted_majority_vote(votes: Iterable[T]) -> T:
    """Return the mode of committee votes (Condorcet Jury baseline)."""
    vote_list: List[T] = list(votes)
    if not vote_list:
        raise ValueError("Cannot compute majority vote over empty iterable.")
    return Counter(vote_list).most_common(1)[0][0]
