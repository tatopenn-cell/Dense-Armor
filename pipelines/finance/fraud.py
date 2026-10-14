"""Fraud detection on a streaming transaction feed.

Chains two online trees on the transaction features. Both are
trained online, one transaction at a time, on the same stream.
The pipeline returns the two scores and their average.

- **MondrianForestClassifier** (Lakshminarayanan et al. 2014, 2016):
  a random hierarchical partition of the feature space, splits
  sampled with rate proportional to the extent of each block.
  On the OpenML creditcard dataset it reaches 0.969 ROC-AUC.
- **SGTClassifier** (Gouk et al. 2019): incremental decision tree
  with Newton-step leaves and t-test splits. On the same dataset
  it reaches 0.926 ROC-AUC.

Two independent online learners, both trained on the same stream,
neither of which needs a second pass. See the algorithm pages for
the maths.
"""

from dataclasses import dataclass

from dense_armor.utility.tree.mondrian import MondrianForestClassifier
from dense_armor.utility.tree.sgt import SGTClassifier


@dataclass
class TransactionResult:
    """Verdict on one transaction.

    Attributes:
        index: progressive index in the pipeline run.
        score_mondrian: probability of fraud from the Mondrian forest.
        score_sgt: probability of fraud from the SGD tree.
        score_ensemble: average of the two.
        flagged: True if the ensemble score is above the threshold.
    """

    index: int
    score_mondrian: float
    score_sgt: float
    score_ensemble: float
    flagged: bool


class FraudPipeline:
    """Two online trees on the same transaction stream.

    Args:
        n_trees: number of trees in the Mondrian forest. Default 10.
        min_samples_split_sgt: min_samples_split for the SGD tree.
            Default 50.
        threshold: ensemble score above which a transaction is
            flagged. Default 0.5.
        seed: random seed for the Mondrian forest.
    """

    def __init__(
        self,
        n_trees: int = 10,
        min_samples_split_sgt: int = 50,
        threshold: float = 0.5,
        seed: int = 0,
    ):
        self.mondrian = MondrianForestClassifier(n_trees=n_trees, seed=seed)
        self.sgt = SGTClassifier(min_samples_split=min_samples_split_sgt,
                                 n_bins=32)
        self.threshold = float(threshold)
        self.n_transactions = 0

    def learn_one(self, x: dict, y: int):
        self.mondrian.learn_one(x, y=y)
        self.sgt.learn_one(x, y=y)
        self.n_transactions += 1
        return self

    def score_one(self, x: dict) -> TransactionResult:
        s_m = float(self.mondrian.predict_proba_one(x).get(1, 0.0))
        s_s = float(self.sgt.predict_proba_one(x).get(1, 0.0))
        s_e = 0.5 * (s_m + s_s)
        return TransactionResult(
            index=self.n_transactions,
            score_mondrian=s_m,
            score_sgt=s_s,
            score_ensemble=s_e,
            flagged=bool(s_e >= self.threshold),
        )
