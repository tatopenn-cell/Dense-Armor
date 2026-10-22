# Machine learning

The classes of `dense_armor.ml` read a whole table at once. `fit(X, y)` takes the rows, leaves the results in attributes with a trailing underscore (`mean_`, `classes_`, `components_`), and the class then answers with `predict`, `transform` or `score`. The data in every block of these pages are `synthetic`.

Each family below has its own page, with a self-contained block for every class and the output it prints.

| Family | What it holds |
|---|---|
| [Statistics and sketches](stats.md) | A statistic such as a mean, a variance or a correlation is computed on a whole table at once. |
| [Preprocessing](preprocessing.md) | Before a model sees the data, the table is rescaled, some columns are dropped, the joint records are turned into velocities and the text is turned into numbers. |
| [Linear models](linear.md) | A linear model predicts a number, or the probability of a class, as a weighted sum of the features. |
| [Naive Bayes](bayes.md) | A naive Bayes classifier counts how often each feature value goes with each class and turns the counts into probabilities. |
| [Nearest neighbours](neighbors.md) | A nearest-neighbour model answers a question about a new row by looking at the rows of the table that are closest to it: the most common class among them, or the average of their targets. |
| [Multiclass](multiclass.md) | A binary classifier answers yes or no. To choose among many classes, these strategies train several binary classifiers on the whole table and combine their answers: one against all the others, one for every pair of classes, or one for every bit of a codeword. |
| [Multi-output](multioutput.md) | Some problems have several targets at once, for example the position and the speed of a joint. |
| [Distributions](proba.md) | A distribution summarises a sample in a few numbers. These classes keep the posterior of the Beta, Gaussian and Multinomial families: the prior counts plus the counts observed in the whole sample. |
| [Decision trees](tree.md) | A tree asks a series of yes/no questions about the features and ends in a leaf that gives the answer. |
| [Forests](forest.md) | A forest asks many trees and averages their answers. RandomForestClassifier grows each tree on a bootstrap sample with a random subset of features; ExtraTreesRegressor uses the whole sample and a random threshold at each split. |
| [Ensembles](ensemble.md) | An ensemble combines several models: bagging and boosting train many copies, voting and stacking combine different models, and the exponentially weighted average follows the member with the smallest loss. |
| [Anomaly detection](anomaly.md) | An anomaly detector marks the values that do not look like their neighbours. |
| [Change detection](drift.md) | A change detector reads a series of values and says where its level shifts. |
| [Vision](vision.md) | Vision here works on frames and on tables of features. DenseFlow gives the motion of every pixel between two frames, MotionMask keeps the pixels that move most, GlobalMotion gives the shift of the whole scene, RandomProjection and PCA reduce a table of features to fewer columns. |

A stream that arrives one row at a time is the subject of the tab Online learning; the section Comparisons & limits of the tab Library puts the two side by side.
