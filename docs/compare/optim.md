# Optim: machine learning against online learning

The machine learning classes read the whole table and the online classes read one row at a time. They are two different parts of the library; this page runs both on the same synthetic data and reports how far apart their outputs are.

An optimiser moves the weights downhill on a loss. These classes take the gradient of the loss over the whole dataset and apply the update rule of the paper, step after step, so they can minimise any loss whose gradient you can write.

## The difference, on synthetic data

10 seeds. `abs` is the largest absolute difference between the two outputs, `rel` the largest relative difference; median, 90th percentile and maximum over the seeds.

| class | kind | median | p90 | max | why |
|---|---|---|---|---|---|
| `SGD` | abs | 0 | 0 | 0 | the same update rule applied to the same gradient |
| `AdaGrad` | abs | 0 | 0 | 0 | the same update rule applied to the same gradient |
| `Adam` | abs | 0 | 0 | 0 | the same update rule applied to the same gradient |
| `FTRLProximal` | abs | 0 | 0 | 0 | the same update rule applied to the same gradient |
| `OnlineNewtonStep` | abs | 1.81e-10 | 5.88e-09 | 3.85e-08 | the same update rule applied to the same gradient |

Wilcoxon and A12: not computed: the machine learning and the online value are two outputs of the same formula on the same data, so the table gives their difference and not a ranking of the two.

## Run it

~~~bash
python benchmarks/compare/optim.py
~~~

The command writes `benchmarks/compare/optim.json`, the table above.
