# Riferimenti bibliografici

**262 citazioni** in **120 moduli**, **230 paper unici**.

Estratto automaticamente dai docstring dei moduli.


## Per modulo


### `dense_armor/dynamics/six_dof_pbc_cbf_controller.py`

- **McClamroch (2010)**

### `dense_armor/ml/anomaly.py`

- **Ng, R. T., Sander, J. (2000)** — LOF:
        Identifying Density-Based Local Outliers
- **Chauvenet (1863)**
- **Tukey (1977)**
- **Hampel (1974)**
- **Davies and Gather (1993)**

### `dense_armor/ml/bayes.py`

- **McCallum, A., Nigam, K. (1998)** — A comparison of event models for naive
        Bayes text classification, page 4 (multinomial and Bernoulli models)
        (not in the folder, bibliographic only)
- **Rennie, J. D. M. et al. (2003)** — Tackling the poor assumptions of naive
        Bayes text classifiers, pages 6-7 (complement naive Bayes) (not in
        the folder, bibliographic only)

### `dense_armor/ml/drift.py`

- **Page, E. S. (1954)** — Continuous inspection schemes, Biometrika 41(1-2):
        100-115: the CUSUM with a fixed reference decided once from an
        in-control baseline (not in the folder, bibliographic only)
- **Hawkins, D. M., Olwell, D. H. (1998)** — Cumulative Sum Charts and Charting
        for Quality Improvement, Springer: the slack ``k`` as half the shift to
        detect (not in the folder, bibliographic only)

### `dense_armor/ml/ensemble.py`

- **Breiman, L. (1996)** — Bagging predictors, Machine Learning 24(2): bootstrap
        samples and the mean of the models (not in the folder, bibliographic
        only)
- **Freund, Y., Schapire, R. E. (1997)** — A decision-theoretic generalization of
        on-line learning and an application to boosting, JCSS 55(1): the
        Hedge weights ``exp(-eta L)`` and AdaBoost (not in the folder,
        bibliographic only)
- **Zhu, J. et al. (2009)** — Multi-class AdaBoost
        (SAMME) for the multiclass weights (not in the folder, bibliographic
        only)
- **Wolpert, D. H. (1992)** — Stacked generalization, Neural Networks 5(2): a
        meta-model on out-of-fold predictions (not in the folder,
        bibliographic only)

### `dense_armor/ml/forest.py`

- **Breiman, L. (2001)** — Random forests, Machine Learning 45(1): a tree per
        bootstrap sample, a random subset of features at each split (not in
        the folder, bibliographic only)
- **Geurts, P., Ernst, D., Wehenkel, L. (2006)** — Extremely randomized trees,
        Machine Learning 63(1): the whole sample and a random threshold at
        each split (not in the folder, bibliographic only)

### `dense_armor/ml/linear.py`

- **Bishop, C. M. (2006)** — Pattern Recognition and Machine Learning, Eq

### `dense_armor/ml/multiclass.py`

- **Rifkin, R., Klautau, A. (2004)** — In defense of one-vs-all classification:
        one binary problem per class (not in the folder, bibliographic only)
- **Furnkranz, J. (2002)** — Round robin classification, JMLR 2: one binary problem
        per pair of classes, decided by votes (not in the folder,
        bibliographic only)
- **Dietterich, T. G., Bakiri, G. (1995)** — Solving multiclass learning problems
        via error-correcting output codes, JAIR 2: a codeword per class and the
        closest codeword in Hamming distance (not in the folder, bibliographic
        only)

### `dense_armor/ml/multioutput.py`

- **Spyromitros-Xioufis, E. et al. (2016)** — Multi-target regression via input
        space expansion, Section 2 (not in the folder, bibliographic only)
- **Read, J. et al. (2009)** — Classifier chains for multi-label classification
        (not in the folder, bibliographic only)
- **Dembczynski, K., Cheng, W., Hullermeier, E. (2010)** — Bayes optimal
        multilabel classification via probabilistic classifier chains, Eq

### `dense_armor/ml/neighbors.py`

- **Cover, T., Hart, P. (1967)** — Nearest neighbor pattern classification,
        IEEE Trans

### `dense_armor/ml/optim.py`

- **Orabona, F. (2019)** — A modern introduction to online learning, Algorithms 2
- **Kingma, D. P., Ba, J. (2015)** — Adam, Algorithm 1; Hazan, E
- **Hazan, E.,
        Agarwal, A., Kale, S. (2007)** — Logarithmic regret algorithms for online
        convex optimization, Algorithm 2: update rules applied to the gradient
        of one example at a time (not in the folder, bibliographic only)

### `dense_armor/ml/preprocessing.py`

- **Silajev (2026)**
- **Weinberger, K. et al. (2009)** — Feature hashing for large scale multitask
        learning, Theorem 3: the signed hashing trick (not in the folder,
        bibliographic only)
- **Malialis, K. et al. (2022)** — Queue-based resampling for online class
        imbalance learning: a sequential method over a stream, so
        ``QueueResampler`` is declared in ``STOPPED`` (not in the folder,
        bibliographic only)

### `dense_armor/ml/proba.py`

- **Murphy, K. P. (2007)** — Conjugate Bayesian analysis of the Gaussian
        distribution (not in the folder, bibliographic only)

### `dense_armor/ml/rules.py`

- **Duarte, J., Gama, J., Bifet, A. (2016)** — Adaptive model rules from
        high-speed data streams, ACM TKDD 10(3): the rule set grows and is
        revised as examples arrive (not in the folder, bibliographic only)

### `dense_armor/ml/stats.py`

- **Chan, T. F., Golub, G. H., LeVeque, R. J. (1979)** — Updating formulae and a pairwise
        algorithm for computing sample variances, Compstat: the two-pass formulas for the
        mean, the variance and the covariance (not in the folder, bibliographic only)
- **West, D. H. D. (1979)** — Updating mean and variance estimates, CACM 22(9):
        the exponentially weighted recursion (not in the folder,
        bibliographic only)
- **Bodnar, T., Gupta, A. K., Parolya, N. (2014)** — Optimal linear shrinkage
        estimator for large dimensional precision matrix, Eq

### `dense_armor/ml/tree.py`

- **Breiman, L., Friedman, J., Olshen, R., Stone, C. (1984)** — Classification and
        Regression Trees, Wadsworth, chapter 2: the gini impurity and the
        recursive binary splitting (not in the folder, bibliographic only)
- **Lakshminarayanan, B., Roy, D. M., Teh, Y. W. (2014)** — Mondrian forests:
        efficient online random forests, NIPS, section 2, Algorithms 1 and 2
        (``SampleMondrianTree`` and ``SampleMondrianBlock``, page 4)
- **Lakshminarayanan, B., Roy, D. M., Teh, Y. W. (2016)** — Mondrian forests for
        large-scale regression when uncertainty matters, AISTATS, Algorithm 5 and
        eq

### `dense_armor/ml/vision.py`

- **Horn, B. K., Schunck, B. G. (1981)** — Determining optical flow ([arXiv:2511.16535](https://arxiv.org/abs/2511.16535))
- **Ziani (2025)** — Methods to a Multiresolution Horn-Schunck ([arXiv:2511.16535](https://arxiv.org/abs/2511.16535))
- **Johnson, W. B., Lindenstrauss, J. (1984)** — and Ghojogh et al
- **Ghojogh et al. (2021)**
- **Balsubramani, A., Dasgupta, S., Freund, Y. (2013)** — The fast convergence of
        incremental PCA, NIPS: the target of the rule is the top eigenvector of
        the covariance (not in the folder, bibliographic only)

### `dense_armor/roles/physics.py`

- **Fitzpatrick, R. (2008)** — Maxwell's equations and the principles of
    electromagnetism

### `dense_armor/roles/realtime.py`

- **West, D. H. D. (1979)** — Updating mean and variance estimates: an
    improved method

### `dense_armor/roles/safety.py`

- **Page, E. S. (1954)** — Continuous inspection schemes

### `dense_armor/roles/uncertainty.py`

- **Candes (2021)** — The coverage level is
  re-estimated at every sample from the recent miscoverage frequency,
  so the interval tracks a drifting stream
- **Ramdas (2023)**
- **Gibbs, I., Candes, E. J. (2021)** — Adaptive conformal inference under
    distribution shift ([arXiv:2305.00070](https://arxiv.org/abs/2305.00070))
- **Gupta, C., Ramdas, A. (2023)** — Online Platt scaling with calibeating ([arXiv:2305.00070](https://arxiv.org/abs/2305.00070))

### `dense_armor/utility/deep/neural/activations.py`

- **He, K., Zhang, X., Ren, S., Sun, J. (2015)** — Delving Deep into
        Rectifiers ([arXiv:1502.01852](https://arxiv.org/abs/1502.01852))

### `dense_armor/utility/deep/neural/activations_more.py`

- **Hendrycks, D., Gimpel, K. (2016)** — Gaussian Error Linear Units
        (GELUs) ([arXiv:1606.08415](https://arxiv.org/abs/1606.08415))
- **Elfwing, S., Uchibe, E., Doya, K. (2017)** — Sigmoid-weighted linear
        units for neural network function approximation in
        reinforcement learning ([arXiv:1702.03118](https://arxiv.org/abs/1702.03118))
- **Unterthiner, T., Hochreiter, S. (2016)** — Fast and
        accurate deep network learning by exponential linear units
        (ELUs) ([arXiv:1511.07289](https://arxiv.org/abs/1511.07289))
- **He, K., Zhang, X., Ren, S., Sun, J. (2015)** — Delving deep into
        rectifiers ([arXiv:1502.01852](https://arxiv.org/abs/1502.01852))
- **Dugas et al. (2001)** — is not in this folder and is cited as
    ``bibliographic only``

### `dense_armor/utility/deep/neural/chain.py`

- **He, K., Zhang, X., Ren, S., Sun, J. (2016)** — Deep Residual Learning
        for Image Recognition ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385))

### `dense_armor/utility/deep/neural/conv_more.py`

- **Tran, D., Bourdev, L., Fergus, R., Torresani, L., Paluri, M. (2015)** — Learning spatiotemporal features with 3D convolutional
        networks ([arXiv:1412.0767](https://arxiv.org/abs/1412.0767))
- **Chollet, F. (2017)** — Xception: Deep learning with depthwise
        separable convolutions ([arXiv:1610.02357](https://arxiv.org/abs/1610.02357))

### `dense_armor/utility/deep/neural/convtranspose.py`

- **Dumoulin and Visin (2016)**

### `dense_armor/utility/deep/neural/data_parallel.py`

- **Goyal et al. (2017)**

### `dense_armor/utility/deep/neural/init.py`

- **He, K., Zhang, X., Ren, S., Sun, J. (2015)** — Delving Deep into
        Rectifiers: Surpassing Human-Level Performance on ImageNet
        Classification ([arXiv:1502.01852](https://arxiv.org/abs/1502.01852))

### `dense_armor/utility/deep/neural/layers.py`

- **He, K., Zhang, X., Ren, S., Sun, J. (2016)** — Deep Residual Learning
        for Image Recognition ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385))
- **Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever, I.,
        Salakhutdinov, R. R. (2012)** — Improving neural networks by
        preventing co-adaptation of feature detectors ([arXiv:1207.0580](https://arxiv.org/abs/1207.0580))

### `dense_armor/utility/deep/neural/losses_more.py`

- **Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., Wojna, Z. (2016)** — Rethinking the Inception Architecture for Computer
        Vision ([arXiv:1512.00567](https://arxiv.org/abs/1512.00567))
- **Goyal, P., Girshick, R., He, K., Dollar, P. (2017)** — Focal Loss for Dense Object Detection ([arXiv:1708.02002](https://arxiv.org/abs/1708.02002))

### `dense_armor/utility/deep/neural/models.py`

- **He, K., Zhang, X., Ren, S., Sun, J. (2016)** — Deep Residual Learning
        for Image Recognition ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385))
- **He, K., Zhang, X., Ren, S., Sun, J. (2015)** — Delving Deep into
        Rectifiers ([arXiv:1502.01852](https://arxiv.org/abs/1502.01852))

### `dense_armor/utility/deep/neural/moe.py`

- **Shazeer et al. (2017)** — use a noisy top-k
gating (Eq
- **Shazeer
et al. (2017)**
- **Fedus et al. (2021)** — uses top-1, where the weight is 1 by construction
- **Fedus et al. (2021)**

### `dense_armor/utility/deep/neural/norm.py`

- **Ioffe, S., Szegedy, C. (2015)** — Batch Normalization: Accelerating
        Deep Network Training by Reducing Internal Covariate Shift ([arXiv:1502.03167](https://arxiv.org/abs/1502.03167))
- **Ba, J. L., Kiros, J. R., Hinton, G. E. (2016)** — Layer Normalization ([arXiv:1607.06450](https://arxiv.org/abs/1607.06450))

### `dense_armor/utility/deep/neural/norm_more.py`

- **Wu, Y., He, K. (2018)** — Group Normalization ([arXiv:1803.08494](https://arxiv.org/abs/1803.08494))
- **Ulyanov, D., Vedaldi, A., Lempitsky, V. (2016)** — Instance
        Normalization: The Missing Ingredient for Fast Stylization ([arXiv:1607.08022](https://arxiv.org/abs/1607.08022))

### `dense_armor/utility/deep/neural/online_autoencoder.py`

- **Chalapathy and Chawla (2019)**

### `dense_armor/utility/deep/neural/positions.py`

- **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
        Gomez, A. N., Kaiser, L., Polosukhin, I. (2017)** — Attention Is
        All You Need ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762))

### `dense_armor/utility/evaluation/evaluate/events.py`

- **Tatbul, N., Lee, T. J., Zdonik, S., Alam, M., Gottschlich, J. (2018)** — Precision and recall for time series ([arXiv:1803.03639](https://arxiv.org/abs/1803.03639))
- **Lavin, A., Ahmad, S. (2015)** — Evaluating real-time anomaly detection
    algorithms --- the Numenta Anomaly Benchmark ([arXiv:1510.03336](https://arxiv.org/abs/1510.03336))

### `dense_armor/utility/evaluation/evaluate/model_selection.py`

- **Ksieniewicz, P., Zyblewski, P. (2020)** — Stream-learn: open-source Python
    library for difficult data stream batch analysis ([arXiv:2001.11077](https://arxiv.org/abs/2001.11077))

### `dense_armor/utility/evaluation/evaluate/prequential.py`

- **Amekoe et al. (2024)** — References
----------
Ksieniewicz, P ([arXiv:2001.11077](https://arxiv.org/abs/2001.11077))
- **Ksieniewicz, P., Zyblewski, P. (2020)** — Stream-learn ([arXiv:2001.11077](https://arxiv.org/abs/2001.11077))
- **Dagdia, Z. (2024)** — arXiv:2409 ([arXiv:2409.10111](https://arxiv.org/abs/2409.10111))

### `dense_armor/utility/evaluation/metrics/classification.py`

- **Brier, G. W. (1950)** — Verification of forecasts expressed in terms of
    probability
- **Cohen, J. (1960)** — A coefficient of agreement for nominal scales
- **Sasaki, Y. (2007)** — The truth of the F-measure

### `dense_armor/utility/evaluation/metrics/events.py`

- **Tatbul, N., Lee, T. J., Zdonik, S., Alam, M., Gottschlich, J. (2018)** — Precision and recall for time series ([arXiv:2109.05257](https://arxiv.org/abs/2109.05257))
- **Lavin, A., Ahmad, S. (2015)** — Evaluating real-time anomaly detection
    algorithms --- the Numenta Anomaly Benchmark ([arXiv:2109.05257](https://arxiv.org/abs/2109.05257))
- **Lee, B., Yoon, S. (2021)** — Towards a
    rigorous evaluation of time-series anomaly detection ([arXiv:2109.05257](https://arxiv.org/abs/2109.05257))
- **Xu, H. et al. (2018)** — Unsupervised anomaly detection via variational
    auto-encoder for seasonal KPIs in web applications

### `dense_armor/utility/evaluation/metrics/regression.py`

- **Cook, R. D., Weisberg, S. (1982)** — Residuals and Influence in
    Regression
- **Good, I. J. (1952)** — Rational decisions
- **Welford, B. P. (1962)** — Technometrics 4(3), 419-420
- **Chan, T. F., Golub, G. H., LeVeque, R. J. (1979)** — In Compstat
- **Vovk, V., Gammerman, A., Shafer, G. (2005)** — Algorithmic Learning in a
    Random World

### `dense_armor/utility/evaluation/metrics/roc_auc.py`

- **Mann, H. B., Whitney, D. R. (1947)** — On a test of whether one of two
    random variables is stochastically larger than the other
- **Bamber, D. (1975)** — The area above the ordinal dominance graph and the
    area below the receiver operating characteristic graph

### `dense_armor/utility/online/active/strategies.py`

- **Zliobaite et al. (2011)**

### `dense_armor/utility/online/anomaly/hst.py`

- **Tan, S. C., Ting, K. M., Liu, T. F. (2011)** — Fast anomaly detection for
    streaming data ([arXiv:2505.08489](https://arxiv.org/abs/2505.08489))
- **Oplatkova, Z., Vala, R. (2025)** — Isolation Forest in Novelty Detection Scenario ([arXiv:2505.08489](https://arxiv.org/abs/2505.08489))

### `dense_armor/utility/online/anomaly/loda.py`

- **Lou et al. (2024)**
- **Pevny (2016)** — Details not fixed by Lou et al
- **Lou et al. (2024)**
- **Lou, B., Boland, D., Leong, P. H. W. (2024)** — fSEAD: a Composable
    FPGA-based Streaming Ensemble Anomaly Detection Library ([arXiv:2406.05999](https://arxiv.org/abs/2406.05999))
- **Pevny, T. (2016)** — LODA: Lightweight on-line detector of anomalies

### `dense_armor/utility/online/anomaly/lof.py`

- **Breunig et al. (2000)**
- **Ng, R. T., Sander, J. (2000)** — LOF: Identifying Density-Based Local Outliers

### `dense_armor/utility/online/anomaly/mahalanobis.py`

- **Guillot, A., Godichon-Baggioni, A., Robin, S. (2025)** — Online robust
    covariance and outlier detection ([arXiv:2601.03957](https://arxiv.org/abs/2601.03957))
- **Cardot, H., Godichon-Baggioni, A. (2017)** — TEST 26(3), 461-480

### `dense_armor/utility/online/anomaly/ocsvm.py`

- **Suk and Kpotufe (2025)** — rewrites the OCSVM objective in an
empirical risk form, replaces the kernel with Random Fourier Features
(RFF) so gradient updates depend only on the current sample, adds a
strongly convex regularisation, and solves the resulting objective by
stochas
- **Suk, J., Kpotufe, S. (2025)** — An efficient variant of one-class SVM
    with lifelong online learning guarantees ([arXiv:2512.11052](https://arxiv.org/abs/2512.11052))
- **Rahimi, A., Recht, B. (2007)** — Random features for large-scale kernel
    machines

### `dense_armor/utility/online/anomaly/oiforest.py`

- **Leveni et al. (2024)**
- **Leveni, F., Cassales, G. W., Pfahringer, B., Bifet, A., Boracchi, G. (2024)** — Online Isolation Forest ([arXiv:2505.09593](https://arxiv.org/abs/2505.09593))

### `dense_armor/utility/online/anomaly/robust_filters.py`

- **Chauvenet (1863)**

### `dense_armor/utility/online/bandit/evaluate.py`

- **Li et al. (2011)**

### `dense_armor/utility/online/bandit/linucb.py`

- **Li et al. (2010)**

### `dense_armor/utility/online/bandit/ts.py`

- **Chapelle and Li (2011)**

### `dense_armor/utility/online/bandit/ucb.py`

- **Auer, Cesa-Bianchi and Fischer (2002)** — The formulas follow the paper as cited

### `dense_armor/utility/online/bayes/bernoulli.py`

- **McCallum and Nigam (1998)** — A sample is a binary vector over the vocabulary; the class-
conditional probability of a feature is smoothed by Laplace, eq

### `dense_armor/utility/online/bayes/complement.py`

- **Rennie, Shih, Teevan and Karger (2003)**

### `dense_armor/utility/online/bayes/multinomial.py`

- **McCallum and Nigam (1998)**

### `dense_armor/utility/online/cluster/denstream.py`

- **Zubaroglu, A., Atalay, V. (2020)** — Data stream clustering: a
        review ([arXiv:2007.10781](https://arxiv.org/abs/2007.10781))

### `dense_armor/utility/online/cluster/kmeans.py`

- **Bhattacharjee et al. (2021)** — compares the
total loss
- **Bhattacharjee, R., Dasgupta, S., Imola, J. J., Moshkovitz, M. (2021)** — Online k-means clustering on arbitrary data streams ([arXiv:2102.09101](https://arxiv.org/abs/2102.09101))
- **Roth, K. et al. (2021)** — Towards total recall in industrial anomaly
        detection ([arXiv:2106.08265](https://arxiv.org/abs/2106.08265))

### `dense_armor/utility/online/drift/bocpd.py`

- **MacKay, D. J. C. (2007)** — *Bayesian Online
Changepoint Detection ([arXiv:0710.3742](https://arxiv.org/abs/0710.3742))

### `dense_armor/utility/online/drift/cusum.py`

- **Page, E.S. (1954)**

### `dense_armor/utility/online/drift/page_hinkley.py`

- **Page, E. S. (1954)** — Continuous inspection schemes
- **Xie, Zou, Xie, Veeravalli (2021)**

### `dense_armor/utility/online/ensemble/bagging.py`

- **Oza, N. C., Russell, S. (2001)** — Online bagging and boosting

### `dense_armor/utility/online/ensemble/boosting.py`

- **Oza and Russell (2001)**
- **Oza, N. C., Russell, S. (2001)** — Online bagging and boosting
- **Freund, Y., Schapire, R. E. (1997)** — A decision-theoretic
        generalization of on-line learning and an application to
        boosting

### `dense_armor/utility/online/ensemble/ewa.py`

- **Freund and Schapire (1997)** — and the exponentiated
gradient of Kivinen and Warmuth (1997) give each member a weight
``w_i = exp(-eta * L_i)`` that decays with its cumulative loss ``L_i``;
the prediction is the weighted average

    y_hat = sum_i w_i y_hat_i / sum_i w_i
- **Kivinen and Warmuth (1997)** — give each member a weight
``w_i = exp(-eta * L_i)`` that decays with its cumulative loss ``L_i``;
the prediction is the weighted average

    y_hat = sum_i w_i y_hat_i / sum_i w_i
- **Freund, Y., Schapire, R. E. (1997)** — A decision-theoretic
        generalization of on-line learning and an application to
        boosting
- **Kivinen, J., Warmuth, M. K. (1997)** — Exponentiated gradient versus
        gradient descent for linear predictors

### `dense_armor/utility/online/facto/ffm.py`

- **Juan et al. (2016)**

### `dense_armor/utility/online/facto/fm.py`

- **Rendle (2010)**

### `dense_armor/utility/online/facto/fwfm.py`

- **Pan et al. (2018)**

### `dense_armor/utility/online/facto/hofm.py`

- **Blondel et al. (2016)**

### `dense_armor/utility/online/forest/amf.py`

- **Lakshminarayanan, B., Roy, D. M., Teh, Y. W. (2014)** — Mondrian
        forests: efficient online random forests

### `dense_armor/utility/online/forest/arf.py`

- **Gomes et al. (2017)** — A single Hoeffding tree follows a change through
one detector; the forest follows it through one detector per tree, and
the spread of the votes says how sure it is
- **Gomes, H. M., Bifet, A., Read, J., Barddal, J. P., Enembreck, F.,
        Pfahringer, B., Holmes, G., Abdessalem, T. (2017)** — Adaptive
        random forests for evolving data stream classification

### `dense_armor/utility/online/forest/oxt.py`

- **Mastelini, S. M. (2023)** — Efficient online tree, rule-based and distance-based
        algorithms

### `dense_armor/utility/online/learn/online_dynamics.py`

- **Ljung, L., Soderstrom, T. (1983)** — Theory and Practice of Recursive
Identification
- **Page, E. S. (1954)** — Continuous inspection schemes

### `dense_armor/utility/online/linear/bayes.py`

- **Bishop, C. M. (2006)** — Pattern Recognition and Machine Learning
- **Sherman, J., Morrison, W. J. (1950)** — Adjustment of an inverse
    matrix corresponding to a change in one element of a given matrix

### `dense_armor/utility/online/linear/glm.py`

- **Orabona, F. (2019)** — A Modern Introduction to Online Learning ([arXiv:1412.6980](https://arxiv.org/abs/1412.6980))
- **Kingma, D. P., Ba, J. (2014)** — Adam: a method for stochastic
    optimization ([arXiv:1412.6980](https://arxiv.org/abs/1412.6980))

### `dense_armor/utility/online/linear/payload.py`

- **Foster, J., McCrory, S., DeBuys, C., Bertrand, S., Griffin, R. (2024)** — Physically Consistent Online Inertial Adaptation for
    Humanoid Loco-manipulation ([arXiv:2405.07901](https://arxiv.org/abs/2405.07901))
- **Rucker, C., Wensing, P. M. (2022)** — Smooth parameterization of
    rigid-body inertia
- **Ljung, L., Soderstrom, T. (1983)** — Theory and Practice of
    Recursive Identification

### `dense_armor/utility/online/multiclass/multiclass.py`

- **Fuerkranz (2002)**
- **Fuerkranz (2002)**
- **Dietterich and Bakiri (1995)**

### `dense_armor/utility/online/multioutput/multioutput.py`

- **Spyromitros-Xioufis et al. (2016)**
- **Dembczynski et al. (2010)**
- **Spyromitros-Xioufis et al. (2016)**
- **Dembczynski et al. (2010)**
- **Dembczynski et al. (2010)**

### `dense_armor/utility/online/optim/__init__.py`

- **Orabona, F. (2019)** — A Modern Introduction to Online Learning ([arXiv:1412.6980](https://arxiv.org/abs/1412.6980))
- **Kingma, D. P., Ba, J. (2014)** — Adam: a method for stochastic
    optimization ([arXiv:1412.6980](https://arxiv.org/abs/1412.6980))

### `dense_armor/utility/online/proba/gaussian.py`

- **Welford (1962)**

### `dense_armor/utility/online/reco/reco.py`

- **Paterek (2007)** — and Koren (2009)
- **Koren (2009)**

### `dense_armor/utility/online/rules/amrules.py`

- **Duarte, J., Gama, J., Bifet, A. (2016)** — Adaptive model rules from
        high-speed data streams

### `dense_armor/utility/online/timeseries/ar.py`

- **Zaman et al. (2019)**
- **Ramos, L. M., Romero, D., Beferull-Lozano, B. (2019)** — Online topology identification from vector autoregressive
    time series ([arXiv:1904.01864](https://arxiv.org/abs/1904.01864))

### `dense_armor/utility/online/timeseries/garch.py`

- **Orlovits (2012)** — for the
recursive QML extension, but does not implement it
- **Bollerslev, T. (1986)** — Generalized autoregressive conditional
    heteroskedasticity ([DOI:10.1016/0304-4076(86](https://doi.org/10.1016/0304-4076(86))
- **Gerencsér, L., Orlovits, Z. (2012)** — Recursive estimation of GARCH
    processes

### `dense_armor/utility/online/timeseries/kalman.py`

- **Mbalawata, I. S., Sarkka, S., Vihola, M., Haario, H. (2013)** — Adaptive Metropolis algorithm using variational Bayesian adaptive
    Kalman filter ([arXiv:1308.5875](https://arxiv.org/abs/1308.5875))
- **Popescu, M., Mronga, D., Kumar, S., Kirchner, F. (2026)** — Residual-based adaptive Kalman filtering for legged robot state
    estimation ([arXiv:2608.02316](https://arxiv.org/abs/2608.02316))

### `dense_armor/utility/online/tree/adaptive.py`

- **Bifet and Gavalda (2009)** — adds a drift detector at each node:
when the detector warns, an alternate subtree grows in the background
on the recent samples; when the alternate is significantly more
accurate than the main one, it replaces the corresponding branch
- **Bifet, A., Gavalda, R. (2009)** — Adaptive learning from evolving
        data streams
- **Esteban, A., Cano, A., Zafra, A., Ventura, S. (2024)** — Hoeffding
        adaptive trees for multi-label classification on data streams
- **Domingos, P., Hulten, G. (2000)** — Mining high-speed data streams

### `dense_armor/utility/online/tree/batch.py`

- **Breiman, Friedman, Olshen, Stone (1984)** — *Classification and
Regression Trees

### `dense_armor/utility/online/tree/efdt.py`

- **Manapragada, C., Webb, G. I., Salehi, M. (2018)** — Extremely fast
        decision tree
- **Domingos, P., Hulten, G. (2000)** — Mining high-speed data streams

### `dense_armor/utility/online/tree/fimt.py`

- **Ikonomovska (2012)** — The algorithm is FIMT-DD of Ikonomovska (2012), Chapter 6 of the PhD
thesis, pages 68-77 of the printed pages
- **Ikonomovska (2012)**

### `dense_armor/utility/online/tree/hoeffding.py`

- **Manapragada, C., Webb, G. I., Salehi, M. (2018)** — Extremely fast
        decision tree
- **Domingos, P., Hulten, G. (2000)** — Mining high-speed data streams

### `dense_armor/utility/online/tree/mondrian.py`

- **Lakshminarayanan, B., Roy, D. M., Teh, Y. W. (2014)** — Mondrian
        forests: efficient online random forests
- **Lakshminarayanan, B., Roy, D. M., Teh, Y. W. (2016)** — Mondrian
        forests for large-scale regression when uncertainty matters
- **Roy, D. M., Teh, Y. W. (2009)** — The Mondrian process

### `dense_armor/utility/online/tree/sgt.py`

- **Gouk, H., Pfahringer, B., Frank, E. (2019)** — Stochastic gradient
        trees

### `dense_armor/utility/robot/control/cbf_filter.py`

- **Tabuada, P. (2019)** ([arXiv:1903.11199](https://arxiv.org/abs/1903.11199))

### `dense_armor/utility/robot/control/rate_limiter.py`

- **Kroger, T. (2021)** ([arXiv:2105.04830](https://arxiv.org/abs/2105.04830))

### `dense_armor/utility/signals/datasets/casper.py`

- **Kayan, H., Rana, O., Burnap, P., Perera, C. (2023)** ([arXiv:2303.01300](https://arxiv.org/abs/2303.01300))

### `dense_armor/utility/signals/datasets/drift_stream.py`

- **Ksieniewicz, P., Zyblewski, P. (2020)**

### `dense_armor/utility/signals/preprocessing/imbalance.py`

- **Malialis, Panayiotou, Polycarpou (2018)**
- **Malialis, K., Panayiotou, C., Polycarpou, M. M. (2018)** — Queue-based
    resampling for online class imbalance learning

### `dense_armor/utility/signals/preprocessing/vectorizer.py`

- **Silajev, I. (2026)** — *TF-IDF and BM25 as Kullback-Leibler
divergences ([arXiv:2609.14016](https://arxiv.org/abs/2609.14016))
- **Rennie et al. (2003)**

### `dense_armor/utility/signals/stats/dependence.py`

- **Welford, B. P. (1962)** — Note on a method for calculating corrected sums
    of squares and products
- **Chan, T. F., Golub, G. H., LeVeque, R. J. (1979)** — Updating formulae and
    a pairwise algorithm for computing sample variances
- **Chatfield, C. (2003)** — The Analysis of Time Series: An Introduction,
    6th ed

### `dense_armor/utility/signals/stats/mixture.py`

- **Moulines (2009)** — The
generic update, Eq
- **Cappé, O., Moulines, E. (2009)** — On-line expectation-maximization
    algorithm for latent data models
- **Neal, R. M., Hinton, G. E. (1998)** — A view of the EM algorithm that
    justifies incremental, sparse, and other variants

### `dense_armor/utility/signals/stats/moments.py`

- **Welford, B. P. (1962)** — Note on a method for calculating corrected sums
    of squares and products
- **Chan, T. F., Golub, G. H., LeVeque, R. J. (1979)** — Updating formulae and
    a pairwise algorithm for computing sample variances
- **Pébay, P. (2008)** — Formulas for robust, one-pass parallel computation of
    covariances and arbitrary-order statistical moments
- **West, D. H. D. (1979)** — Updating mean and variance estimates: an improved
    method

### `dense_armor/utility/signals/stats/quantiles.py`

- **Masson, C., Rim, J. E., Lee, H. K. (2019)** — DDSketch: a fast and
    fully-mergeable quantile sketch with relative-error guarantees ([arXiv:1902.04023](https://arxiv.org/abs/1902.04023))
- **Dunning, T., Ertl, O. (2019)** — Computing extremely accurate quantiles
    using t-digests ([arXiv:1902.04023](https://arxiv.org/abs/1902.04023))

### `dense_armor/utility/signals/stats/robust.py`

- **Hampel, F. R. (1974)** — The influence curve and its role in robust
    estimation
- **Tukey, J. W. (1977)** — Exploratory Data Analysis
- **Hyndman, R. J., Fan, Y. (1996)** — Sample quantiles in statistical
    packages

### `dense_armor/utility/signals/stats/sdft.py`

- **Jacobsen and Lyons (2003)**

### `dense_armor/utility/signals/stats/shrinkage.py`

- **Parolya, N. (2014)** — *On the
Strong Convergence of the Optimal Linear Shrinkage Estimator for Large
Dimensional Covariance Matrix ([arXiv:1308.2608](https://arxiv.org/abs/1308.2608))

### `dense_armor/utility/signals/stats/sketches.py`

- **Muthukrishnan (2005)** — Frequency
  estimation with ``eps · N`` one-sided error and probability
  ``1 - delta``, using ``w = ceil(e / eps)`` counters per row and
  ``d = ceil(ln(1 / delta))`` rows
- **Flajolet et al. (2007)** — Distinct count with about
  ``1
- **Bloom (1970)** — Membership test with no false
  negatives and a false positive rate of at most ``p_false``
- **Metwally et al. (2005)** — Top-k heavy hitters with a
  deterministic error bound: the reported count of the true k-th most
  frequent item is at least its true count minus ``n / k``
- **Cormode, G., Muthukrishnan, S. (2005)** — An improved data stream summary:
    the Count-Min sketch and its applications
- **Flajolet, P., Fusy, E., Gandouet, O., Meunier, F. (2007)** — HyperLogLog:
    the analysis of a near-optimal cardinality estimation algorithm
- **Bloom, B. H. (1970)** — Space/time trade-offs in hash coding with
    allowable errors
- **Abbadi, A. (2005)** — Efficient computation
    of frequent and top-k elements in data streams

### `dense_armor/utility/signals/stats/skyline.py`

- **Börzsönyi et al. (2001)** — and Tao and Papadias (2006)
- **Tao and Papadias (2006)** — Two definitions are used:

- dominance (Börzsönyi et al

### `dense_armor/utility/signals/stats/stable.py`

- **Ogita, T., Rump, S. M., Oishi, S. (2005)** — Accurate Sum and Dot
        Product
- **Shewchuk, J. R. (1997)** — Adaptive Precision Floating-Point Arithmetic
        and Fast Robust Geometric Predicates
- **Healy, J. (2022)** — Accurate and consistent calculation of the mean and
        variance in Monte-Carlo simulations ([arXiv:2206.10662](https://arxiv.org/abs/2206.10662))
- **Healy (2022)**
- **Chan, Golub and LeVeque (1983)** — for the choice
of K; the paper of 1983 was not read

### `dense_armor/utility/signals/stats/student_t.py`

- **Rubin (1995)** — is not implemented because the
paper is not in the folder and the reference paper in the folder
(Ng et al ([arXiv:2603.00269](https://arxiv.org/abs/2603.00269))
- **Ng, A., Zhu, S., Zhang, A. G., Reid, N. (2026)** — Robust Regression
    with Student's T: The Role of Degrees of Freedom ([arXiv:2603.00269](https://arxiv.org/abs/2603.00269)) ([DOI:10.1080/01621459](https://doi.org/10.1080/01621459))
- **Zellner, A. (1976)** — Bayesian and non-Bayesian analysis of the
    Student-t model ([DOI:10.1080/01621459](https://doi.org/10.1080/01621459))
- **Liu, C., Rubin, D. B. (1995)** — ML estimation of the t distribution with
    unknown degrees of freedom

### `dense_armor/utility/signals/vision/dense.py`

- **Horn and Schunck (1981)** — written by Ziani (2025, equation 6 and the Euler-Lagrange pair 7-8)
- **Horn, B. K., Schunck, B. G. (1981)** — Determining optical flow ([arXiv:2511.16535](https://arxiv.org/abs/2511.16535))
- **Ziani (2025)** — Methods to a Multiresolution Horn-Schunck ([arXiv:2511.16535](https://arxiv.org/abs/2511.16535))

### `dense_armor/utility/signals/vision/features.py`

- **Huang and
  Huang (2017)**
- **Lucas and Kanade (1981)** — least
  squares is solved, as written in Ziani (2025), equations 4 and 5
- **Ziani (2025)**
- **Ziani (2025)**

### `dense_armor/utility/signals/vision/landmarks.py`

- **Pearson et al. (2016)**

### `dense_armor/utility/signals/vision/motion.py`

- **Horn and Schunck (1981)**
- **Ziani (2025)** — Methods to a Multiresolution Horn-Schunck ([arXiv:2511.16535](https://arxiv.org/abs/2511.16535))
- **Horn, B. K., Schunck, B. G. (1981)** — Determining optical flow
- **Lucas, B. D., Kanade, T. (1981)** — An iterative image registration
        technique with an application to stereo vision

### `dense_armor/utility/signals/vision/patches.py`

- **Roth, K., Pemula, L., Zepeda, J., Scholkopf, B., Brox, T., Gehler, P. (2021)** — Towards total recall in industrial anomaly detection ([arXiv:2106.08265](https://arxiv.org/abs/2106.08265))
- **Huang, C., Huang, J. (2017)** — A fast HOG descriptor using lookup table
    and integral image ([arXiv:1703.06256](https://arxiv.org/abs/1703.06256))

### `dense_armor/utility/signals/vision/reduce.py`

- **Johnson
  and Lindenstrauss (1984)** — lemma guarantees that the pairwise distances
  are preserved up to a factor ``(1 + eps)`` with high probability,
  provided the output dimension ``k`` is large enough
- **Ghojogh et al. (2021)**
- **Balsubramani, Dasgupta,
  Freund (2013)**

### `dense_armor/utility/signals/vision/registration.py`

- **Ziani (2025)** — Methods to a Multiresolution Horn-Schunck ([arXiv:2511.16535](https://arxiv.org/abs/2511.16535))

### `dense_armor/utility/signals/vision/scene.py`

- **Reynolds, M. R. et al. (1975)** — The CUSUM chart with fast initial
        response
- **Reynolds, M. R. (1971)** — Approximations to the average
        run length of the CUSUM
- **Bifet, A., Gavalda, R. (2007)** — Learning from time-changing data with
        adaptive windowing
- **Raab, C. et al. (2020)** — Reactive soft prototype computing for concept
        drift
- **Roth, K. et al. (2022)** — Towards total recall in industrial anomaly
        detection

### `dense_armor/utility/signals/vision/tracking.py`

- **Ma, Y. et al. (2023)** — Adaptive confidence threshold for ByteTrack ([arXiv:2312.01650](https://arxiv.org/abs/2312.01650))
- **Mbalawata, I. S., Sarkka, S. (2013)** — Moment conditions for
        convergence of the variational Bayes approximation in Kalman
        filtering ([arXiv:1308.5875](https://arxiv.org/abs/1308.5875))

### `dense_armor/utility/signals/vision/vocabulary.py`

- **Huang, C., Huang, J. (2017)** — A fast HOG descriptor using lookup table
        and integral image ([arXiv:1703.06256](https://arxiv.org/abs/1703.06256))


## Per paper


### Abbadi, A. (2005)
- Titolo: Efficient computation
    of frequent and top-k elements in data streams
- Citato in: `utility/signals/stats/sketches.py`

### Amekoe et al. (2024)
- Titolo: References
----------
Ksieniewicz, P
- [arXiv:2001.11077](https://arxiv.org/abs/2001.11077)
- Citato in: `utility/evaluation/evaluate/prequential.py`

### Auer, Cesa-Bianchi and Fischer (2002)
- Titolo: The formulas follow the paper as cited
- Citato in: `utility/online/bandit/ucb.py`

### Ba, J. L., Kiros, J. R., Hinton, G. E. (2016)
- Titolo: Layer Normalization
- [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
- Citato in: `utility/deep/neural/norm.py`

### Balsubramani, A., Dasgupta, S., Freund, Y. (2013)
- Titolo: The fast convergence of
        incremental PCA, NIPS: the target of the rule is the top eigenvector of
        the covariance (not in the folder, bibliographic only)
- Citato in: `ml/vision.py`

### Balsubramani, Dasgupta,
  Freund (2013)
- Citato in: `utility/signals/vision/reduce.py`

### Bamber, D. (1975)
- Titolo: The area above the ordinal dominance graph and the
    area below the receiver operating characteristic graph
- Citato in: `utility/evaluation/metrics/roc_auc.py`

### Bhattacharjee et al. (2021)
- Titolo: compares the
total loss
- Citato in: `utility/online/cluster/kmeans.py`

### Bhattacharjee, R., Dasgupta, S., Imola, J. J., Moshkovitz, M. (2021)
- Titolo: Online k-means clustering on arbitrary data streams
- [arXiv:2102.09101](https://arxiv.org/abs/2102.09101)
- Citato in: `utility/online/cluster/kmeans.py`

### Bifet and Gavalda (2009)
- Titolo: adds a drift detector at each node:
when the detector warns, an alternate subtree grows in the background
on the recent samples; when the alternate is significantly more
accurate than the main one, it replaces the corresponding branch
- Citato in: `utility/online/tree/adaptive.py`

### Bifet, A., Gavalda, R. (2007)
- Titolo: Learning from time-changing data with
        adaptive windowing
- Citato in: `utility/signals/vision/scene.py`

### Bifet, A., Gavalda, R. (2009)
- Titolo: Adaptive learning from evolving
        data streams
- Citato in: `utility/online/tree/adaptive.py`

### Bishop, C. M. (2006)
- Titolo: Pattern Recognition and Machine Learning, Eq
- Citato in: `ml/linear.py`

### Bishop, C. M. (2006)
- Titolo: Pattern Recognition and Machine Learning
- Citato in: `utility/online/linear/bayes.py`

### Blondel et al. (2016)
- Citato in: `utility/online/facto/hofm.py`

### Bloom (1970)
- Titolo: Membership test with no false
  negatives and a false positive rate of at most ``p_false``
- Citato in: `utility/signals/stats/sketches.py`

### Bloom, B. H. (1970)
- Titolo: Space/time trade-offs in hash coding with
    allowable errors
- Citato in: `utility/signals/stats/sketches.py`

### Bodnar, T., Gupta, A. K., Parolya, N. (2014)
- Titolo: Optimal linear shrinkage
        estimator for large dimensional precision matrix, Eq
- Citato in: `ml/stats.py`

### Bollerslev, T. (1986)
- Titolo: Generalized autoregressive conditional
    heteroskedasticity
- [DOI:10.1016/0304-4076(86](https://doi.org/10.1016/0304-4076(86)
- Citato in: `utility/online/timeseries/garch.py`

### Breiman, Friedman, Olshen, Stone (1984)
- Titolo: *Classification and
Regression Trees
- Citato in: `utility/online/tree/batch.py`

### Breiman, L. (1996)
- Titolo: Bagging predictors, Machine Learning 24(2): bootstrap
        samples and the mean of the models (not in the folder, bibliographic
        only)
- Citato in: `ml/ensemble.py`

### Breiman, L. (2001)
- Titolo: Random forests, Machine Learning 45(1): a tree per
        bootstrap sample, a random subset of features at each split (not in
        the folder, bibliographic only)
- Citato in: `ml/forest.py`

### Breiman, L., Friedman, J., Olshen, R., Stone, C. (1984)
- Titolo: Classification and
        Regression Trees, Wadsworth, chapter 2: the gini impurity and the
        recursive binary splitting (not in the folder, bibliographic only)
- Citato in: `ml/tree.py`

### Breunig et al. (2000)
- Citato in: `utility/online/anomaly/lof.py`

### Brier, G. W. (1950)
- Titolo: Verification of forecasts expressed in terms of
    probability
- Citato in: `utility/evaluation/metrics/classification.py`

### Börzsönyi et al. (2001)
- Titolo: and Tao and Papadias (2006)
- Citato in: `utility/signals/stats/skyline.py`

### Candes (2021)
- Titolo: The coverage level is
  re-estimated at every sample from the recent miscoverage frequency,
  so the interval tracks a drifting stream
- Citato in: `roles/uncertainty.py`

### Cappé, O., Moulines, E. (2009)
- Titolo: On-line expectation-maximization
    algorithm for latent data models
- Citato in: `utility/signals/stats/mixture.py`

### Cardot, H., Godichon-Baggioni, A. (2017)
- Titolo: TEST 26(3), 461-480
- Citato in: `utility/online/anomaly/mahalanobis.py`

### Chalapathy and Chawla (2019)
- Citato in: `utility/deep/neural/online_autoencoder.py`

### Chan, Golub and LeVeque (1983)
- Titolo: for the choice
of K; the paper of 1983 was not read
- Citato in: `utility/signals/stats/stable.py`

### Chan, T. F., Golub, G. H., LeVeque, R. J. (1979)
- Titolo: Updating formulae and a pairwise
        algorithm for computing sample variances, Compstat: the two-pass formulas for the
        mean, the variance and the covariance (not in the folder, bibliographic only)
- Citato in: `ml/stats.py`

### Chan, T. F., Golub, G. H., LeVeque, R. J. (1979)
- Titolo: Updating formulae and
    a pairwise algorithm for computing sample variances
- Citato in: `utility/signals/stats/dependence.py`, `utility/signals/stats/moments.py`

### Chan, T. F., Golub, G. H., LeVeque, R. J. (1979)
- Titolo: In Compstat
- Citato in: `utility/evaluation/metrics/regression.py`

### Chapelle and Li (2011)
- Citato in: `utility/online/bandit/ts.py`

### Chatfield, C. (2003)
- Titolo: The Analysis of Time Series: An Introduction,
    6th ed
- Citato in: `utility/signals/stats/dependence.py`

### Chauvenet (1863)
- Citato in: `ml/anomaly.py`, `utility/online/anomaly/robust_filters.py`

### Chollet, F. (2017)
- Titolo: Xception: Deep learning with depthwise
        separable convolutions
- [arXiv:1610.02357](https://arxiv.org/abs/1610.02357)
- Citato in: `utility/deep/neural/conv_more.py`

### Cohen, J. (1960)
- Titolo: A coefficient of agreement for nominal scales
- Citato in: `utility/evaluation/metrics/classification.py`

### Cook, R. D., Weisberg, S. (1982)
- Titolo: Residuals and Influence in
    Regression
- Citato in: `utility/evaluation/metrics/regression.py`

### Cormode, G., Muthukrishnan, S. (2005)
- Titolo: An improved data stream summary:
    the Count-Min sketch and its applications
- Citato in: `utility/signals/stats/sketches.py`

### Cover, T., Hart, P. (1967)
- Titolo: Nearest neighbor pattern classification,
        IEEE Trans
- Citato in: `ml/neighbors.py`

### Dagdia, Z. (2024)
- Titolo: arXiv:2409
- [arXiv:2409.10111](https://arxiv.org/abs/2409.10111)
- Citato in: `utility/evaluation/evaluate/prequential.py`

### Davies and Gather (1993)
- Citato in: `ml/anomaly.py`

### Dembczynski et al. (2010)
- Citato in: `utility/online/multioutput/multioutput.py`

### Dembczynski, K., Cheng, W., Hullermeier, E. (2010)
- Titolo: Bayes optimal
        multilabel classification via probabilistic classifier chains, Eq
- Citato in: `ml/multioutput.py`

### Dietterich and Bakiri (1995)
- Citato in: `utility/online/multiclass/multiclass.py`

### Dietterich, T. G., Bakiri, G. (1995)
- Titolo: Solving multiclass learning problems
        via error-correcting output codes, JAIR 2: a codeword per class and the
        closest codeword in Hamming distance (not in the folder, bibliographic
        only)
- Citato in: `ml/multiclass.py`

### Domingos, P., Hulten, G. (2000)
- Titolo: Mining high-speed data streams
- Citato in: `utility/online/tree/adaptive.py`, `utility/online/tree/efdt.py`, `utility/online/tree/hoeffding.py`

### Duarte, J., Gama, J., Bifet, A. (2016)
- Titolo: Adaptive model rules from
        high-speed data streams, ACM TKDD 10(3): the rule set grows and is
        revised as examples arrive (not in the folder, bibliographic only)
- Citato in: `ml/rules.py`, `utility/online/rules/amrules.py`

### Dugas et al. (2001)
- Titolo: is not in this folder and is cited as
    ``bibliographic only``
- Citato in: `utility/deep/neural/activations_more.py`

### Dumoulin and Visin (2016)
- Citato in: `utility/deep/neural/convtranspose.py`

### Dunning, T., Ertl, O. (2019)
- Titolo: Computing extremely accurate quantiles
    using t-digests
- [arXiv:1902.04023](https://arxiv.org/abs/1902.04023)
- Citato in: `utility/signals/stats/quantiles.py`

### Elfwing, S., Uchibe, E., Doya, K. (2017)
- Titolo: Sigmoid-weighted linear
        units for neural network function approximation in
        reinforcement learning
- [arXiv:1702.03118](https://arxiv.org/abs/1702.03118)
- Citato in: `utility/deep/neural/activations_more.py`

### Esteban, A., Cano, A., Zafra, A., Ventura, S. (2024)
- Titolo: Hoeffding
        adaptive trees for multi-label classification on data streams
- Citato in: `utility/online/tree/adaptive.py`

### Fedus et al. (2021)
- Titolo: uses top-1, where the weight is 1 by construction
- Citato in: `utility/deep/neural/moe.py`

### Fedus et al. (2021)
- Citato in: `utility/deep/neural/moe.py`

### Fitzpatrick, R. (2008)
- Titolo: Maxwell's equations and the principles of
    electromagnetism
- Citato in: `roles/physics.py`

### Flajolet et al. (2007)
- Titolo: Distinct count with about
  ``1
- Citato in: `utility/signals/stats/sketches.py`

### Flajolet, P., Fusy, E., Gandouet, O., Meunier, F. (2007)
- Titolo: HyperLogLog:
    the analysis of a near-optimal cardinality estimation algorithm
- Citato in: `utility/signals/stats/sketches.py`

### Foster, J., McCrory, S., DeBuys, C., Bertrand, S., Griffin, R. (2024)
- Titolo: Physically Consistent Online Inertial Adaptation for
    Humanoid Loco-manipulation
- [arXiv:2405.07901](https://arxiv.org/abs/2405.07901)
- Citato in: `utility/online/linear/payload.py`

### Freund and Schapire (1997)
- Titolo: and the exponentiated
gradient of Kivinen and Warmuth (1997) give each member a weight
``w_i = exp(-eta * L_i)`` that decays with its cumulative loss ``L_i``;
the prediction is the weighted average

    y_hat = sum_i w_i y_hat_i / sum_i w_i
- Citato in: `utility/online/ensemble/ewa.py`

### Freund, Y., Schapire, R. E. (1997)
- Titolo: A decision-theoretic generalization of
        on-line learning and an application to boosting, JCSS 55(1): the
        Hedge weights ``exp(-eta L)`` and AdaBoost (not in the folder,
        bibliographic only)
- Citato in: `ml/ensemble.py`

### Freund, Y., Schapire, R. E. (1997)
- Titolo: A decision-theoretic
        generalization of on-line learning and an application to
        boosting
- Citato in: `utility/online/ensemble/boosting.py`, `utility/online/ensemble/ewa.py`

### Fuerkranz (2002)
- Citato in: `utility/online/multiclass/multiclass.py`

### Furnkranz, J. (2002)
- Titolo: Round robin classification, JMLR 2: one binary problem
        per pair of classes, decided by votes (not in the folder,
        bibliographic only)
- Citato in: `ml/multiclass.py`

### Gerencsér, L., Orlovits, Z. (2012)
- Titolo: Recursive estimation of GARCH
    processes
- Citato in: `utility/online/timeseries/garch.py`

### Geurts, P., Ernst, D., Wehenkel, L. (2006)
- Titolo: Extremely randomized trees,
        Machine Learning 63(1): the whole sample and a random threshold at
        each split (not in the folder, bibliographic only)
- Citato in: `ml/forest.py`

### Ghojogh et al. (2021)
- Citato in: `ml/vision.py`, `utility/signals/vision/reduce.py`

### Gibbs, I., Candes, E. J. (2021)
- Titolo: Adaptive conformal inference under
    distribution shift
- [arXiv:2305.00070](https://arxiv.org/abs/2305.00070)
- Citato in: `roles/uncertainty.py`

### Gomes et al. (2017)
- Titolo: A single Hoeffding tree follows a change through
one detector; the forest follows it through one detector per tree, and
the spread of the votes says how sure it is
- Citato in: `utility/online/forest/arf.py`

### Gomes, H. M., Bifet, A., Read, J., Barddal, J. P., Enembreck, F.,
        Pfahringer, B., Holmes, G., Abdessalem, T. (2017)
- Titolo: Adaptive
        random forests for evolving data stream classification
- Citato in: `utility/online/forest/arf.py`

### Good, I. J. (1952)
- Titolo: Rational decisions
- Citato in: `utility/evaluation/metrics/regression.py`

### Gouk, H., Pfahringer, B., Frank, E. (2019)
- Titolo: Stochastic gradient
        trees
- Citato in: `utility/online/tree/sgt.py`

### Goyal et al. (2017)
- Citato in: `utility/deep/neural/data_parallel.py`

### Goyal, P., Girshick, R., He, K., Dollar, P. (2017)
- Titolo: Focal Loss for Dense Object Detection
- [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- Citato in: `utility/deep/neural/losses_more.py`

### Guillot, A., Godichon-Baggioni, A., Robin, S. (2025)
- Titolo: Online robust
    covariance and outlier detection
- [arXiv:2601.03957](https://arxiv.org/abs/2601.03957)
- Citato in: `utility/online/anomaly/mahalanobis.py`

### Gupta, C., Ramdas, A. (2023)
- Titolo: Online Platt scaling with calibeating
- [arXiv:2305.00070](https://arxiv.org/abs/2305.00070)
- Citato in: `roles/uncertainty.py`

### Hampel (1974)
- Citato in: `ml/anomaly.py`

### Hampel, F. R. (1974)
- Titolo: The influence curve and its role in robust
    estimation
- Citato in: `utility/signals/stats/robust.py`

### Hawkins, D. M., Olwell, D. H. (1998)
- Titolo: Cumulative Sum Charts and Charting
        for Quality Improvement, Springer: the slack ``k`` as half the shift to
        detect (not in the folder, bibliographic only)
- Citato in: `ml/drift.py`

### Hazan, E.,
        Agarwal, A., Kale, S. (2007)
- Titolo: Logarithmic regret algorithms for online
        convex optimization, Algorithm 2: update rules applied to the gradient
        of one example at a time (not in the folder, bibliographic only)
- Citato in: `ml/optim.py`

### He, K., Zhang, X., Ren, S., Sun, J. (2015)
- Titolo: Delving Deep into
        Rectifiers: Surpassing Human-Level Performance on ImageNet
        Classification
- [arXiv:1502.01852](https://arxiv.org/abs/1502.01852)
- Citato in: `utility/deep/neural/init.py`

### He, K., Zhang, X., Ren, S., Sun, J. (2015)
- Titolo: Delving deep into
        rectifiers
- [arXiv:1502.01852](https://arxiv.org/abs/1502.01852)
- Citato in: `utility/deep/neural/activations.py`, `utility/deep/neural/activations_more.py`, `utility/deep/neural/models.py`

### He, K., Zhang, X., Ren, S., Sun, J. (2016)
- Titolo: Deep Residual Learning
        for Image Recognition
- [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Citato in: `utility/deep/neural/chain.py`, `utility/deep/neural/layers.py`, `utility/deep/neural/models.py`

### Healy (2022)
- Citato in: `utility/signals/stats/stable.py`

### Healy, J. (2022)
- Titolo: Accurate and consistent calculation of the mean and
        variance in Monte-Carlo simulations
- [arXiv:2206.10662](https://arxiv.org/abs/2206.10662)
- Citato in: `utility/signals/stats/stable.py`

### Hendrycks, D., Gimpel, K. (2016)
- Titolo: Gaussian Error Linear Units
        (GELUs)
- [arXiv:1606.08415](https://arxiv.org/abs/1606.08415)
- Citato in: `utility/deep/neural/activations_more.py`

### Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever, I.,
        Salakhutdinov, R. R. (2012)
- Titolo: Improving neural networks by
        preventing co-adaptation of feature detectors
- [arXiv:1207.0580](https://arxiv.org/abs/1207.0580)
- Citato in: `utility/deep/neural/layers.py`

### Horn and Schunck (1981)
- Citato in: `utility/signals/vision/motion.py`

### Horn and Schunck (1981)
- Titolo: written by Ziani (2025, equation 6 and the Euler-Lagrange pair 7-8)
- Citato in: `utility/signals/vision/dense.py`

### Horn, B. K., Schunck, B. G. (1981)
- Titolo: Determining optical flow
- [arXiv:2511.16535](https://arxiv.org/abs/2511.16535)
- Citato in: `ml/vision.py`, `utility/signals/vision/dense.py`, `utility/signals/vision/motion.py`

### Huang and
  Huang (2017)
- Citato in: `utility/signals/vision/features.py`

### Huang, C., Huang, J. (2017)
- Titolo: A fast HOG descriptor using lookup table
        and integral image
- [arXiv:1703.06256](https://arxiv.org/abs/1703.06256)
- Citato in: `utility/signals/vision/vocabulary.py`

### Huang, C., Huang, J. (2017)
- Titolo: A fast HOG descriptor using lookup table
    and integral image
- [arXiv:1703.06256](https://arxiv.org/abs/1703.06256)
- Citato in: `utility/signals/vision/patches.py`

### Hyndman, R. J., Fan, Y. (1996)
- Titolo: Sample quantiles in statistical
    packages
- Citato in: `utility/signals/stats/robust.py`

### Ikonomovska (2012)
- Titolo: The algorithm is FIMT-DD of Ikonomovska (2012), Chapter 6 of the PhD
thesis, pages 68-77 of the printed pages
- Citato in: `utility/online/tree/fimt.py`

### Ikonomovska (2012)
- Citato in: `utility/online/tree/fimt.py`

### Ioffe, S., Szegedy, C. (2015)
- Titolo: Batch Normalization: Accelerating
        Deep Network Training by Reducing Internal Covariate Shift
- [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
- Citato in: `utility/deep/neural/norm.py`

### Jacobsen and Lyons (2003)
- Citato in: `utility/signals/stats/sdft.py`

### Johnson
  and Lindenstrauss (1984)
- Titolo: lemma guarantees that the pairwise distances
  are preserved up to a factor ``(1 + eps)`` with high probability,
  provided the output dimension ``k`` is large enough
- Citato in: `utility/signals/vision/reduce.py`

### Johnson, W. B., Lindenstrauss, J. (1984)
- Titolo: and Ghojogh et al
- Citato in: `ml/vision.py`

### Juan et al. (2016)
- Citato in: `utility/online/facto/ffm.py`

### Kayan, H., Rana, O., Burnap, P., Perera, C. (2023)
- [arXiv:2303.01300](https://arxiv.org/abs/2303.01300)
- Citato in: `utility/signals/datasets/casper.py`

### Kingma, D. P., Ba, J. (2014)
- Titolo: Adam: a method for stochastic
    optimization
- [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- Citato in: `utility/online/linear/glm.py`, `utility/online/optim/__init__.py`

### Kingma, D. P., Ba, J. (2015)
- Titolo: Adam, Algorithm 1; Hazan, E
- Citato in: `ml/optim.py`

### Kivinen and Warmuth (1997)
- Titolo: give each member a weight
``w_i = exp(-eta * L_i)`` that decays with its cumulative loss ``L_i``;
the prediction is the weighted average

    y_hat = sum_i w_i y_hat_i / sum_i w_i
- Citato in: `utility/online/ensemble/ewa.py`

### Kivinen, J., Warmuth, M. K. (1997)
- Titolo: Exponentiated gradient versus
        gradient descent for linear predictors
- Citato in: `utility/online/ensemble/ewa.py`

### Koren (2009)
- Citato in: `utility/online/reco/reco.py`

### Kroger, T. (2021)
- [arXiv:2105.04830](https://arxiv.org/abs/2105.04830)
- Citato in: `utility/robot/control/rate_limiter.py`

### Ksieniewicz, P., Zyblewski, P. (2020)
- Citato in: `utility/signals/datasets/drift_stream.py`

### Ksieniewicz, P., Zyblewski, P. (2020)
- Titolo: Stream-learn
- [arXiv:2001.11077](https://arxiv.org/abs/2001.11077)
- Citato in: `utility/evaluation/evaluate/prequential.py`

### Ksieniewicz, P., Zyblewski, P. (2020)
- Titolo: Stream-learn: open-source Python
    library for difficult data stream batch analysis
- [arXiv:2001.11077](https://arxiv.org/abs/2001.11077)
- Citato in: `utility/evaluation/evaluate/model_selection.py`

### Lakshminarayanan, B., Roy, D. M., Teh, Y. W. (2014)
- Titolo: Mondrian forests:
        efficient online random forests, NIPS, section 2, Algorithms 1 and 2
        (``SampleMondrianTree`` and ``SampleMondrianBlock``, page 4)
- Citato in: `ml/tree.py`

### Lakshminarayanan, B., Roy, D. M., Teh, Y. W. (2014)
- Titolo: Mondrian
        forests: efficient online random forests
- Citato in: `utility/online/forest/amf.py`, `utility/online/tree/mondrian.py`

### Lakshminarayanan, B., Roy, D. M., Teh, Y. W. (2016)
- Titolo: Mondrian forests for
        large-scale regression when uncertainty matters, AISTATS, Algorithm 5 and
        eq
- Citato in: `ml/tree.py`

### Lakshminarayanan, B., Roy, D. M., Teh, Y. W. (2016)
- Titolo: Mondrian
        forests for large-scale regression when uncertainty matters
- Citato in: `utility/online/tree/mondrian.py`

### Lavin, A., Ahmad, S. (2015)
- Titolo: Evaluating real-time anomaly detection
    algorithms --- the Numenta Anomaly Benchmark
- [arXiv:2109.05257](https://arxiv.org/abs/2109.05257)
- Citato in: `utility/evaluation/evaluate/events.py`, `utility/evaluation/metrics/events.py`

### Lee, B., Yoon, S. (2021)
- Titolo: Towards a
    rigorous evaluation of time-series anomaly detection
- [arXiv:2109.05257](https://arxiv.org/abs/2109.05257)
- Citato in: `utility/evaluation/metrics/events.py`

### Leveni et al. (2024)
- Citato in: `utility/online/anomaly/oiforest.py`

### Leveni, F., Cassales, G. W., Pfahringer, B., Bifet, A., Boracchi, G. (2024)
- Titolo: Online Isolation Forest
- [arXiv:2505.09593](https://arxiv.org/abs/2505.09593)
- Citato in: `utility/online/anomaly/oiforest.py`

### Li et al. (2010)
- Citato in: `utility/online/bandit/linucb.py`

### Li et al. (2011)
- Citato in: `utility/online/bandit/evaluate.py`

### Liu, C., Rubin, D. B. (1995)
- Titolo: ML estimation of the t distribution with
    unknown degrees of freedom
- Citato in: `utility/signals/stats/student_t.py`

### Ljung, L., Soderstrom, T. (1983)
- Titolo: Theory and Practice of
    Recursive Identification
- Citato in: `utility/online/linear/payload.py`

### Ljung, L., Soderstrom, T. (1983)
- Titolo: Theory and Practice of Recursive
Identification
- Citato in: `utility/online/learn/online_dynamics.py`

### Lou et al. (2024)
- Citato in: `utility/online/anomaly/loda.py`

### Lou, B., Boland, D., Leong, P. H. W. (2024)
- Titolo: fSEAD: a Composable
    FPGA-based Streaming Ensemble Anomaly Detection Library
- [arXiv:2406.05999](https://arxiv.org/abs/2406.05999)
- Citato in: `utility/online/anomaly/loda.py`

### Lucas and Kanade (1981)
- Titolo: least
  squares is solved, as written in Ziani (2025), equations 4 and 5
- Citato in: `utility/signals/vision/features.py`

### Lucas, B. D., Kanade, T. (1981)
- Titolo: An iterative image registration
        technique with an application to stereo vision
- Citato in: `utility/signals/vision/motion.py`

### Ma, Y. et al. (2023)
- Titolo: Adaptive confidence threshold for ByteTrack
- [arXiv:2312.01650](https://arxiv.org/abs/2312.01650)
- Citato in: `utility/signals/vision/tracking.py`

### MacKay, D. J. C. (2007)
- Titolo: *Bayesian Online
Changepoint Detection
- [arXiv:0710.3742](https://arxiv.org/abs/0710.3742)
- Citato in: `utility/online/drift/bocpd.py`

### Malialis, K. et al. (2022)
- Titolo: Queue-based resampling for online class
        imbalance learning: a sequential method over a stream, so
        ``QueueResampler`` is declared in ``STOPPED`` (not in the folder,
        bibliographic only)
- Citato in: `ml/preprocessing.py`

### Malialis, K., Panayiotou, C., Polycarpou, M. M. (2018)
- Titolo: Queue-based
    resampling for online class imbalance learning
- Citato in: `utility/signals/preprocessing/imbalance.py`

### Malialis, Panayiotou, Polycarpou (2018)
- Citato in: `utility/signals/preprocessing/imbalance.py`

### Manapragada, C., Webb, G. I., Salehi, M. (2018)
- Titolo: Extremely fast
        decision tree
- Citato in: `utility/online/tree/efdt.py`, `utility/online/tree/hoeffding.py`

### Mann, H. B., Whitney, D. R. (1947)
- Titolo: On a test of whether one of two
    random variables is stochastically larger than the other
- Citato in: `utility/evaluation/metrics/roc_auc.py`

### Masson, C., Rim, J. E., Lee, H. K. (2019)
- Titolo: DDSketch: a fast and
    fully-mergeable quantile sketch with relative-error guarantees
- [arXiv:1902.04023](https://arxiv.org/abs/1902.04023)
- Citato in: `utility/signals/stats/quantiles.py`

### Mastelini, S. M. (2023)
- Titolo: Efficient online tree, rule-based and distance-based
        algorithms
- Citato in: `utility/online/forest/oxt.py`

### Mbalawata, I. S., Sarkka, S. (2013)
- Titolo: Moment conditions for
        convergence of the variational Bayes approximation in Kalman
        filtering
- [arXiv:1308.5875](https://arxiv.org/abs/1308.5875)
- Citato in: `utility/signals/vision/tracking.py`

### Mbalawata, I. S., Sarkka, S., Vihola, M., Haario, H. (2013)
- Titolo: Adaptive Metropolis algorithm using variational Bayesian adaptive
    Kalman filter
- [arXiv:1308.5875](https://arxiv.org/abs/1308.5875)
- Citato in: `utility/online/timeseries/kalman.py`

### McCallum and Nigam (1998)
- Titolo: A sample is a binary vector over the vocabulary; the class-
conditional probability of a feature is smoothed by Laplace, eq
- Citato in: `utility/online/bayes/bernoulli.py`

### McCallum and Nigam (1998)
- Citato in: `utility/online/bayes/multinomial.py`

### McCallum, A., Nigam, K. (1998)
- Titolo: A comparison of event models for naive
        Bayes text classification, page 4 (multinomial and Bernoulli models)
        (not in the folder, bibliographic only)
- Citato in: `ml/bayes.py`

### McClamroch (2010)
- Citato in: `dynamics/six_dof_pbc_cbf_controller.py`

### Metwally et al. (2005)
- Titolo: Top-k heavy hitters with a
  deterministic error bound: the reported count of the true k-th most
  frequent item is at least its true count minus ``n / k``
- Citato in: `utility/signals/stats/sketches.py`

### Moulines (2009)
- Titolo: The
generic update, Eq
- Citato in: `utility/signals/stats/mixture.py`

### Murphy, K. P. (2007)
- Titolo: Conjugate Bayesian analysis of the Gaussian
        distribution (not in the folder, bibliographic only)
- Citato in: `ml/proba.py`

### Muthukrishnan (2005)
- Titolo: Frequency
  estimation with ``eps · N`` one-sided error and probability
  ``1 - delta``, using ``w = ceil(e / eps)`` counters per row and
  ``d = ceil(ln(1 / delta))`` rows
- Citato in: `utility/signals/stats/sketches.py`

### Neal, R. M., Hinton, G. E. (1998)
- Titolo: A view of the EM algorithm that
    justifies incremental, sparse, and other variants
- Citato in: `utility/signals/stats/mixture.py`

### Ng, A., Zhu, S., Zhang, A. G., Reid, N. (2026)
- Titolo: Robust Regression
    with Student's T: The Role of Degrees of Freedom
- [arXiv:2603.00269](https://arxiv.org/abs/2603.00269)
- [DOI:10.1080/01621459](https://doi.org/10.1080/01621459)
- Citato in: `utility/signals/stats/student_t.py`

### Ng, R. T., Sander, J. (2000)
- Titolo: LOF:
        Identifying Density-Based Local Outliers
- Citato in: `ml/anomaly.py`

### Ng, R. T., Sander, J. (2000)
- Titolo: LOF: Identifying Density-Based Local Outliers
- Citato in: `utility/online/anomaly/lof.py`

### Ogita, T., Rump, S. M., Oishi, S. (2005)
- Titolo: Accurate Sum and Dot
        Product
- Citato in: `utility/signals/stats/stable.py`

### Oplatkova, Z., Vala, R. (2025)
- Titolo: Isolation Forest in Novelty Detection Scenario
- [arXiv:2505.08489](https://arxiv.org/abs/2505.08489)
- Citato in: `utility/online/anomaly/hst.py`

### Orabona, F. (2019)
- Titolo: A modern introduction to online learning, Algorithms 2
- Citato in: `ml/optim.py`

### Orabona, F. (2019)
- Titolo: A Modern Introduction to Online Learning
- [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- Citato in: `utility/online/linear/glm.py`, `utility/online/optim/__init__.py`

### Orlovits (2012)
- Titolo: for the
recursive QML extension, but does not implement it
- Citato in: `utility/online/timeseries/garch.py`

### Oza and Russell (2001)
- Citato in: `utility/online/ensemble/boosting.py`

### Oza, N. C., Russell, S. (2001)
- Titolo: Online bagging and boosting
- Citato in: `utility/online/ensemble/bagging.py`, `utility/online/ensemble/boosting.py`

### Page, E. S. (1954)
- Titolo: Continuous inspection schemes, Biometrika 41(1-2):
        100-115: the CUSUM with a fixed reference decided once from an
        in-control baseline (not in the folder, bibliographic only)
- Citato in: `ml/drift.py`

### Page, E. S. (1954)
- Titolo: Continuous inspection schemes
- Citato in: `roles/safety.py`, `utility/online/drift/page_hinkley.py`, `utility/online/learn/online_dynamics.py`

### Page, E.S. (1954)
- Citato in: `utility/online/drift/cusum.py`

### Pan et al. (2018)
- Citato in: `utility/online/facto/fwfm.py`

### Parolya, N. (2014)
- Titolo: *On the
Strong Convergence of the Optimal Linear Shrinkage Estimator for Large
Dimensional Covariance Matrix
- [arXiv:1308.2608](https://arxiv.org/abs/1308.2608)
- Citato in: `utility/signals/stats/shrinkage.py`

### Paterek (2007)
- Titolo: and Koren (2009)
- Citato in: `utility/online/reco/reco.py`

### Pearson et al. (2016)
- Citato in: `utility/signals/vision/landmarks.py`

### Pevny (2016)
- Titolo: Details not fixed by Lou et al
- Citato in: `utility/online/anomaly/loda.py`

### Pevny, T. (2016)
- Titolo: LODA: Lightweight on-line detector of anomalies
- Citato in: `utility/online/anomaly/loda.py`

### Popescu, M., Mronga, D., Kumar, S., Kirchner, F. (2026)
- Titolo: Residual-based adaptive Kalman filtering for legged robot state
    estimation
- [arXiv:2608.02316](https://arxiv.org/abs/2608.02316)
- Citato in: `utility/online/timeseries/kalman.py`

### Pébay, P. (2008)
- Titolo: Formulas for robust, one-pass parallel computation of
    covariances and arbitrary-order statistical moments
- Citato in: `utility/signals/stats/moments.py`

### Raab, C. et al. (2020)
- Titolo: Reactive soft prototype computing for concept
        drift
- Citato in: `utility/signals/vision/scene.py`

### Rahimi, A., Recht, B. (2007)
- Titolo: Random features for large-scale kernel
    machines
- Citato in: `utility/online/anomaly/ocsvm.py`

### Ramdas (2023)
- Citato in: `roles/uncertainty.py`

### Ramos, L. M., Romero, D., Beferull-Lozano, B. (2019)
- Titolo: Online topology identification from vector autoregressive
    time series
- [arXiv:1904.01864](https://arxiv.org/abs/1904.01864)
- Citato in: `utility/online/timeseries/ar.py`

### Read, J. et al. (2009)
- Titolo: Classifier chains for multi-label classification
        (not in the folder, bibliographic only)
- Citato in: `ml/multioutput.py`

### Rendle (2010)
- Citato in: `utility/online/facto/fm.py`

### Rennie et al. (2003)
- Citato in: `utility/signals/preprocessing/vectorizer.py`

### Rennie, J. D. M. et al. (2003)
- Titolo: Tackling the poor assumptions of naive
        Bayes text classifiers, pages 6-7 (complement naive Bayes) (not in
        the folder, bibliographic only)
- Citato in: `ml/bayes.py`

### Rennie, Shih, Teevan and Karger (2003)
- Citato in: `utility/online/bayes/complement.py`

### Reynolds, M. R. (1971)
- Titolo: Approximations to the average
        run length of the CUSUM
- Citato in: `utility/signals/vision/scene.py`

### Reynolds, M. R. et al. (1975)
- Titolo: The CUSUM chart with fast initial
        response
- Citato in: `utility/signals/vision/scene.py`

### Rifkin, R., Klautau, A. (2004)
- Titolo: In defense of one-vs-all classification:
        one binary problem per class (not in the folder, bibliographic only)
- Citato in: `ml/multiclass.py`

### Roth, K. et al. (2021)
- Titolo: Towards total recall in industrial anomaly
        detection
- [arXiv:2106.08265](https://arxiv.org/abs/2106.08265)
- Citato in: `utility/online/cluster/kmeans.py`

### Roth, K. et al. (2022)
- Titolo: Towards total recall in industrial anomaly
        detection
- Citato in: `utility/signals/vision/scene.py`

### Roth, K., Pemula, L., Zepeda, J., Scholkopf, B., Brox, T., Gehler, P. (2021)
- Titolo: Towards total recall in industrial anomaly detection
- [arXiv:2106.08265](https://arxiv.org/abs/2106.08265)
- Citato in: `utility/signals/vision/patches.py`

### Roy, D. M., Teh, Y. W. (2009)
- Titolo: The Mondrian process
- Citato in: `utility/online/tree/mondrian.py`

### Rubin (1995)
- Titolo: is not implemented because the
paper is not in the folder and the reference paper in the folder
(Ng et al
- [arXiv:2603.00269](https://arxiv.org/abs/2603.00269)
- Citato in: `utility/signals/stats/student_t.py`

### Rucker, C., Wensing, P. M. (2022)
- Titolo: Smooth parameterization of
    rigid-body inertia
- Citato in: `utility/online/linear/payload.py`

### Sasaki, Y. (2007)
- Titolo: The truth of the F-measure
- Citato in: `utility/evaluation/metrics/classification.py`

### Shazeer
et al. (2017)
- Citato in: `utility/deep/neural/moe.py`

### Shazeer et al. (2017)
- Titolo: use a noisy top-k
gating (Eq
- Citato in: `utility/deep/neural/moe.py`

### Sherman, J., Morrison, W. J. (1950)
- Titolo: Adjustment of an inverse
    matrix corresponding to a change in one element of a given matrix
- Citato in: `utility/online/linear/bayes.py`

### Shewchuk, J. R. (1997)
- Titolo: Adaptive Precision Floating-Point Arithmetic
        and Fast Robust Geometric Predicates
- Citato in: `utility/signals/stats/stable.py`

### Silajev (2026)
- Citato in: `ml/preprocessing.py`

### Silajev, I. (2026)
- Titolo: *TF-IDF and BM25 as Kullback-Leibler
divergences
- [arXiv:2609.14016](https://arxiv.org/abs/2609.14016)
- Citato in: `utility/signals/preprocessing/vectorizer.py`

### Spyromitros-Xioufis et al. (2016)
- Citato in: `utility/online/multioutput/multioutput.py`

### Spyromitros-Xioufis, E. et al. (2016)
- Titolo: Multi-target regression via input
        space expansion, Section 2 (not in the folder, bibliographic only)
- Citato in: `ml/multioutput.py`

### Suk and Kpotufe (2025)
- Titolo: rewrites the OCSVM objective in an
empirical risk form, replaces the kernel with Random Fourier Features
(RFF) so gradient updates depend only on the current sample, adds a
strongly convex regularisation, and solves the resulting objective by
stochas
- Citato in: `utility/online/anomaly/ocsvm.py`

### Suk, J., Kpotufe, S. (2025)
- Titolo: An efficient variant of one-class SVM
    with lifelong online learning guarantees
- [arXiv:2512.11052](https://arxiv.org/abs/2512.11052)
- Citato in: `utility/online/anomaly/ocsvm.py`

### Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., Wojna, Z. (2016)
- Titolo: Rethinking the Inception Architecture for Computer
        Vision
- [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)
- Citato in: `utility/deep/neural/losses_more.py`

### Tabuada, P. (2019)
- [arXiv:1903.11199](https://arxiv.org/abs/1903.11199)
- Citato in: `utility/robot/control/cbf_filter.py`

### Tan, S. C., Ting, K. M., Liu, T. F. (2011)
- Titolo: Fast anomaly detection for
    streaming data
- [arXiv:2505.08489](https://arxiv.org/abs/2505.08489)
- Citato in: `utility/online/anomaly/hst.py`

### Tao and Papadias (2006)
- Titolo: Two definitions are used:

- dominance (Börzsönyi et al
- Citato in: `utility/signals/stats/skyline.py`

### Tatbul, N., Lee, T. J., Zdonik, S., Alam, M., Gottschlich, J. (2018)
- Titolo: Precision and recall for time series
- [arXiv:2109.05257](https://arxiv.org/abs/2109.05257)
- Citato in: `utility/evaluation/evaluate/events.py`, `utility/evaluation/metrics/events.py`

### Tran, D., Bourdev, L., Fergus, R., Torresani, L., Paluri, M. (2015)
- Titolo: Learning spatiotemporal features with 3D convolutional
        networks
- [arXiv:1412.0767](https://arxiv.org/abs/1412.0767)
- Citato in: `utility/deep/neural/conv_more.py`

### Tukey (1977)
- Citato in: `ml/anomaly.py`

### Tukey, J. W. (1977)
- Titolo: Exploratory Data Analysis
- Citato in: `utility/signals/stats/robust.py`

### Ulyanov, D., Vedaldi, A., Lempitsky, V. (2016)
- Titolo: Instance
        Normalization: The Missing Ingredient for Fast Stylization
- [arXiv:1607.08022](https://arxiv.org/abs/1607.08022)
- Citato in: `utility/deep/neural/norm_more.py`

### Unterthiner, T., Hochreiter, S. (2016)
- Titolo: Fast and
        accurate deep network learning by exponential linear units
        (ELUs)
- [arXiv:1511.07289](https://arxiv.org/abs/1511.07289)
- Citato in: `utility/deep/neural/activations_more.py`

### Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
        Gomez, A. N., Kaiser, L., Polosukhin, I. (2017)
- Titolo: Attention Is
        All You Need
- [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Citato in: `utility/deep/neural/positions.py`

### Vovk, V., Gammerman, A., Shafer, G. (2005)
- Titolo: Algorithmic Learning in a
    Random World
- Citato in: `utility/evaluation/metrics/regression.py`

### Weinberger, K. et al. (2009)
- Titolo: Feature hashing for large scale multitask
        learning, Theorem 3: the signed hashing trick (not in the folder,
        bibliographic only)
- Citato in: `ml/preprocessing.py`

### Welford (1962)
- Citato in: `utility/online/proba/gaussian.py`

### Welford, B. P. (1962)
- Titolo: Note on a method for calculating corrected sums
    of squares and products
- Citato in: `utility/signals/stats/dependence.py`, `utility/signals/stats/moments.py`

### Welford, B. P. (1962)
- Titolo: Technometrics 4(3), 419-420
- Citato in: `utility/evaluation/metrics/regression.py`

### West, D. H. D. (1979)
- Titolo: Updating mean and variance estimates, CACM 22(9):
        the exponentially weighted recursion (not in the folder,
        bibliographic only)
- Citato in: `ml/stats.py`

### West, D. H. D. (1979)
- Titolo: Updating mean and variance estimates: an
    improved method
- Citato in: `roles/realtime.py`

### West, D. H. D. (1979)
- Titolo: Updating mean and variance estimates: an improved
    method
- Citato in: `utility/signals/stats/moments.py`

### Wolpert, D. H. (1992)
- Titolo: Stacked generalization, Neural Networks 5(2): a
        meta-model on out-of-fold predictions (not in the folder,
        bibliographic only)
- Citato in: `ml/ensemble.py`

### Wu, Y., He, K. (2018)
- Titolo: Group Normalization
- [arXiv:1803.08494](https://arxiv.org/abs/1803.08494)
- Citato in: `utility/deep/neural/norm_more.py`

### Xie, Zou, Xie, Veeravalli (2021)
- Citato in: `utility/online/drift/page_hinkley.py`

### Xu, H. et al. (2018)
- Titolo: Unsupervised anomaly detection via variational
    auto-encoder for seasonal KPIs in web applications
- Citato in: `utility/evaluation/metrics/events.py`

### Zaman et al. (2019)
- Citato in: `utility/online/timeseries/ar.py`

### Zellner, A. (1976)
- Titolo: Bayesian and non-Bayesian analysis of the
    Student-t model
- [DOI:10.1080/01621459](https://doi.org/10.1080/01621459)
- Citato in: `utility/signals/stats/student_t.py`

### Zhu, J. et al. (2009)
- Titolo: Multi-class AdaBoost
        (SAMME) for the multiclass weights (not in the folder, bibliographic
        only)
- Citato in: `ml/ensemble.py`

### Ziani (2025)
- Titolo: Methods to a Multiresolution Horn-Schunck
- [arXiv:2511.16535](https://arxiv.org/abs/2511.16535)
- Citato in: `ml/vision.py`, `utility/signals/vision/dense.py`, `utility/signals/vision/motion.py`, `utility/signals/vision/registration.py`

### Ziani (2025)
- Citato in: `utility/signals/vision/features.py`

### Zliobaite et al. (2011)
- Citato in: `utility/online/active/strategies.py`

### Zubaroglu, A., Atalay, V. (2020)
- Titolo: Data stream clustering: a
        review
- [arXiv:2007.10781](https://arxiv.org/abs/2007.10781)
- Citato in: `utility/online/cluster/denstream.py`