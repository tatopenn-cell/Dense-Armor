# Online learning

A robot arm picks objects. While it works, four things need to keep learning, because
the world will not stay the same:

- **"How sure am I that this grasp will hold?"** — the classifier that predicts
  "good grasp / bad grasp" was trained once, on a fixed set. If it says 0.8, that
  should mean 8 times out of 10. A model whose probabilities are not honest is
  dangerous on a robot: the control policy trusts them.
  → [Calibration](calibration.md)

- **"Which past situations look like this one?"** — the nearest-neighbour step
  underneath every retrieval, imitation-learning and anomaly-detection module needs a
  way to say "these two situations are similar". Euclidean distance is a default, not
  a decision: on real robot features it is often wrong.
  → [Metric learning](metric_learning.md)

- **"Am I moving freely, touching something, or colliding?"** — the robot's own state
  has to be recognised from its joint signals, sample by sample. If the contact
  signature changes (different payload, worn gripper), a classifier trained yesterday
  is stale today.
  → [Online classifiers](online_classifiers.md)

- **"How much torque does this move really need?"** — the URDF model gives the
  rigid-body torque, but a real robot also has the payload in its gripper, friction in
  its joints, and wear. Those are not in the file; they have to be learned.
  → [Online dynamics](online_dynamics.md)

The four share one property: they update one sample at a time, while the robot runs,
without a separate training phase. That is what "online" means in this section.

## Pages

- **[Calibration](calibration.md)** — online Platt scaling: corrects the probabilities
  of any classifier one sample at a time.
- **[Metric learning](metric_learning.md)** — OASIS, LEGO and POLA, and a k-NN
  classifier that learns its distance online.
- **[Online classifiers](online_classifiers.md)** — robot-state classifiers that adapt
  when the data drifts.
- **[Online dynamics](online_dynamics.md)** — learns, sample by sample, the torque the
  URDF model misses.
