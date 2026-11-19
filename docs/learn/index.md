# Online learning: sequential models and incremental architectures for real-time adaptation

## The map of the adaptive learners

Click a page to open it, or the job on the left (or the button at the top) to go back to the [map of the library](../index.md#the-library-folder-by-folder).

<object type="image/svg+xml" data="../assets/maps/learn.svg" style="width:100%"></object>

## Gamestory

New Year's Eve in London. A mysterious thief is emptying the banks of the city.

While the chase goes on, the young detective realizes that police manuals are no longer enough: reality changes constantly and he has to learn one clue at a time, while running through the streets. To avoid being fooled, he asks himself four precise questions:

- **"How sure am I really?"** (Calibration): When the detective says "I am 80% sure it is a trap", he must truly be right 8 times out of 10. He learns to measure his certainty honestly, without exaggerating or underestimating the danger.
- **"Which old heists look like this one?"** (Metric learning): To compare two clues it is not enough to look at numbers at random, otherwise a huge detail (like the millions stolen) hides a small but decisive detail (like the time of the heist). The detective learns on the fly which details truly matter to say that two situations are similar.
- **"Is this move normal or is it a trick?"** (Online classifiers): The rules learned yesterday become old as soon as the thief changes habits. The detective continuously updates his judgment to immediately tell a normal passerby apart from an accomplice.
- **"How much does reality differ from theory?"** (Online dynamics): In school books everything looks easy, but in reality there are rain, wind, and tiredness slowing down the run. The detective learns step by step everything that is not written in the manuals to calculate the exact timing.

## The four questions

A bank watches card payments all day, and four questions come up again and again. Each one is answered by a tool in this section, and each tool learns one sample at a time, while the data arrive, without a separate training phase.

- **"How sure am I that this payment is fraud?"** — a fraud model that says 0.8 should be right 8 times out of 10. If its numbers are not honest, whoever reads them (block the payment or not) decides badly.
  → [Calibration](calibration.md)

- **"Which past customers look like this one?"** — to find similar customers you need a distance, and the obvious one is often wrong: a column with big numbers (a balance in euros) drowns a column with small ones (an age in years).
  → [Metric learning](metric_learning.md)

- **"Is this payment normal?"** — a model trained yesterday goes stale the day customers change habit, or fraudsters change trick.
  → [Online classifiers](online_classifiers.md)

- **"How much does reality differ from the formula?"** — a formula from a manual is almost right, but a real machine has a load, friction and wear. Those are not in the formula: they have to be learned.
  → [Online dynamics](online_dynamics.md)

## Pages

- **[Calibration](calibration.md)** — online Platt scaling: corrects the probabilities
  of any classifier one sample at a time.
- **[Metric learning](metric_learning.md)** — OASIS, LEGO and POLA, and a k-NN
  classifier that learns its distance online.
- **[Online classifiers](online_classifiers.md)** — robot-state classifiers that adapt
  when the data drifts.
- **[Online dynamics](online_dynamics.md)** — learns, sample by sample, the torque the
  URDF model misses.

## The robot examples

The same four questions, for a robot arm:

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
