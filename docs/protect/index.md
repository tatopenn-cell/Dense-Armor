# Protection

Once a sample looks wrong, what do you do with it?

There are three honest answers, and they are not the same:

- It is a **spike** — an isolated impulse, one sample that got corrupted. Replace it
  with the local baseline and move on.
- It is a **regime change** — the signal has genuinely moved to a new level. Pass it
  through, unchanged. The robot is doing something different now.
- It is a **slow drift** — neither a spike nor a level change, but a sustained movement
  of the average. The CUSUM on [Drift detection](../drift/index.md) is the right tool.

An anomaly detector cannot tell these apart. From a single sample's point of view, a
spike and the first sample of a regime change look identical: both are far from the
recent window. Only the *run* of deviating samples, and what happens after the run,
distinguishes them.

## The signal

Same 100 Hz joint velocity throughout the batch: a collision spike at 15 s and a slow
gear drift from 20 s.

![Joint velocity: nominal, spike at 15 s, drift from 20 s](../assets/running_example/signal.png)

The spike is one sample; the drift is a run. The Arbiter decides which is which.

## Pages

- **[Arbiter](arbiter.md)** — labels each point clean, spike or regime change, and
  routes it to the correct corrector. Both a batch version (`classify_segments` +
  `route_and_correct`) and a streaming version (`StreamingArbiter`) with bounded delay.
  Also wired into `Orca(use_arbiter=True)`.
