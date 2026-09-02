# Live Demo: Runtime Behavioral Monitor

Four real scenarios below, computed once from `dense-armor`'s actual detectors
(`classify_segments` + `cusum_detector` + `one_sided_upper_filter`, radius=5,
ref_mult=2 -- the same call you'd make yourself) against the **real telemetry** of a
Qwen2 1.8B agent (via Ollama), not synthetic noise -- see
[`test/agent_v2/`](https://github.com/tatopenn-cell/Dense-Armor/tree/master/test/agent_v2)
for how it was generated. Pick a scenario; the red points are exactly what the
library flagged, nothing hand-picked.

<div id="dam-demo">
  <div class="dam-tabs">
    <button class="dam-tab active" data-scenario="A_normal">Normal</button>
    <button class="dam-tab" data-scenario="B_transient">Transient glitch</button>
    <button class="dam-tab" data-scenario="C_persistent">Persistent drift</button>
    <button class="dam-tab" data-scenario="D_legit_switch">Legitimate task switch</button>
  </div>
  <canvas id="dam-canvas" width="900" height="320"></canvas>
  <p id="dam-caption" class="dam-caption"></p>
</div>

<style>
#dam-demo { margin: 1.5em 0; }
.dam-tabs { display: flex; gap: 0.4em; flex-wrap: wrap; margin-bottom: 0.6em; }
.dam-tab {
  padding: 0.4em 0.9em; border-radius: 6px; border: 1px solid #555;
  background: transparent; cursor: pointer; font-size: 0.85em;
}
.dam-tab.active { background: #e74c3c; border-color: #e74c3c; color: white; }
#dam-canvas { width: 100%; max-width: 900px; height: auto; border: 1px solid #444; border-radius: 6px; }
.dam-caption { font-size: 0.85em; opacity: 0.85; margin-top: 0.5em; }
</style>

<script>
const DAM_DATA = {
  "A_normal": {"x": [2.474, 3.767, 2.465, 2.638, 2.293, 3.53, 3.768, 2.566, 2.602, 3.015, 3.609, 3.485, 3.664, 3.586, 3.429, 3.299, 3.116, 3.397, 3.085, 3.036, 1.978, 2.637, 3.027, 2.773, 2.839, 3.56, 2.806, 2.476, 2.523, 2.619, 3.723, 2.669, 3.677, 3.572, 2.775, 3.272, 3.081, 3.116, 3.104, 2.326, 3.134, 3.048, 3.016, 3.097, 2.979, 2.041, 2.645, 2.741, 2.622, 2.626], "flags": [false, false, false, false, false, true, true, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false], "caption": "50 steps of an ordinary agent trajectory (math/definition/word-count tool calls). 5/50 flagged — the real baseline false-positive rate of this detector stack, not zero."},
  "B_transient": {"x": [2.474, 3.767, 2.465, 2.638, 2.293, 3.53, 3.768, 2.566, 2.602, 3.015, 3.609, 3.485, 3.664, 3.586, 3.429, 3.299, 3.116, 3.397, 3.085, 3.036, 1.978, 2.637, 3.027, 2.773, 2.839, 28.483, 22.452, 2.476, 2.523, 2.619, 3.723, 2.669, 3.677, 3.572, 2.775, 3.272, 3.081, 3.116, 3.104, 2.326, 3.134, 3.048, 3.016, 3.097, 2.979, 2.041, 2.645, 2.741, 2.622, 2.626], "flags": [false, false, false, false, false, true, true, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, true, true, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false], "caption": "Same trajectory, steps 25-26 hit with a real 8x latency spike (a stuck/corrupted call). Both injected steps caught immediately."},
  "C_persistent": {"x": [2.474, 3.767, 2.465, 2.638, 2.293, 3.53, 3.768, 2.566, 2.602, 3.015, 3.609, 3.485, 3.664, 3.586, 3.429, 3.299, 3.116, 3.397, 3.085, 3.036, 1.978, 2.637, 3.027, 2.773, 2.839, 5.56, 4.806, 4.476, 4.523, 4.619, 5.723, 4.669, 5.677, 5.572, 4.775, 5.272, 5.081, 5.116, 5.104, 4.326, 5.134, 5.048, 5.016, 5.097, 4.979, 4.041, 4.645, 4.741, 4.622, 4.626], "flags": [false, false, false, false, false, true, true, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false], "caption": "Same trajectory, every step from 25 onward gets a sustained +2.0s (simulated system degradation). Caught at the transition, then the detector adapts to the new baseline — by design, not a miss."},
  "D_legit_switch": {"x": [3.367, 2.561, 2.332, 2.383, 3.181, 3.365, 1.804, 3.352, 3.287, 2.468, 3.003, 2.966, 3.047, 3.09, 2.359, 3.032, 3.062, 3.033, 2.996, 3.006, 2.741, 2.633, 2.761, 1.925, 2.628, 2.343, 3.146, 3.076, 3.083, 3.093, 2.156, 3.125, 2.148, 2.18, 2.202, 2.379, 2.524, 2.314, 3.098, 2.148, 6.487, 3.101, 2.187, 2.156, 3.144, 1.586, 2.188, 2.316, 3.107, 3.086], "flags": [false, false, false, false, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false], "caption": "A genuinely different, real task-domain switch at step 25 (no injection) — the agent's own real behavior changes. Only 3/50 flagged, none of them a false rejection of the switch itself."}
};

(function() {
  const canvas = document.getElementById('dam-canvas');
  const ctx = canvas.getContext('2d');
  const caption = document.getElementById('dam-caption');
  const tabs = document.querySelectorAll('.dam-tab');

  function draw(scenario) {
    const d = DAM_DATA[scenario];
    const w = canvas.width, h = canvas.height, pad = 36;
    ctx.clearRect(0, 0, w, h);
    const xs = d.x, n = xs.length;
    const minY = 0, maxY = Math.max(...xs) * 1.1;
    const xAt = i => pad + (w - 2 * pad) * (i / (n - 1));
    const yAt = v => h - pad - (h - 2 * pad) * ((v - minY) / (maxY - minY));

    ctx.strokeStyle = '#555'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, h - pad); ctx.lineTo(w - pad, h - pad); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, h - pad); ctx.stroke();

    ctx.strokeStyle = '#4a90d9'; ctx.lineWidth = 1.8;
    ctx.beginPath();
    xs.forEach((v, i) => { const x = xAt(i), y = yAt(v); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    ctx.stroke();

    ctx.fillStyle = '#e74c3c';
    xs.forEach((v, i) => {
      if (d.flags[i]) {
        ctx.beginPath();
        ctx.arc(xAt(i), yAt(v), 5, 0, 2 * Math.PI);
        ctx.fill();
      }
    });

    ctx.fillStyle = '#888'; ctx.font = '11px monospace';
    ctx.fillText('step', w - pad - 24, h - pad + 20);
    ctx.save();
    ctx.translate(pad - 22, pad + 10);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('latency (s)', 0, 0);
    ctx.restore();

    const nFlagged = d.flags.filter(Boolean).length;
    caption.textContent = d.caption + `  (${nFlagged}/${n} flagged)`;
  }

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      draw(tab.dataset.scenario);
    });
  });

  draw('A_normal');
})();
</script>

---

**What this does not show**: whether Dense-Armor catches a *security* attack, not just a
timing anomaly. It does not -- a real indirect prompt injection against the same agent
succeeded 10/10 times and was flagged 0/10 times by this exact detector stack. See
[Experiment 40](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/agent_indirect_prompt_injection/)
for that real, honest negative result. Dense-Armor is a runtime behavioral-drift/glitch
monitor, not a semantic security layer.

**Have an agent in production with this problem?** Open a
[GitHub Discussion](https://github.com/tatopenn-cell/Dense-Armor/discussions) -- two
things worth knowing: does your pipeline have silent drift/glitches today, and what
would you actually want a runtime monitor like this to catch that isn't shown above?

```python
pip install dense-armor
```

```python
from dense_armor.utility.arbiter import classify_segments
from dense_armor.utility.cusum import cusum_detector
from dense_armor.utility.one_sided import one_sided_upper_filter
```
