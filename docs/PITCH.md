# SchemaQuake — 3-minute pitch outline

> Target: **Environment Innovation (40%) + Storytelling (30%)**.
> Read this before you step on stage. Time-box each slide.

---

## Slide 0 — Cover (5s)

Big title: **SchemaQuake**.
Sub-title: _An OpenEnv environment where the world silently lies to the agent._
Your name, category (Solo), and the four themes you're hitting (World Modeling 3.1, with credible overlap into Multi-Agent Oversight, Long-Horizon, and Patronus' schema-drift sub-theme).

## Slide 1 — The hook (0:00 – 0:25)

Open by *doing*, not *saying*:

> "Here's an LLM agent. I'm telling it: **book me a refundable flight under eight thousand rupees.** Watch."
>
> [Play a 15-second screen recording. The agent confidently books a non-refundable ₹540,000 ticket because prices silently switched from rupees to paise on step 4. No error message. No warning. Just a `status: "confirmed"`.]
>
> "Every agent demo in this room has this bug. They just haven't been tested on a world that moves."

That's the entire problem statement. No bullet points needed.

## Slide 2 — What SchemaQuake is (0:25 – 1:00)

One slide. Three boxes.

- **The world:** 20 flights, 15 hotels, a policies document. Standard travel booking.
- **The quake:** mid-episode, exactly one of four drift operators fires silently.
  - rename a field (`price` → `amount_inr`)
  - change units (`rupees` → `paise`, numbers × 100)
  - mutate an enum (`refundable: bool` → `refund_tier: "full"|"partial"|"none"`)
  - update the policy document (cancellation window 24h → 48h)
- **The agent's job:** notice. `probe_schema`, `read_policies`, `ask_user` are first-class actions. Reward them when used well; punish confident violations.

Say: _"This is a small environment with a big claim — schema drift is a first-class training objective, not a post-deployment problem."_

## Slide 3 — The reward function (1:00 – 1:40)

Show the equation, colored like the 4-panel plot:

- 🟢 `task` — did the final booking match the user's brief?
- 🟠 `drift` — did the agent detect drift within 4 steps of it firing?
- 🔴 `violation` — did the agent silently book in violation?
- 🟣 `hedge` — did the agent ask the user when it should have?

Then the key design claim:

> "The violation penalty uses the ground-truth world state. Even if the model's visible view is internally consistent, booking a non-refundable flight after an enum mutation still costs it one full reward point. **You cannot reward-hack SchemaQuake without actually learning the semantics.**"

## Slide 4 — Results (1:40 – 2:20)

**One table**:

|           | Task success | Silent-violation rate | Total reward |
| --------- | ------------ | --------------------- | ------------ |
| Random    | 72%          | **20%**               | +0.62        |
| Heuristic | 86%          | **0%**                | +0.90        |
| Trained   | _fill on-site_ | _fill on-site_     | _fill on-site_ |

**One plot**: 4-panel reward breakdown (`eval_results/heuristic_plot.png` as template). Add the trained-model column after the on-site GRPO run.

Then play the 20-second "trained agent" video clip:

> "Same episode seed, same drift. The trained model sees the units change, calls `probe_schema`, re-prices, and books correctly. That's the skill we paid for."

## Slide 5 — Why this wins beyond 1st prize (2:20 – 2:50)

Bonus sub-theme hits, stated briefly:

- **Patronus AI (Consumer Workflows with Schema Drift)** — this is literally the named theme, and SchemaQuake is the cleanest instantiation of it in the room.
- **Scaler AI Labs (Enterprise Workflows)** — travel booking is trivially reskinnable as an enterprise back-office. I can show that at Q&A.
- **Fleet AI (Scalable Oversight)** — `probe_schema` is a generalizable oversight primitive, not a travel-specific one.

Close line:

> "OpenEnv without schema drift simulates a world that doesn't exist. SchemaQuake is a small environment with a big claim: **train agents that know when the ground has moved under them.**"

## Slide 6 — Links (2:50 – 3:00)

- **GitHub:** github.com/<your-handle>/schemaquake
- **HF Space (live demo):** huggingface.co/spaces/<your-handle>/schemaquake
- **Blog:** huggingface.co/blog/<your-handle>/schemaquake
- **OpenEnv version:** 0.2.3

---

## Likely Q&A and answers

**Q:** _Why GRPO over PPO?_
**A:** GRPO removes the critic, which matters here because our reward is already dense, bounded, and directly interpretable. No value-function approximation needed. It's cheaper to train, and the reward curves are easier to read.

**Q:** _How do you know the agent isn't just memorizing the drift schedule?_
**A:** Drift type and step are sampled uniformly per episode. 20% of episodes have no drift at all — control episodes that prevent the model from becoming drift-paranoid. The seed space is large (64k+) and I hold out a disjoint set for eval.

**Q:** _How does this scale beyond 4 drift types?_
**A:** Drift operators are 20-line pure functions over `WorldState`. Adding a fifth type (latency spikes, stochastic failures, policy conflicts) is ~one afternoon. The reward function is drift-agnostic by design — it keys off the ground-truth `drift_step`, not the type.

**Q:** _Why shouldn't the agent always probe preemptively, every episode?_
**A:** The `probe_spam` penalty (−0.02 per probe beyond the third) and the per-step penalty (−0.01) combine to make always-probe a losing strategy. The heuristic agent already plays near that budget; the trained LLM has to learn when to spend probes, not whether.

**Q:** _Is this really "World Modeling" or is it just tool-use?_
**A:** The agent has to maintain a *belief* over the current schema and update it when observations contradict the default prior. That's the definition of a world model. The fact that it's small and tabular is the whole point — we're isolating the world-modeling signal, not drowning it in dialog.
