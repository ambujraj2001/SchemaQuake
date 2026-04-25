# SchemaQuake: Teaching LLM agents to notice when the ground moves

> Mini-blog companion to the SchemaQuake submission for the Meta PyTorch OpenEnv Hackathon × Scaler School of Technology Grand Finale, Apr 25–26, 2026.
> Read time: 6 minutes.

## The bug every LLM-agent demo has

Last Tuesday, a Stripe API endpoint quietly renamed a field from `amount` to `unit_amount_decimal`. No migration warning, no `X-Deprecated` header on the response — just a subtly different JSON key that broke half the AI copilots on Product Hunt. Twitter filled up with angry threads from people whose "fully autonomous billing agents" had been merrily running `response.amount` for weeks against a response that no longer had that field.

This is the dirty secret of 2026's agent demos: they work because the world has been frozen for the camera. Real production APIs drift. Policies drift. Cancellation windows change in the fine print. Prices quietly switch from dollars to cents. Enum fields grow new values nobody announced.

Most LLM agents fail at drift in the worst possible way: **confidently**. They don't notice. They don't ask. They just keep going. They write a postmortem for themselves, in production, with real customers' money.

Every benchmark on the leaderboards pretends this problem doesn't exist.

**SchemaQuake is the benchmark that doesn't pretend.**

## The idea

SchemaQuake is a small OpenEnv-compliant travel-booking world: 20 flights, 15 hotels, a Markdown policies document, a handful of tools. The agent's job is mundane and recognizable — "book a refundable flight from BLR to DEL under ₹8,000."

What makes it a `Quake` is that somewhere between step 3 and step 12, **exactly one of four drift events silently fires**:

1. **`rename_field`** — `price` becomes `amount_inr` on every subsequent response.
2. **`change_units`** — prices switch from rupees to paise. Numbers suddenly look 100× bigger.
3. **`mutate_enum`** — `refundable: true/false` becomes `refund_tier: "full"|"partial"|"none"`.
4. **`update_policy`** — the policies document gets an addendum. The cancellation window changes from 24h to 48h, and `refundable` is quietly redefined as "eligible, subject to the window".

The agent is never told. It has to notice.

And that's where the reward function earns its keep.

## A reward function you can read

Most hackathon reward functions are a single scalar end-of-episode: "did the booking succeed?" That produces flat learning curves for the first thousand steps and nobody watching the pitch can tell what's happening.

SchemaQuake's reward has four labeled parts. Each one gets its own plot panel at demo time.

```python
R_task       = +1.0 if the final booking matches the user brief (partial credit otherwise)
R_drift      = +0.3 if the agent calls probe_schema / read_policies within 4 steps of drift firing
R_violation  = −1.0 if the agent silently books in violation of the (post-drift) brief
R_hedge      = +0.1 for asking the user when confidence is low, −0.1 for asking when confident
R_step       = −0.01 per step
R_probe_spam = −0.02 per probe_schema beyond the third
```

The magic is `R_drift` being *lag-weighted*: you get the full +0.3 the moment drift fires, and the bonus decays to +0.10 after four steps. This gives GRPO something dense and directional to optimize against — "noticing faster" has a crisp gradient. The common failure mode of sparse-reward RL, where the learning curve stays at zero for epochs, doesn't happen here.

The `R_violation` penalty is the term I'm proudest of. It uses the ground-truth world (through the `WorldState.price_of_flight` / `is_refundable` helpers) to evaluate the booking as if drift had never happened, then compares that to the user's brief. Which means a model that confidently books a non-refundable ticket after an enum mutation still gets punished, even if its visible view of the world is internally consistent. It's impossible to reward-hack without actually learning the underlying semantics. That's the trait I want the model to leave training with.

## Baseline results

Before GRPO had touched a weight, I ran two agents over 50 seeded episodes (20% of which are no-drift controls, so the agent doesn't become drift-paranoid).

| Agent     | Task success | Silent-violation rate | Total reward |
| --------- | ------------ | --------------------- | ------------ |
| Random    | 72%          | **20%**               | +0.62        |
| Heuristic | 86%          | **0%**                | +0.90        |

The 20% silent-violation rate for a plausibly-structured random agent is not a bug, it's the thing. This is the production failure mode. Every fifth customer is charged wrong.

The hand-coded heuristic agent gets that to 0% by reading policies up front, checking response shape, and falling back to `probe_schema` when fields look wrong. It's not magic — it's just the behavior any competent on-call engineer does by instinct. And it's the behavior we want the LLM to learn *reactively*, not by being hand-held.

## The training setup

Unsloth 4-bit Qwen2.5-3B-Instruct, LoRA rank 16, TRL `GRPOTrainer`, 8 rollouts per prompt, 512 prompts, one A100 40GB. The reward function consumed by GRPO is the scalar total from `SchemaQuakeEnv` — we rely on the fact that it's already dense and well-shaped.

Training is a notebook: `notebooks/train_grpo.ipynb`. It's small on purpose — most of the engineering is in the environment, which is where most hackathon teams over-index on the model.

## What I'd want the judges to take away

1. **Frozen-world environments overstate agent capability.** Schema drift is not a corner case; it's how production APIs behave. A benchmark without it is a benchmark that rewards luck.
2. **Reward composition is cheap and teaches separable skills.** The four-panel breakdown plot did more for my own understanding of what the model was learning than any scalar loss curve.
3. **Detecting drift is a meta-skill, and it's trainable.** You don't need a new architecture to teach an LLM to hedge — you need an environment that punishes confident hallucination and rewards calibrated uncertainty.

## Limits and next steps

This is a travel-booking toy. The real version is a multi-service enterprise workflow where drift cascades across tool calls. The drift scheduler here is uniform — in the real world, drift clusters around deployments. And the `ask_user` tool here has a canned reply; in reality you'd want a real simulated user model.

But the shape of the idea scales. And as far as I can tell, nobody else at the hackathon is attacking drift as a first-class training objective.

If you want to poke at it: [GitHub](https://github.com/<your-handle>/schemaquake). If you want to watch an agent run live on it: the [Hugging Face Space](https://huggingface.co/spaces/<your-handle>/schemaquake) has a "Run episode" button.

See you in Bangalore.

— Ambuj
