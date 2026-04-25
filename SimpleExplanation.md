Alright, let me explain this the way I'd explain it to a friend over chai. No jargon without unpacking it.

---

## 1. What's the big picture?

You're training an AI assistant (an **LLM agent** — basically ChatGPT but told to use tools) to handle a job that sounds simple but has a sneaky twist.

**The job:** "Book me a refundable flight from Bangalore to Delhi for under ₹8,000."

**The twist:** Partway through the booking, the system silently changes something. Maybe it renames a column in the database. Maybe prices suddenly switch from rupees to paise (so ₹5,400 now looks like 540,000). Maybe the cancellation policy gets quietly updated.

The AI is **not told** that anything changed. It has to notice on its own.

A **bad** AI agent keeps going as if nothing happened and confidently books the wrong thing.
A **good** AI agent pauses, says "hmm that number looks weird", re-checks the rules, and asks the user if it's unsure.

**SchemaQuake is the gym where we train the AI to have that "hmm, something feels off" instinct.**

---

## 2. Why does this matter?

Every AI demo you've seen on Twitter works because the demo world is **frozen** — nobody changes anything while the agent runs.

Real life isn't like that.
- Stripe renames an API field on a Tuesday → every AI billing bot in the world breaks.
- A hotel website quietly changes its cancellation window → AI travel bots give wrong refunds to customers.
- An e-commerce API changes prices from dollars to cents → AI shopping assistant charges someone $10,000 instead of $100.

Almost **no hackathon project** is tackling this. That's exactly why the judges will remember yours.

---

## 3. What are we *actually* building?

Three things, which I'll explain like Lego pieces.

### Piece 1: A pretend travel-booking world

Think of this as a tiny fake website with tiny fake data.
- 20 fake flights sitting in a JSON file (`src/schemaquake/data/flights.json`)
- 15 fake hotels in another JSON file (`src/schemaquake/data/hotels.json`)
- A fake terms-and-conditions document in Markdown (`src/schemaquake/data/policies.md`)

The AI agent can "use" this world through **tools** — small Python functions that simulate clicking buttons:
- `search_flights()` — like using a search bar
- `book()` — like clicking "Confirm"
- `cancel()` — like clicking "Cancel my booking"
- `read_policies()` — like reading the fine print
- `probe_schema()` — like opening DevTools to inspect the response
- `ask_user()` — like texting the customer "hey, should I still book this?"

All of these live in `src/schemaquake/tools.py`.

### Piece 2: The "Quake" (the silent changes)

This is the interesting part. A separate file, `src/schemaquake/drift.py`, has four "mischief functions" we can fire at any moment during the booking:

1. **Rename a field** — `price` suddenly becomes `amount_inr`. The AI's code `response["price"]` will now crash or return nothing.
2. **Change units** — All prices multiply by 100 (rupees → paise). `₹5,400` becomes `540,000`. A careless AI thinks that's expensive and filters it out.
3. **Mutate enum** — `refundable: true/false` becomes `refund_tier: "full"/"partial"/"none"`. A careless AI that expected a yes/no answer now has no idea what to do.
4. **Update policy** — The T&Cs document gets new paragraphs quietly appended.

A small "scheduler" (`DriftScheduler`) rolls a dice at the start of every booking:
- 20% of the time: nothing changes (control case, so AI doesn't become paranoid)
- 80% of the time: one of the four changes fires at a random step

### Piece 3: The scoring system (the reward function)

This is where your project *really* wins or loses. It's in `src/schemaquake/rewards.py`.

Every time the AI takes an action, we give it points. Think of it like a video game score:

| What the AI did                                       | Points |
|---|---|
| Booked the right flight matching the user's ask       | **+1.0** |
| Called `probe_schema` within 4 steps of something changing | **+0.3** (decaying) |
| Booked something the user explicitly didn't want      | **−1.0** (big punishment) |
| Asked the user when unsure (confidence < 0.55)        | **+0.1** |
| Asked the user when it was already sure (conf > 0.85) | **−0.1** (wasted their time) |
| Every step it takes                                   | **−0.01** (don't dawdle) |
| Spammed `probe_schema` more than 3 times              | **−0.02** each |

The **total** of these across a full booking is the "reward" for that episode. Higher = better AI.

---

## 4. How are we *training* the AI?

This is the machine-learning part. It sounds fancy but the idea is simple:

1. Start with a small base model (**Qwen2.5-3B** — a Chinese-built open-source LLM, 3 billion parameters, small enough to run on one GPU).
2. Make it play through the booking task thousands of times.
3. Every time it earns more reward, nudge its brain slightly in that direction.
4. Every time it earns less reward, nudge it away.
5. Repeat for hours. Eventually it gets good.

The specific technique is called **GRPO** (Group Relative Policy Optimization). It's from the people who made DeepSeek. The "math" is in a library called `TRL` — we don't implement it, we just configure it. That's what the notebook `notebooks/train_grpo.ipynb` does.

The tool that makes training fast on a small GPU is called **Unsloth**. It compresses the model to 4-bit numbers (instead of 16-bit) so it fits in memory. Again, we don't write this — we use it.

---

## 5. What have we built so far, concretely?

Pull up each folder and this is what you'll see:

| Folder/file                       | What it does                                          |
|---|---|
| `src/schemaquake/types.py`        | Defines Action, Observation, State (the shapes of data that move between AI and environment) |
| `src/schemaquake/world.py`        | The fake travel world + helpers to read "true" prices  |
| `src/schemaquake/tools.py`        | The 7 tools the AI can call                           |
| `src/schemaquake/drift.py`        | The 4 drift/quake functions                           |
| `src/schemaquake/rewards.py`      | The scoring system                                    |
| `src/schemaquake/env.py`          | The main "Environment" class that ties it all together |
| `agents/random_agent.py`          | A dumb baseline AI (so we can prove our training improved things) |
| `agents/heuristic_agent.py`       | A hand-coded "smart" AI (shows it's possible to win)  |
| `agents/llm_agent.py`             | The real LLM agent (plugs into OpenAI / HF)            |
| `eval/run_eval.py`                | Runs the AI 50 times and makes pretty graphs          |
| `tests/`                          | 26 automated checks that prove everything works       |
| `notebooks/train_grpo.ipynb`      | The actual training notebook (for Google Colab)       |
| `demo/app.py`                     | Gradio web UI (the live demo for the judges)          |
| `docs/PITCH.md`                   | Your slide-by-slide script for the 3-minute pitch     |
| `docs/BLOG.md`                    | The mini-blog post for HuggingFace                    |

---

## 6. What will the judges actually check?

They have a rubric. I memorized it from the email. Here's what each item maps to in our project:

### Environment Innovation — 40% of the score
**The question:** Is this environment novel and interesting?
**Our answer:** "Yes — almost no one else is modeling schema drift as a first-class training target. Everyone else has a frozen world."
**Where to show:** `src/schemaquake/drift.py` (the 4 drift operators). Open this file on your laptop during Q&A and walk them through it in 60 seconds.

### Storytelling — 30% of the score
**The question:** Can we understand what you built in 30 seconds?
**Our answer:** The Stripe/Tuesday anecdote at the start of `docs/BLOG.md` and slide 1 of `docs/PITCH.md`.
**Where to show:** Your pitch video + your live demo on Gradio.

### Reward improvement — 20% of the score
**The question:** Does the training actually make the AI better?
**Our answer:** The 4-panel matplotlib plot in `eval_results/`. Random agent (+0.62) vs Heuristic agent (+0.90) today. Trained LLM (hopefully +1.2 or higher) after Colab training on Day 1 onsite.
**Where to show:** `eval_results/heuristic_plot.png` (already exists); the trained-model plot you'll generate onsite.

### Reward & pipeline — 10% of the score
**The question:** Is the reward function thoughtful, not just "did it succeed"?
**Our answer:** Four separate components with unit tests. Capped to prevent reward-hacking. Uses ground-truth semantics, not just visible fields.
**Where to show:** `src/schemaquake/rewards.py` + `tests/test_rewards.py`.

### Minimum requirements (zero-or-one)
- ✅ Uses OpenEnv latest release (0.2.3) — in `src/schemaquake/env.py`, you literally inherit from `openenv.core.env_server.interfaces.Environment`.
- ⏳ Minimal training script in Colab with Unsloth or TRL — that's `notebooks/train_grpo.ipynb` (you'll run it onsite).
- ⏳ <2 min YouTube video OR HuggingFace blog — `docs/BLOG.md` is your blog draft (you post it during the hackathon).

---

## 7. What do *you* need to be able to do?

This is the important part. Judges will ask pointed questions in Q&A. You don't need to remember everything, but you need to be able to:

1. **Open `src/schemaquake/drift.py` and explain what each drift function does** in one sentence each.
2. **Open `src/schemaquake/rewards.py` and explain each of the 6 reward components** in one sentence each.
3. **Show them the 4-panel plot** and point at which panel proves which skill improved.
4. **Answer "why GRPO and not PPO?"** → answer is in `docs/PITCH.md` under Q&A.

If you can do those four things without panicking, you'll clear the bar for 1st prize. Spend 30 minutes today just reading the three files I named. That's the homework.

---

## 8. What's missing / what you still need to do

I couldn't do these because they need **you**:

1. `git add . && git commit -m "..."` — commit with *your* authorship, not mine.
2. Create a GitHub repo and `git push`.
3. Create a Hugging Face account → make a Space → upload `demo/app.py` to it.
4. Actually run the training notebook (you need a GPU — use Google Colab free tier for a practice run before Bangalore, then burn the real compute credits onsite).
5. Record a 2-minute screen-recording video of the demo for YouTube.
6. Publish the blog post on HuggingFace.

Want me to walk you through any of these one-by-one? Start with whichever makes you least comfortable.