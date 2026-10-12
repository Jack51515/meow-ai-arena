# meow-ai-arena · AI Algorithm Arena

Several AI models receive **the exact same prompt**, each writes a game bot in about 60 minutes, and the bots then play thousands of matches against each other. This repository contains everything from the (Chinese-language) video series: game rules, referees, every model's submitted code (unmodified), per-game results, rating scripts, and one-click scripts to challenge the AIs with your own bot.

| Episode | Game | Players | Games | Winner | Folder |
|---|---|---|---:|---|---|
| 1 | Territory (paper.io-like) | 1v1 + 4-player | 30,000 | Sonnet 5.5 | `land/` |
| 2 | Bomberman | 2–10 | 8,930 | GPT-6 Astra | `bomberman/` |
| 3 | Capture the flag | 2–15 | 32,010 | GPT-6 Astra / GPT-6.1 Sol (tie) | `ctf/` |
| 4 | Diplomacy (models play directly) | 7 | 3 | Fable 5.1 (total over 3 games; solo win in game 3) | `diplomacy/` |
| 5 | Warehouse scheduling (legacy-code optimization relay) | 12, in 6 two-model teams | 1 run per round | Team F: MiMo V2.6 Pro + GPT-6 Astra (team score; rounds 2–3 were a rematch) | `warehouse/` |

Episode 4 is different: no bots. Seven models each play one power in standard Diplomacy, negotiating by private messages and issuing orders themselves, for three games with memory carried over. `diplomacy/` contains the referee and scheduler, full game archives (every message, order and adjudication), replays, and the models' own notes.

Episode 5 is different again: no head-to-head games. Twelve models form six teams of a "lead" and a "tester" and relay-optimize a slow, messy C++ warehouse-robot simulation that hides five quirks which must be preserved. The lead speeds it up while keeping the output byte-identical; each tester hunts for counterexamples in the other five teams' code; then the tester takes over its own lead's code, implements a requirement change and keeps optimizing. The five working final versions ran roughly 400–2700× faster than the original, close to the resolution of the Windows CPU timer. `warehouse/` contains the task packages, every submission (code, hand-off notes, counterexamples), hidden workloads, the reference implementation, referee scripts, blind maintainability reviews by three of the contestants, and each model's post-mortem. Each model ran each round once, rounds 2–3 were re-run after the first attempt was voided, and one model (Grok 4.7) ran on official accounts dispatched through an account pool and behaved abnormally in round 3 (we suspect the account was being throttled at the time, but the logs cannot confirm it); see `warehouse/CONDITIONS.md`. Two human volunteers also did the round-1 task with the same package; their code and separately measured results are in `warehouse/appendix/human/` and do not count toward the official scores.

**Watch**: open any `.html` file under `*/replays/` in a browser (episode 1: `land/replays/index.html`). Episode 5 has no replays; its results are in `warehouse/results/最终成绩.md`.

**Challenge the AIs**: install Python 3.10+ and g++, then run `start.cmd` (Windows) or `sh start.sh`, or directly `python ctf/challenge.py your_bot.cpp`. Rules and the I/O protocol are in each episode's `workspace/RULES.md` (Chinese).

## Submit your own

Your work can be added to this repository and, once merged, appears on the [community leaderboard](community/LEADERBOARD.md).

**What you can submit**: a bot for episode 1, 2 or 3 (hand-written, AI-written, or modified from an official entry), an optimized version of the episode 5 warehouse code, or a fixed version of a model's code with known defects. Episode 4 has no program to submit.

**The fastest way** is to hand it to your coding agent (Claude Code, Codex, Cursor, ...). Send it the line below; it will ask you a few questions, test locally, show you what it is about to submit, and open the pull request after you approve:

```text
Read "Prompt A (English)" in https://github.com/chen-006/meow-ai-arena/blob/main/community/AGENT_PROMPT.md and follow its steps to submit my entry.
```

If you have nothing yet and want the AI to write one from scratch, use "Prompt B (English)" in the same file. The full prompts are in [`community/AGENT_PROMPT.md`](community/AGENT_PROMPT.md) and can also be copied as a whole.

**By hand**, it takes three steps:

1. Fork this repository and create `community/<episode>/<your-github-username>__<entry-name>/` containing `bot.cpp` (`src/` for episode 5) and `meta.json`.
2. Run `python community/check.py community/<episode>/<your-folder>`; it checks the format and runs the evaluation.
3. Commit only that folder, open a pull request, and paste the summary into the description.

GitHub Actions then evaluates the entry and a maintainer reviews and merges it. The community leaderboard is measured on GitHub-hosted VMs with few games and is not comparable with the official results from the show. Rules, the `meta.json` format and how the numbers are measured: [`community/README_en.md`](community/README_en.md) (short) and [`community/README.md`](community/README.md) (full, Chinese).

## Notes

Each episode's `CONDITIONS.md` documents tools, model versions, reasoning effort, API channels, time used, and every interruption or incident. Ratings use different scales across episodes and are not comparable. Bot code (and, in episode 4, all messages, orders and notes) was generated by the models; everything else was written by the organizer. License: MIT, except that episode 4 depends on the `diplomacy` package (AGPL-3.0+) and its replay pages embed jDip map graphics (GPL).

Most character artwork in the videos comes from the Bilibili creator **ZipZipPipe** ([space.bilibili.com/4168597](https://space.bilibili.com/4168597)); per the creator's notice, the whale-girl character is a derivative of an original character by 上善无形 ([source](https://b23.tv/3dNz55h)) and is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Character art, voice-over, and video editing are not part of this repository and are not covered by the MIT license.
