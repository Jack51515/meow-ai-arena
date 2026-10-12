# Community submissions

Viewers can submit their own work here: a bot you wrote, a bot you had an AI write, or your optimized version of the warehouse code. Submissions come in as pull requests and, once merged, appear on the [community leaderboard](LEADERBOARD.md). The full guide is in Chinese: [README.md](README.md). This page is a short version.

You can do it by hand, or paste the prompt in [`AGENT_PROMPT.md`](AGENT_PROMPT.md) into your coding agent (Claude Code, Codex, Cursor, ...) and let it do the work.

## What you can submit

| Episode | Folder | What to submit |
|---|---|---|
| 1 · Territory | `community/land/` | one `bot.cpp` |
| 2 · Bomberman | `community/bomberman/` | one `bot.cpp` |
| 3 · Capture the flag | `community/ctf/` | one `bot.cpp` |
| 5 · Warehouse scheduling | `community/warehouse/` | a `src/` folder with your optimized program |

Hand-written bots, AI-written bots, modified versions of the official bots, optimized versions of the warehouse baseline, and fixed versions of a model's code with known defects are all welcome. Episode 4 (Diplomacy) has nothing to submit. Malicious code, code that breaks the episode's rules (file access, network access, spawning processes), and uncredited copies of other people's work are not accepted.

## Layout

```text
community/<episode>/<github-username>__<entry-name>/
    bot.cpp        (bot episodes; single file, up to 256 KB)
    src/           (warehouse; flat folder of .cpp/.h/.hpp/.md, up to 768 KB in total)
    meta.json      (required)
    NOTES.md       (optional)
```

Both parts of the folder name use only lowercase letters, digits and single hyphens, joined by two underscores. The first part must be the GitHub account that opens the pull request, in lowercase.

```json
{
  "author": "Display name",
  "github": "your-github-username",
  "name": "Entry name shown on the leaderboard",
  "description": "One sentence about the approach.",
  "written_by": "human",
  "model": "",
  "tool": "",
  "based_on": "",
  "license": "MIT"
}
```

`written_by` is `human`, `ai` or `human+ai`; fill it in honestly. `model` is required unless `written_by` is `human`. `based_on` names the code you started from, if any. `"license": "MIT"` means you agree to publish the entry under this repository's MIT license.

The code must follow the same rules the models had: C++20, standard library only, compiled with `g++ -O2 -std=c++20`, no file or network access. Bots get 2 seconds for the first turn and 50 ms per turn afterwards. The warehouse program must produce byte-identical output to `warehouse/workspace/baseline/` within 1 GB of memory. The authoritative rules are in each episode's `workspace/` (Chinese).

## Test locally

```bash
python community/check.py community/<episode>/<your-folder>
```

This validates the folder, runs the episode's evaluation (for bots, the unmodified `challenge.py`: a small tournament against all official bots; for the warehouse, baseline and submission are both run on your machine, outputs compared, then speedups computed), and prints a Markdown summary. `--skip-eval` only checks structure and compilation. `--full` plays more games, or runs all eight hidden warehouse workloads instead of the default quick set (five public workloads plus the three fastest hidden ones; the baseline is very slow).

## Open a pull request

1. Fork the repository, clone your fork, create a branch.
2. Add your folder, run `check.py` until it passes.
3. Commit only that folder, push, open a pull request against `main`, and paste the summary into the description.

One entry per pull request, and do not touch other files in the same pull request.

After that: GitHub Actions checks that only your folder changed and runs `check.py` on a GitHub-hosted VM; a maintainer reviews the code; after the merge the entry is evaluated again and the leaderboard is regenerated.

## About the numbers

Community results are measured on GitHub-hosted VMs with a few hundred games per bot. **They are not comparable with the official results from the show.** Bot ratings come from a small tournament of your entry against the official field (baseline = 0); warehouse speedups are the ratio of baseline to submission CPU time on the same machine.

## License and safety

Entries are published under the repository's [MIT license](../LICENSE). Submissions are C++ programs written by strangers: do not build and run them on a machine you care about; use a VM or a container.
