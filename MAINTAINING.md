# Maintaining the board

Everything here is a command in this repo. Set these first:

```sh
export VF_MODELS=/path/to/models    # where manifests' paths point
export VF_DATA=/path/to/data        # holds sets/musdb30, sets/cc30, sets/private30
```

`private30` is licensed, so it's only on the maintainers' machine. Never publish its audio, outputs or per-clip scores. `vf` writes only its means to the repo.

## Add an entry

1. Add `baselines/<name>/manifest.yaml`. `kind` decides who holds the area: `baseline` is someone's model as they run it, `tuned` is the same weights on stock runtimes, and `ours` is the maintainers' own work, which counts in the payout's denominator and is never paid.
2. `vf validate <name>`.
3. Quality, one of:
   - **New weights:** `vf quality <name> --sets musdb30 cc30 private30`, 50 clips each, which takes hours for a big model.
   - **Same weights as a scored entry:** `vf transfer <name> --parent <parent>`, then copy the parent's `results/quality` file to `results/quality/<name>.json` and add `"inherits": "<parent>"`. `vf board` leaves the row off without a recorded pass.
   - **A failed transfer:** `vf quality <name> --backend browser` (or `webgpu`) scores it in full through its own pipeline. A failure can be one low-SDR clip, so decide whether the full run is worth the hours.
4. `vf build`, then `cd dist && pnpx wrangler deploy`. Official runs load from the live site, so it has to be there first.
5. `vf official all https://vocal-frontier.kendell.dev <name>`. Interleave only what you need: the reference is added for you. The board uses each device's newest run, even one made with a file the entry no longer uses, so rerun this after changing a file or engine. One failed run right after a successful one is held until the next session; rerun it then.
6. `vf board`, which also rewrites the README's payout table and the picture a shared link shows. Then `vf build`, deploy again, and commit what changed. The promo video's build script is kept outside this repo and reads the same board.

After the start, a new entry of ours also needs a line in `results/queue.json` (step 2 below), or it lands behind everyone.

## At the start

Write the starting board's names to `results/starting.json` (a JSON list) and commit it. From then on a `baselines/` entry that isn't on it
queues by time like a contestant's. Fill in the dates, `SITE_URL` and `REPO_URL` in the README, and `REPO` and `DEADLINE` in `site/config.js`.

## A contestant's PR

1. Check that the manifest's `github` is the PR's author. One person's entries share one 5-entry limit, whatever logins they use.
2. Add `"<name>": <unix time>` to `results/queue.json`: the time GitHub received the first push that passed `vf validate` with every url and sha256 filled in.
   A later push that changes a file, hash or recipe moves it to that push's time.
3. Download each file from its url, check its sha256, and put it at its path under `VF_MODELS`. `vf validate <name>` again on your machine.
4. Read any engine before running it.
5. Run `private30` and a public spot check (README: How entries get scored), and `vf transfer` for any file that differs from the native one.
   Regenerate the entry's `results/` files yourself rather than keeping the ones in the PR.
6. Then steps 4 to 6 of "Add an entry".

Every `vf` command that times or scores takes the lock, waits for memory and pins threads, so don't wrap them.

## Checks before publishing

- `.venv/bin/python -m pytest -q tests`
- The page's `Output` column and `results/devices` `sdr` compare each device with the native clip result. More than 0.5 dB off counts as failed.
- A provisional S leaves out unmeasured GPUs. Don't quote one as proven.
- A fused export can return near-silence on OpenVINO with no error, so check output level before trusting a score. `sdr: null` from the page means NaN output, not a missing value.

## What has bitten us

- **One heavy job at a time.** `VF_LOCK` (default `~/.cache/vf-timing.lock`) is what enforces this among `vf` commands. Work outside `vf` has to take the same lock, or its contention gets into timings and browsers die at the memory limit, which looks like a model failure. If timings shift for a run of neighbouring models and then recover, suspect the machine.
- **Memory.** The Ryzen has 7 GB. Commands wait for `VF_MIN_MEM_MB` free (default 4800), and a browser run is killed under 1200 MB free. Don't lower the 1200 to make something fit: it's the limit contestants are told about.
- **`/tmp` may be RAM.** Set `TMPDIR` to a disk for exports and browser profiles.
- **Don't end a working session with work pending** unless something is watching it and will act on the result. Put a long queue in one script that runs its steps in order and logs them. Don't edit a script bash is running: it reads the file as it goes.
- **`pkill -f` and `pgrep -f` match your own shell.** Kill by PID.
- **ONNX Runtime maps external data natively.** Overwriting a weights file an open session uses swaps that session's weights without an error.

## What depends on the maintainers' machine

- `vf official` is the Ryzen 5 3501U mini PC with its Vega 8 at `/usr/bin/chromium`. Adreno, Valhall and Gen12LP come from their devices through the page: open `?device=<name>`, run it, and put the downloaded JSON in `results/devices/<name>/`.
- `wrangler.toml` from `vf build` has the account and domain, and deploying needs that Cloudflare login.
- On the mini PC, other projects share the lock file through `VF_LOCK`, so set it to the machine's shared path if there is one.
- `starter/` is reference code with paths to the private bench. Expect to edit them.
