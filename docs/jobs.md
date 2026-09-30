# Local folding jobs

The optional local queue runs **ColabFold only**. It accepts a single FASTA
record or a complete local A3M file, snapshots the input and executes one MPS
job at a time. It has no HTTP listener and no MMseqs2 search option. Missing
weights cause an error rather than a download; the worker blocks upstream
MMseqs2 and weight-download calls as a second safeguard. Jobs are intended for
local agents and humans running as the same macOS user.

Install a durable CLI before installing the login worker. From this checkout:

```bash
uv tool install --python 3.12 .
macfoldkit setup colabfold
macfoldkit fetch colabfold
macfoldkit jobs install
macfoldkit jobs status
```

The first two ColabFold commands are unnecessary if that runtime and its
weights are already installed. Complexes require the separately downloaded
`alphafold2_multimer_v3` weights. `jobs install` creates a per-user LaunchAgent
using the Python interpreter running the CLI; reinstall the tool and stop/reinstall
the worker if that Python environment moves. The worker starts on login and
processes pending jobs. It does not keep the Mac awake while idle.

```bash
macfoldkit jobs submit sequence.fasta
macfoldkit jobs submit local.a3m --model-type alphafold2_ptm
macfoldkit jobs submit complex.fasta --model-type alphafold2_multimer_v3
macfoldkit jobs list
macfoldkit jobs status JOB_ID
macfoldkit jobs logs JOB_ID
macfoldkit jobs logs JOB_ID --all
macfoldkit jobs cancel JOB_ID
macfoldkit jobs retry JOB_ID
macfoldkit jobs delete JOB_ID --yes
```

## Input contract

`jobs submit PATH` accepts **one existing local UTF-8 file** with a
case-insensitive `.fasta`, `.fa`, `.faa` or `.a3m` suffix. It does not accept
directories, stdin, CSV, PDB, YAML, or multiple FASTA jobs in one file. Blank
lines are ignored and sequences can wrap across lines. Each sequence needs a
nonempty `>` header. Submission copies the original bytes to `input.<suffix>`
in the job directory and records their SHA-256; changing the source file later
does not change the job. The worker verifies the snapshot and cached weights
again before running.

- **FASTA:** Exactly one record. The sequence consists of letters A-Z
  (case-insensitive), optionally separated into nonempty chains with `:`.
  Multiple chains require `--model-type alphafold2_multimer_v3`; otherwise
  the default is `alphafold2_ptm`. No gaps or whitespace within the sequence.

  ```fasta
  >sample
  ACDEFGHIKLMNPQRSTVWY
  ```

  A two-chain FASTA for `--model-type alphafold2_multimer_v3`:

  ```fasta
  >complex
  ACDE:FGH
  ```

- **A3M:** The first record must be an ungapped, **uppercase** query of
  letters A-Z. Any subsequent alignment rows can contain uppercase aligned
  residues, `-` gaps, lowercase insertions, and `.`; after removing lowercase
  letters and `.` each row must span the query's aligned length. Supply the
  **whole local alignment**, not a path or identifier for a remote search.

  ```text
  >query
  ACDEFGH
  >homolog
  ACdDEFGH
  ```

  For multiple chains, use a ColabFold multimer A3M whose first line is
  `#L1,L2<TAB>C1,C2`: comma-separated chain lengths, a **literal tab**, then
  positive chain copy counts. The lists must have the same number of entries
  and the lengths must sum to the first query's length. More than one chain
  requires `--model-type alphafold2_multimer_v3`. For example, this header has
  lengths 4 and 3, one copy of each chain (the space between `4,3` and `1,1`
  below is a tab):

  ```text
  #4,3	1,1
  >query
  ACDEFGH
  >paired
  ACDEFGH
  ```

  Without a multimer header, ColabFold treats the A3M as a single chain.
  The queue allows `.` syntactically, but pinned ColabFold handles it
  differently when splitting multimer rows; prefer lowercase insertions and
  `-` gaps for multimer alignments.
  Syntax validation does not guarantee a usable or biologically meaningful
  alignment. The queue preserves local A3M rows; a FASTA instead runs
  single-sequence inference. Neither input starts an MMseqs2 search.

The only submit options are `--model-type` (`alphafold2_ptm` by default,
or `alphafold2_multimer_v3`), `--num-recycle N` (default 3), and `--seed N`
(default 7); the two numbers must be nonnegative. The runner uses model 1,
one seed and no relaxation. No arbitrary ColabFold flags are accepted.
`macfoldkit --home PATH jobs submit ...` selects a runtime/weight location
saved with that job; the job database and outputs remain in Application
Support. The selected model's weights must already be cached.
The queue does not accept PDB/CIF initial guesses or structure templates;
use the [standalone ColabFold command](colabfold.md#local-templates-and-initial-guesses)
for those inputs.

## CLI output contract

`submit`, `status JOB_ID`, `cancel` and `retry` each print one JSON object with
the same per-job fields. The following **illustrative** `status JOB_ID` response
shows a completed job (paths depend on your macOS username):

```json
{
  "id": "0123456789abcdef0123456789abcdef",
  "status": "succeeded",
  "created_at": "2026-09-30T20:00:00+00:00",
  "updated_at": "2026-09-30T20:00:08+00:00",
  "started_at": "2026-09-30T20:00:01+00:00",
  "ended_at": "2026-09-30T20:00:08+00:00",
  "source_name": "sequence.fasta",
  "extension": ".fasta",
  "sha256": "5f6b43655364ee435039d662ecf0eca136f6ac982025cfcd6a71d0fc48ccd5c0",
  "runtime_home": "/Users/example/Library/Caches/macfoldkit",
  "model_type": "alphafold2_ptm",
  "num_recycle": 3,
  "seed": 7,
  "attempt": 1,
  "cancel_requested": 0,
  "pid": null,
  "returncode": 0,
  "error": null,
  "input": "/Users/example/Library/Application Support/macfoldkit/jobs/0123456789abcdef0123456789abcdef/input.fasta",
  "results": "/Users/example/Library/Application Support/macfoldkit/jobs/0123456789abcdef0123456789abcdef/attempts/1/output",
  "log": "/Users/example/Library/Application Support/macfoldkit/jobs/0123456789abcdef0123456789abcdef/attempts/1/run.log",
  "storage_bytes": 184000
}
```

| Fields | Type and meaning |
|---|---|
| `id`, `status` | 32 lowercase hex characters; `pending`, `running`, `succeeded`, `failed`, `cancelled` or `interrupted`. |
| `created_at`, `updated_at`, `started_at`, `ended_at` | UTC ISO-8601 timestamps (`+00:00`); the latter two can be `null`. Retry clears start/end timestamps. |
| `source_name`, `extension`, `sha256`, `input` | Original basename, normalized suffix, SHA-256 hex digest of the copied bytes, absolute input-snapshot path. |
| `runtime_home`, `model_type`, `num_recycle`, `seed` | Runtime/weights path and fixed submission settings. |
| `attempt`, `results`, `log` | Attempt number (0 until first claim); absolute output and worker-log paths, or `null` before any attempt. Paths can precede file creation. After retry is queued but before it starts, these still refer to the *previous* attempt. |
| `cancel_requested`, `pid` | Integer `0` or `1` for a requested running-job cancellation; worker-launched process PID or `null`. A cancel request can initially return `status: "running"`; poll for completion. |
| `returncode`, `error` | ColabFold process exit code (`0` on success), or `null` if no exit code was recorded; diagnostic string or `null`. |
| `storage_bytes` | Current sum of job input, log and result file sizes; excludes the SQLite database and worker log. It can grow during a run. |

`jobs list` returns an object with `jobs` (an array of per-job objects, newest
first) and `storage_bytes` (their total). `jobs status` **without an ID**
returns `jobs` as counts for all six statuses, plus `storage_bytes`, `home`
(the absolute jobs root) and `worker_loaded` (a Boolean). `worker_loaded`
means launchd has the agent registered, not that a prediction is succeeding.
`jobs logs JOB_ID` prints the most recent attempt's last 100 lines as text;
`--tail N` changes the line count, `--all` returns the full log, and `--json`
returns
`{"id": "JOB_ID", "log": "..."}`. Before the first attempt, there is no log;
while a retry is pending, `logs` still shows the previous attempt.
`jobs delete JOB_ID --yes` returns `{"deleted": "JOB_ID"}`.
Install/uninstall also return JSON; `jobs worker` is a long-running command,
not a JSON API.

Accepted submissions are asynchronous: a successful `submit` means the job
was queued, **not** that inference succeeded. Poll `status JOB_ID`; a running
cancel sets `cancel_requested` before the status changes. `failed`, `cancelled`
and `interrupted` jobs need explicit `retry`; a successful job cannot be
retried. Validation/runtime errors before acceptance print `macfoldkit: ...`
to stderr and exit 1 without a job ID; bad CLI syntax exits 2. Worker failures
after acceptance appear in `status.error` and `logs`, not in the original
submission's exit status.

## Result files

The per-user layout is:

```text
~/Library/Application Support/macfoldkit/
  jobs.sqlite3
  worker.log
  jobs/<id>/
    input.<suffix>             # unmodified submission snapshot
    attempts/<n>/
      run.log                  # worker + runner stdout/stderr
      output/
        benchmark.json        # MacFoldKit runner metrics
        model-device-audit.jsonl
        process.log           # ColabFold subprocess stdout/stderr
        <query>_unrelaxed_rank_*.pdb
        ...                   # other upstream ColabFold files
```

For `status: "succeeded"`, `results` points to this attempt's `output/`,
which has at least one **unrelaxed PDB** and a nonempty verified MPS audit.
Read the absolute PDB paths from `benchmark.json`'s `fresh_predictions` rather
than guessing the sanitized ColabFold query name or rank filename. The runner
writes `benchmark.json`, `process.log` and `model-device-audit.jsonl` on
successful runs. Failed, cancelled or interrupted attempts may leave only a
`run.log` or partial results; check status before consuming any artifacts.
Previous attempt directories are preserved on retry.

`benchmark.json` is a single JSON object written by MacFoldKit:

| Field | Meaning |
|---|---|
| `fresh_predictions` | Nonempty list of absolute, newly generated unrelaxed PDB paths on a successful job. |
| `model_output_devices_verified`, `model_apply_calls` | Boolean MPS-output audit result; count of audited model applications (not a count of proteins). A successful job has `true` and a positive count. |
| `returncode`, `failed_queries` | ColabFold subprocess exit code and list of detected per-query error messages; on success, `0` and `[]`. |
| `backend`, `cpu_feature_preprocessing`, `async_dispatch`, `optimized_batching` | Backend name (`"mps"`) and Boolean execution settings. CPU feature preparation is expected. |
| `process_seconds`, `maximum_resident_set_bytes` | Runner subprocess wall time in seconds and peak resident *CPU/process* memory in bytes (not GPU memory or queue wait time). |
| `command`, `versions` | Executed argument array (including local paths) and package-name-to-version mapping. |

`model-device-audit.jsonl` has one JSON object per model application, with
`model_instance`, `apply_call`, `ok`, and, for successful calls,
`platforms: ["mps"]`, `devices`, `coordinate_devices`, `coordinate_shape`,
`coordinate_dtype`, and `jax_array_leaves`. Failed calls include `error`.
The PDB is a prediction, not an experimentally validated structure.
ColabFold may also emit `*_scores_rank_*.json` (e.g., per-residue `plddt`,
scalar `ptm`, pairwise `pae`), `*_predicted_aligned_error_*.json`, an A3M,
plots, `config.json` and `log.txt`. These **upstream filenames and schemas
are not a stable MacFoldKit API**; inspect their contents for the pinned
ColabFold version.

Jobs, outputs and the SQLite database live under
`~/Library/Application Support/macfoldkit`, **not** a purgeable cache. Nothing
is removed automatically. Retry keeps previous attempt logs and outputs; it is
allowed for failed, cancelled or interrupted jobs, but not successful ones.
Uninstalling the worker (`macfoldkit jobs uninstall`) leaves all jobs intact.

An active fold holds an idle-sleep assertion with `caffeinate -i`, **including
on battery**. It does not prevent sleep when the lid is closed. On normal worker
shutdown, an active job is stopped and marked interrupted; after an unexpected
crash, the next worker marks it interrupted. Pending jobs resume automatically,
but interrupted jobs require explicit `retry` and use a fresh attempt directory.
If the worker was force-killed and its last recorded process still exists, the
replacement worker refuses to start another fold until that process exits; inspect
its PID and the attempt log before retrying. `jobs worker --once` runs one pending
job in the foreground for debugging without installing launchd.

ColabFold model predictions and confidence are not experimental evidence of
folding or binding. For an MSA-backed complex, prepare and supply a complete
local A3M yourself. The standalone `macfoldkit fold` command still supports
explicit public MSA search; that is **not** available through `jobs`.
