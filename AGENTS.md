# Repository Agent Rules

## Publishing Major Checkpoints

After a major checkpoint is complete and all required verification passes:

1. Inspect the branch, working tree, recent commits, and `origin` before publishing.
2. Review and stage only the files that clearly belong to that checkpoint.
3. Exclude secrets, raw datasets, caches, generated files not required by the checkpoint, and unrelated or unclear changes.
4. Review the staged diff and rerun the checkpoint's required verification.
5. Create a clear checkpoint commit without rewriting existing history.
6. Push `main` to `origin` only when the current branch is `main` and the remote permits a normal fast-forward push.

Stop without committing or pushing if any file's purpose or scope is unclear, verification fails, or the remote has diverged.

Never force-push, amend commits, reset history, or discard existing changes.
