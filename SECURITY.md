# Security policy

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Report them privately
through GitHub's "Report a vulnerability" button (Security → Advisories) on this
repository, or by email to the maintainers listed in the repository profile.

Include a description, steps to reproduce, the affected version and any suggested fix.
We aim to acknowledge reports within 5 working days and to publish a fix or
mitigation for confirmed issues within 30 days.

## Supported versions

Only the latest released minor version receives security fixes while the project is pre-1.0.

## Design notes

- API keys are accepted as parameters or environment variables, held as
  `pydantic.SecretStr`, and excluded from logs, manifests and serialised models.
- ffmpeg is invoked with an argument list (never through a shell); stream URLs are
  restricted to `http`/`https`.
- Captured audio, transcripts, model weights and `.env` files must never be
  committed; see `.gitignore`.
