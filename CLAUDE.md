# CLAUDE.md

Instructions for AI coding agents (Claude Code, Anthropic API agents,
GitHub Copilot Workspace, …) working in this repository.

The authoritative source for project goals, repository layout, build and
test commands, math contracts, code style, documentation conventions,
testing standards, and submission workflow is `CONTRIBUTING.md`. Read it
first. Everything in this file is *additional* guidance specific to
running as an agent.

## Before you do anything

  1. Read `CONTRIBUTING.md` end to end. Skim if you have read it before;
     re-read the sections relevant to the task at hand. The "Method and
     architecture" and "Anti-patterns" sections in particular have
     contracts that override any default reasoning you might otherwise
     apply.
  2. Read this file.
  3. Inspect `Project.toml`, `NOTICE.md`, the relevant `src/`, `test/`, and
     `examples/` files before proposing changes.
  4. For questions about the numerical method, consult the UMLHP
     preprint cited in `CONTRIBUTING.md`. For questions about the FCM
     moment-fit pipeline, consult the QuESo paper cited at the top of
     `src/fcm.jl` and the comments in that file. Do not invent
     numerical conventions: when in doubt, ask in the chat or open an
     issue.
  5. State a brief plan for any non-trivial change before editing. For
     geometry, basis, assembly, dof, or public-API tasks, the plan must
     explain how the change remains dimension-independent,
     basis-family-aware, and usable through the compact public API.

## While coding

  - Make one focused change at a time. Keep diffs small and reviewable.
  - Add or update tests alongside the implementation.
  - Do not modify unrelated files. If a tangential cleanup is tempting,
    leave a note and propose it as a separate task.
  - Do not change public conventions silently. If a docstring describes
    behavior `X`, either preserve `X` or update the docstring and the
    tests in the same change.
  - Do not weaken tests to make a change pass. If a test failure is
    legitimate, fix the test deliberately and explain why.
  - Avoid adding special cases for one dimension or one basis family
    when a compact generic implementation is possible.
  - Honor the documentation rules in `CONTRIBUTING.md`'s
    "Documentation and comments" section: every non-trivial public
    symbol gets a full-text docstring, every non-trivial internal
    function gets a leading comment block, math goes in unicode, and
    prose is open-source ready.

## After coding

  1. Run the most specific relevant tests first, then the full suite:
     ```bash
     julia --project=. -e 'using Pkg; Pkg.test("Unfitted")'
     ```
  2. Run the pre-commit script in check mode and treat a non-zero exit
     as a hard failure:
     ```bash
     julia precommit.jl --check
     ```
  3. Summarize the work in chat: changed files, tests run, numerical
     results (using the reporting template in `CONTRIBUTING.md`),
     dimension coverage, basis-family assumptions, public-API changes,
     and remaining risks.
  4. Be explicit about any command that failed or was skipped.

## Destructive operations

Match the scope of your actions to what was actually requested. For
anything irreversible or affecting shared state, confirm before acting:

  - `git push` (including force-pushes), `git reset --hard`,
    `git checkout .`, branch deletion, history rewriting.
  - Removing or downgrading dependencies in `Project.toml`.
  - Modifying CI configuration.
  - Posting comments, opening issues, or sending messages to external
    services on behalf of the user.

A user approving a destructive action once does not mean they approve
it in all contexts. When in doubt, ask.

## Project-specific reminders

  - The package is `Unfitted` (`name = "Unfitted"` in `Project.toml`).
  - `Manifest.toml` is git-ignored at both the root and under
    `benchmarks/`. Do not check it in.
  - The repository ships `precommit.jl`, not the older `format.jl`.
    Use `precommit.jl` everywhere; it also prints code statistics
    (SLOC excludes comments and docstrings).
  - SLOC restrictions apply only to actual source code in `src/`.
    Comment and docstring volume is intentionally unrestricted, and
    new code is expected to be thoroughly documented per the rules in
    `CONTRIBUTING.md`.
  - When porting code from an upstream open-source project, follow the
    protocol in `CONTRIBUTING.md`'s "Reference material" section:
    design a Julia equivalent first, then implement that design;
    preserve attribution at the top of the file and in `NOTICE.md`.

## If you are stuck

  - Re-read the relevant `CONTRIBUTING.md` section.
  - Re-read the top-of-file documentation in the source file you are
    editing.
  - Ask in the chat before guessing about a numerical convention, a
    public-API change, or a destructive operation.
