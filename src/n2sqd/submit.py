"""Run the official grader on a result artifact: dry run by default, real submission only on explicit request.

    python -m n2sqd.submit --track b                                  # dry run: validate + recompute, nothing is sent
    python -m n2sqd.submit --track a                                  # same for Track A (results/best/track-a)
    python -m n2sqd.submit --artifact path/to/result.npz              # dry run on any artifact
    python -m n2sqd.submit --track b --submit --email <team email>    # real submission (explicit team decision only)

Project rule: the official grader runs as a dry run; nothing is submitted until the team explicitly decides to. The tool enforces this: without ``--submit`` it never imports the grader's network path,
and ``--submit`` additionally requires ``--email``.

What it does, in order: load the artifact, run the harness (validity rules, grader-identical energy, offline
grader cross-check), print the energy the grader will recompute, then (only with ``--submit``) register the
email with ``save_account`` and call ``grade_lab4b_exbonus``. The grader sends only the email, the two list
sizes, the recomputed energy and the match flag to the organisers' server.

The email is never read from the environment or from files: it must be passed explicitly, because the call is
tied to the team's registration and counts as a submission.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eval.score import score_result
from n2sqd.result import Result

TRACKS = {"a": Path("results/best/track-a/result.npz"), "b": Path("results/best/track-b/result.npz"),
          "baseline": Path("results/best/baseline/result.npz")}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate a result artifact against the official grader. Dry run unless --submit.")
    p.add_argument("--track", choices=sorted(TRACKS), help="which results/best artifact to check or submit")
    p.add_argument("--artifact", type=Path, help="explicit artifact path (overrides --track)")
    p.add_argument("--submit", action="store_true",
                   help="really submit: register --email and call the grader (only on an explicit team decision)")
    p.add_argument("--email", help="the team email registered with the organisers (required with --submit)")
    a = p.parse_args(argv)
    if a.artifact is None and a.track is None:
        p.error("pass --track or --artifact")
    if a.submit and not a.email:
        p.error("--submit requires --email <team email>")
    if a.email and not a.submit:
        p.error("--email only makes sense with --submit; without --submit this is a dry run")
    path = a.artifact or TRACKS[a.track]
    result = Result.load(path)
    score = score_result(result, run_grader_code=True)
    print(f"artifact: {path}")
    print(f"strings per spin: {score.dims[0]} / {score.dims[1]}; claimed {score.claimed_energy:.6f}; "
          f"recomputed {score.energy:.6f}; matches={score.matches}; offline grader agrees={score.grader_agrees}")
    if score.gap_to_reference_mha is not None:
        print(f"gap to the DMRG reference: {score.gap_to_reference_mha:.1f} mHa; below CCSD by {score.below_ccsd_mha:.1f} mHa")
    if not score.matches or score.grader_agrees is False:
        print("refusing to submit: the claim does not match the recomputed energy", file=sys.stderr)
        return 2
    if not a.submit:
        print("dry run: nothing sent (pass --submit --email <team email> to submit)")
        return 0
    from qfest_grader import save_account
    from qfest_grader.challenges.warsaw_2026 import grade_lab4b_exbonus

    save_account(a.email)
    grade_lab4b_exbonus(*result.grader_args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
