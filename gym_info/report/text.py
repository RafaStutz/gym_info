from __future__ import annotations

from .models import EntropyReport


def print_entropy_report(rep: EntropyReport) -> None:
    """
    Render a simple textual report for entropy metrics.

    This is meant as a quick human-readable summary for interactive use
    in terminals or simple scripts.
    """
    print("=== gym_info entropy report ===")
    print(f"env_id       : {rep.env_id}")
    print(f"run_id       : {rep.run_id}")
    print(f"num_episodes : {rep.num_episodes}")
    print()

    ge = rep.global_entropies
    print("Global entropies (bits):")
    print(f"  H(S)     = {ge.H_S:.6f}")
    print(f"  H(A)     = {ge.H_A:.6f}")
    print(f"  H(A | S) = {ge.H_A_given_S:.6f}")
    print()

    print("Per-episode entropies (bits):")
    if not rep.episode_entropies:
        print("  (no completed episodes)")
        return

    header = "  episode   H(S)        H(A)        H(A | S)"
    print(header)
    for idx, ent in enumerate(rep.episode_entropies):
        print(f"  {idx:7d}   {ent.H_S:10.6f} {ent.H_A:10.6f} {ent.H_A_given_S:10.6f}")
