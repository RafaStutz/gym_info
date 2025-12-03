from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym

from ..api import Entropies, Summary, entropies, entropies_per_episode, summary
from .style import render_entropy_report_html


@dataclass(frozen=True)
class EntropyReport:
    """
    Aggregated view of entropy metrics for a single instrumented run.

    It combines:
    - basic run metadata (env_id, run_id, num_episodes),
    - global entropies over the whole trajectory,
    - per-episode entropies in chronological order.
    """

    env_id: str
    run_id: str | None
    num_episodes: int
    global_entropies: Entropies
    episode_entropies: list[Entropies]

    def as_html(
        self,
        include: tuple[str, ...] = ("H_S", "H_A", "H_A_given_S"),
    ) -> str:
        """
        Render this report as an HTML string with inline CSS.

        Parameters
        ----------
        include:
            Tuple of entropy keys to display. Valid entries are "H_S",
            "H_A", and "H_A_given_S". The same selection is applied to
            the global and per-episode tables.
        """
        return render_entropy_report_html(self, include=include)

    def _repr_html_(self) -> str:
        """
        Rich HTML representation hook for IPython and Jupyter.

        This method allows an EntropyReport instance to be displayed
        directly in notebook environments, without explicitly calling
        any rendering helper. All entropy metrics are shown.
        """
        return render_entropy_report_html(self)


def build_entropy_report(env: gym.Env) -> EntropyReport:
    """
    Build an EntropyReport for the given environment.

    The environment must be one returned by gym_info.attach. The report
    uses:
      - gym_info.summary(env) for run metadata;
      - gym_info.entropies(env) for global entropies;
      - gym_info.entropies_per_episode(env) for per-episode entropies.
    """
    run_summary: Summary = summary(env)
    global_ent: Entropies = entropies(env)
    episode_ents: list[Entropies] = entropies_per_episode(env)

    return EntropyReport(
        env_id=run_summary.env_id,
        run_id=run_summary.run_id,
        num_episodes=run_summary.num_episodes,
        global_entropies=global_ent,
        episode_entropies=episode_ents,
    )


def report(env: gym.Env) -> EntropyReport:
    """
    Convenience alias for build_entropy_report.

    Provided to give a short, user-friendly entry point:
        rep = gym_info.report(env)
    """
    return build_entropy_report(env)
