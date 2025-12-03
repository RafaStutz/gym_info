from __future__ import annotations

import html

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import EntropyReport


_CSS = """
.gym_info-report {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 14px;
  line-height: 1.4;
  color: #222;
  max-width: 720px;
}

.gym_info-report .report-header {
  border-bottom: 1px solid #ddd;
  padding-bottom: 8px;
  margin-bottom: 12px;
}

.gym_info-report .report-title {
  font-weight: 600;
  font-size: 16px;
  margin-bottom: 4px;
}

.gym_info-report .report-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  font-size: 13px;
  color: #555;
}

.gym_info-report .report-meta .meta-label {
  font-weight: 600;
}

.gym_info-report .report-section {
  margin-top: 12px;
  margin-bottom: 12px;
}

.gym_info-report .section-title {
  font-weight: 600;
  margin-bottom: 4px;
}

.gym_info-report .entropy-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.gym_info-report .entropy-table th,
.gym_info-report .entropy-table td {
  border: 1px solid #ddd;
  padding: 4px 6px;
  text-align: right;
}

.gym_info-report .entropy-table th:first-child,
.gym_info-report .entropy-table td:first-child {
  text-align: left;
}

.gym_info-report .entropy-table thead {
  background-color: #f5f5f5;
}

.gym_info-report .entropy-table tbody tr:nth-child(odd) {
  background-color: #fafafa;
}

.gym_info-report .entropy-table tbody tr:nth-child(even) {
  background-color: #ffffff;
}

.gym_info-report .entropy-table .epi-index {
  text-align: right;
}

.gym_info-report .entropy-table .epi-value {
  text-align: right;
}

.gym_info-report .no-episodes {
  font-size: 13px;
  color: #666;
  margin-top: 4px;
  font-style: italic;
}
"""


def render_entropy_report_html(
    rep: EntropyReport,
    include: tuple[str, ...] = ("H_S", "H_A", "H_A_given_S"),
) -> str:
    """
    Render an EntropyReport as an HTML snippet with inline CSS.

    Parameters
    ----------
    rep:
        The entropy report to render.
    include:
        Tuple of entropy keys to display. Valid entries are "H_S",
        "H_A", and "H_A_given_S". The same selection is applied to the
        global and per-episode tables.

    Intended for notebook environments (Jupyter, Colab, etc.). The
    returned string can be passed directly to IPython.display.HTML.
    """
    env_id = html.escape(rep.env_id)
    run_id = html.escape(rep.run_id) if rep.run_id is not None else "—"
    num_episodes = rep.num_episodes

    ge = rep.global_entropies

    valid_keys = ("H_S", "H_A", "H_A_given_S")
    selected = tuple(k for k in include if k in valid_keys)

    label_map: dict[str, str] = {
        "H_S": "H(S)",
        "H_A": "H(A)",
        "H_A_given_S": "H(A | S)",
    }

    if not selected:
        global_table_html = """
        <div class="no-episodes">
          No entropy metrics were selected for display.
        </div>
        """
        episodes_table_html = """
        <div class="no-episodes">
          No entropy metrics were selected for display.
        </div>
        """
    else:
        global_header_cells = "".join(f"<th>{label_map[key]}</th>" for key in selected)
        global_value_cells = "".join(
            f"<td>{getattr(ge, key):.6f}</td>" for key in selected
        )

        global_table_html = f"""
        <table class="entropy-table entropy-table-global">
          <thead>
            <tr>
              <th></th>
              {global_header_cells}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Run</td>
              {global_value_cells}
            </tr>
          </tbody>
        </table>
        """

        episode_rows: list[str] = []
        for idx, ent in enumerate(rep.episode_entropies):
            value_cells = "".join(
                f"<td class='epi-value'>{getattr(ent, key):.6f}</td>"
                for key in selected
            )
            episode_rows.append(
                (f"<tr><td class='epi-index'>{idx}</td>{value_cells}</tr>")
            )

        if episode_rows:
            header_cells = "".join(f"<th>{label_map[key]}</th>" for key in selected)
            episodes_table_body = "\n".join(episode_rows)
            episodes_table_html = f"""
            <table class="entropy-table entropy-table-episodes">
              <thead>
                <tr>
                  <th>Episode</th>
                  {header_cells}
                </tr>
              </thead>
              <tbody>
                {episodes_table_body}
              </tbody>
            </table>
            """
        else:
            episodes_table_html = """
            <div class="no-episodes">
              No completed episodes were recorded for this run.
            </div>
            """

    html_str = f"""
<style>
{_CSS}
</style>

<div class="gym_info-report">
  <div class="report-header">
    <div class="report-title">gym_info entropy report</div>
    <div class="report-meta">
      <span class="meta-item">
        <span class="meta-label">env_id:</span> {env_id}
      </span>
      <span class="meta-item">
        <span class="meta-label">run_id:</span> {run_id}
      </span>
      <span class="meta-item">
        <span class="meta-label">episodes:</span> {num_episodes}
      </span>
    </div>
  </div>

  <div class="report-section">
    <div class="section-title">Global entropies (bits)</div>
    {global_table_html}
  </div>

  <div class="report-section">
    <div class="section-title">Per-episode entropies (bits)</div>
    {episodes_table_html}
  </div>
</div>
"""
    return html_str
