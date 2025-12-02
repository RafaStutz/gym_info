from __future__ import annotations

import html

from .models import EntropyReport

_CSS = """
.gym-info-report {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 14px;
  line-height: 1.4;
  color: #222;
  max-width: 720px;
}

.gym-info-report .report-header {
  border-bottom: 1px solid #ddd;
  padding-bottom: 8px;
  margin-bottom: 12px;
}

.gym-info-report .report-title {
  font-weight: 600;
  font-size: 16px;
  margin-bottom: 4px;
}

.gym-info-report .report-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  font-size: 13px;
  color: #555;
}

.gym-info-report .report-meta .meta-label {
  font-weight: 600;
}

.gym-info-report .report-section {
  margin-top: 12px;
  margin-bottom: 12px;
}

.gym-info-report .section-title {
  font-weight: 600;
  margin-bottom: 4px;
}

.gym-info-report .entropy-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.gym-info-report .entropy-table th,
.gym-info-report .entropy-table td {
  border: 1px solid #ddd;
  padding: 4px 6px;
  text-align: right;
}

.gym-info-report .entropy-table th:first-child,
.gym-info-report .entropy-table td:first-child {
  text-align: left;
}

.gym-info-report .entropy-table thead {
  background-color: #f5f5f5;
}

.gym-info-report .entropy-table tbody tr:nth-child(odd) {
  background-color: #fafafa;
}

.gym-info-report .entropy-table tbody tr:nth-child(even) {
  background-color: #ffffff;
}

.gym-info-report .no-episodes {
  font-size: 13px;
  color: #666;
  margin-top: 4px;
  font-style: italic;
}
"""


def render_entropy_report_html(rep: EntropyReport) -> str:
    """
    Render an EntropyReport as an HTML snippet with inline CSS.

    Intended for notebook environments (Jupyter, Colab, etc.). The
    returned string can be passed directly to IPython.display.HTML.
    """
    env_id = html.escape(rep.env_id)
    run_id = html.escape(rep.run_id) if rep.run_id is not None else ""
    num_episodes = rep.num_episodes

    ge = rep.global_entropies

    episode_rows: list[str] = []
    for idx, ent in enumerate(rep.episode_entropies):
        episode_rows.append(
            (
                "<tr>"
                f"<td class='epi-index'>{idx}</td>"
                f"<td class='epi-value'>{ent.H_S:.6f}</td>"
                f"<td class='epi-value'>{ent.H_A:.6f}</td>"
                f"<td class='epi-value'>{ent.H_A_given_S:.6f}</td>"
                "</tr>"
            )
        )

    if episode_rows:
        episodes_table_body = "\n".join(episode_rows)
        episodes_table_html = f"""
        <table class="entropy-table entropy-table-episodes">
          <thead>
            <tr>
              <th>Episode</th>
              <th>H(S)</th>
              <th>H(A)</th>
              <th>H(A | S)</th>
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

<div class="gym-info-report">
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
    <table class="entropy-table entropy-table-global">
      <thead>
        <tr>
          <th></th>
          <th>H(S)</th>
          <th>H(A)</th>
          <th>H(A | S)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Run</td>
          <td>{ge.H_S:.6f}</td>
          <td>{ge.H_A:.6f}</td>
          <td>{ge.H_A_given_S:.6f}</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="report-section">
    <div class="section-title">Per-episode entropies (bits)</div>
    {episodes_table_html}
  </div>
</div>
"""
    return html_str
