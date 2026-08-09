#!/usr/bin/env python3
"""Compile scored case studies into an interactive quality-and-cost comparison."""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RADAR_AXES = (
    ("argument", "Argument", (("index.md", "argument"), ("INSTRUCTIONS.md", "argument"))),
    ("grounding", "Grounding", (("index.md", "grounding"),)),
    ("action", "Action", (("index.md", "action"),)),
    ("framework", "Framework", (("INSTRUCTIONS.md", "framework"),)),
    ("readability", "Readability", (("index.md", "readability"), ("INSTRUCTIONS.md", "readability"))),
    ("voice", "Voice & closing", (("index.md", "closure"), ("INSTRUCTIONS.md", "voice"))),
    ("cost_efficiency", "Cost efficiency", ()),
)

COLORS = ("#38bdf8", "#f97316", "#a78bfa", "#34d399", "#facc15", "#fb7185")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def money(value: float | None) -> str:
    return "Unpriced local run" if value is None else "$" + f"{value:.6f}"


def company_for(model: str) -> str:
    return {"openai": "OpenAI", "google": "Google", "deepseek": "DeepSeek"}.get(model.split("/", 1)[0], "Local / other")


def display_name(model: str) -> str:
    return model.split("/", 1)[-1].replace("-it", "").replace("-", " ").title().replace("Gpt", "GPT")


def paid_cost(usage_path: Path) -> float | None:
    if not usage_path.exists():
        return None
    total = 0.0
    for line in usage_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        usage = record.get("usage", {})
        if isinstance(usage, dict):
            total += float(usage.get("cost", 0.0) or 0.0)
    return total


def rubric_category_scores(report: dict[str, Any], rubric: dict[str, Any]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for artifact, artifact_score in report["artifacts"].items():
        rubric_criteria = {item["id"]: item["category"] for item in rubric["artifacts"][artifact]["criteria"]}
        grouped: dict[str, list[float]] = defaultdict(list)
        for criterion in artifact_score["criteria"]:
            category = rubric_criteria.get(criterion["id"])
            if category:
                grouped[category].append(float(criterion["score"]))
        output[artifact] = {category: round(sum(scores) / len(scores), 2) for category, scores in grouped.items()}
    return output


def axis_scores(categories: dict[str, dict[str, float]]) -> dict[str, float | None]:
    axes: dict[str, float | None] = {}
    for axis_id, _label, sources in RADAR_AXES:
        if not sources:
            axes[axis_id] = None
            continue
        values = [categories.get(artifact, {}).get(category) for artifact, category in sources]
        present = [value for value in values if value is not None]
        axes[axis_id] = round(sum(present) / len(present), 2) if present else None
    return axes


def compile_comparison(root: Path) -> dict[str, Any]:
    rubric = load_json(root.parent / "content_scoring" / "rubrics" / "steadyburn-v1.json")
    models, awaiting_score = [], []
    for comparison_path in sorted(root.rglob("MODEL_COMPARISON.md")):
        study_root = comparison_path.parent
        score_path = study_root / "CONTENT_SCORE.json"
        if not score_path.exists():
            awaiting_score.append(study_root.relative_to(root).as_posix())
            continue
        report = load_json(score_path)
        model = str(report.get("model", "unknown"))
        categories = rubric_category_scores(report, rubric)
        models.append({
            "model": model,
            "display_name": display_name(model),
            "company": company_for(model),
            "case_study": study_root.relative_to(root).as_posix(),
            "content_score": float(report["content_score"]),
            "confidence": float(report["confidence"]),
            "cost_usd": paid_cost(study_root / "OPENROUTER_USAGE.jsonl"),
            "categories": categories,
            "criteria": {artifact: {item["id"]: float(item["score"]) for item in value["criteria"]} for artifact, value in report["artifacts"].items()},
            "measurements": {artifact: value["metrics"] for artifact, value in report["artifacts"].items()},
            "radar": axis_scores(categories),
        })
    prices = [item["cost_usd"] for item in models if item["cost_usd"] is not None and item["cost_usd"] > 0]
    cheapest = min(prices) if prices else None
    for model in models:
        cost = model["cost_usd"]
        model["radar"]["cost_efficiency"] = round(100 * cheapest / cost, 2) if cheapest and cost else None
    return {
        "schema_version": "model-comparison-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rubric_version": rubric["version"],
        "radar_axes": [{"id": axis_id, "label": label} for axis_id, label, _sources in RADAR_AXES],
        "models": sorted(models, key=lambda item: item["display_name"].lower()),
        "awaiting_score": awaiting_score,
    }


def dashboard_html(data: dict[str, Any]) -> str:
    embedded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    title = "SteadyBurn model comparison"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<style>
body{{margin:0;background:#0b1020;color:#e5e7eb;font:16px system-ui,sans-serif}}main{{max-width:1200px;margin:auto;padding:32px}}h1{{margin-bottom:6px}}.muted{{color:#a5b4c7}}#controls{{display:flex;flex-wrap:wrap;gap:10px;margin:24px 0}}label{{background:#18243a;border-radius:999px;padding:8px 12px;cursor:pointer}}input{{margin-right:6px}}svg{{width:100%;max-width:760px;background:#111a2d;border-radius:16px}}.axis{{stroke:#41506d;fill:none}}.label{{fill:#cbd5e1;font-size:12px}}.legend{{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0}}.swatch{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}}table{{width:100%;border-collapse:collapse;margin-top:30px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #24324b}}th{{color:#a5b4c7}}code{{font-size:12px}}.warning{{padding:12px;background:#3a2b12;border-radius:8px}}
</style></head><body><main><h1>{html.escape(title)}</h1><p class="muted">Quality axes are 0–100 rubric measurements. Cost efficiency is relative to the cheapest scored paid run; raw recorded bundle cost remains visible below.</p>
<div id="controls"></div><div id="legend" class="legend"></div><svg id="radar" viewBox="0 0 760 620" role="img" aria-label="Model quality and cost radar chart"></svg><div id="table"></div><div id="awaiting"></div>
<script>const comparison={embedded};
const axes=comparison.radar_axes, models=comparison.models, selected=new Set(models.slice(0,5).map(m=>m.model));
const colors={json.dumps(COLORS)}; const svg=d3.select('#radar'), cx=380, cy=310, radius=220;
function point(i,value){{const angle=2*Math.PI*i/axes.length-Math.PI/2, r=radius*value/100;return [cx+r*Math.cos(angle),cy+r*Math.sin(angle)]}}
function draw(){{const visible=models.filter(m=>selected.has(m.model));svg.selectAll('*').remove();for(let ring=1;ring<=5;ring++){{const v=ring*20;svg.append('path').attr('class','axis').attr('d','M'+axes.map((_,i)=>point(i,v).join(',')).join('L')+'Z')}}axes.forEach((axis,i)=>{{const end=point(i,100);svg.append('line').attr('class','axis').attr('x1',cx).attr('y1',cy).attr('x2',end[0]).attr('y2',end[1]);const label=point(i,114);svg.append('text').attr('class','label').attr('x',label[0]).attr('y',label[1]).attr('text-anchor','middle').text(axis.label)}});visible.forEach((model,index)=>{{const values=axes.map(axis=>model.radar[axis.id] ?? 0);const d='M'+values.map((v,i)=>point(i,v).join(',')).join('L')+'Z';svg.append('path').attr('d',d).attr('fill',colors[index%colors.length]).attr('fill-opacity',.16).attr('stroke',colors[index%colors.length]).attr('stroke-width',2.5) }});d3.select('#legend').html(visible.map((m,i)=>`<span><i class="swatch" style="background:${{colors[i%colors.length]}}"></i>${{m.display_name}}</span>`).join(''));const rows=visible.map(m=>`<tr><td>${{m.display_name}}</td><td>${{m.company}}</td><td>${{m.content_score.toFixed(2)}}</td><td>${{m.cost_usd===null?'Unpriced local run':'$'+m.cost_usd.toFixed(6)}}</td><td>${{m.confidence.toFixed(3)}}</td></tr>`).join('');d3.select('#table').html(`<table><thead><tr><th>Model</th><th>Provider</th><th>Content score</th><th>Bundle cost</th><th>Confidence</th></tr></thead><tbody>${{rows}}</tbody></table>`);}}
d3.select('#controls').selectAll('label').data(models).join('label').html((m,i)=>`<input type="checkbox" ${{selected.has(m.model)?'checked':''}}> ${{m.display_name}}`).on('change',(event,m)=>{{event.target.checked?selected.add(m.model):selected.delete(m.model);draw()}});if(comparison.awaiting_score.length)d3.select('#awaiting').html(`<p class="warning">Awaiting content score: ${{comparison.awaiting_score.length}} model run(s). They are excluded from the radar until their standard suite completes.</p>`);draw();
</script></main></body></html>"""


def radar_svg(data: dict[str, Any]) -> str:
    """Render a Markdown-embeddable snapshot of the interactive radar chart."""
    axes = data["radar_axes"]
    models = [
        model for model in sorted(data["models"], key=lambda item: item["content_score"], reverse=True)
        if all(model["radar"].get(axis["id"]) is not None for axis in axes)
    ][:6]
    width, height, center_x, center_y, radius = 1100, 760, 430, 390, 255

    def point(index: int, value: float) -> tuple[float, float]:
        angle = 2 * math.pi * index / len(axes) - math.pi / 2
        distance = radius * value / 100
        return center_x + distance * math.cos(angle), center_y + distance * math.sin(angle)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        '  <title id="title">SteadyBurn model quality and cost comparison</title>',
        '  <desc id="description">A radar chart summarizing the scored, priced case studies. The interactive dashboard contains the complete comparison.</desc>',
        '  <style>.bg{fill:#0b1020}.title{fill:#f8fafc;font:700 28px Arial,sans-serif}.sub,.axis-label,.note{fill:#a5b4c7;font:15px Arial,sans-serif}.grid{fill:none;stroke:#41506d;stroke-width:1}.spoke{stroke:#41506d;stroke-width:1}.legend{fill:#e2e8f0;font:16px Arial,sans-serif}</style>',
        f'  <rect class="bg" width="{width}" height="{height}" rx="18"/>',
        '  <text class="title" x="48" y="58">SteadyBurn model quality and cost</text>',
        '  <text class="sub" x="48" y="86">0–100 rubric measurements; cost efficiency is relative to the least-expensive scored paid run.</text>',
    ]
    for ring in range(1, 6):
        points = " ".join(f"{x:.1f},{y:.1f}" for index in range(len(axes)) for x, y in [point(index, ring * 20)])
        lines.append(f'  <polygon class="grid" points="{points}"/>')
    for index, axis in enumerate(axes):
        x, y = point(index, 100)
        label_x, label_y = point(index, 115)
        lines.extend((
            f'  <line class="spoke" x1="{center_x}" y1="{center_y}" x2="{x:.1f}" y2="{y:.1f}"/>',
            f'  <text class="axis-label" x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="middle">{html.escape(axis["label"])}</text>',
        ))
    for index, model in enumerate(models):
        color = COLORS[index % len(COLORS)]
        points = " ".join(
            f"{x:.1f},{y:.1f}" for axis_index, axis in enumerate(axes)
            for x, y in [point(axis_index, float(model["radar"][axis["id"]]))]
        )
        legend_y = 190 + index * 42
        lines.extend((
            f'  <polygon points="{points}" fill="{color}" fill-opacity="0.16" stroke="{color}" stroke-width="2.5"/>',
            f'  <circle cx="760" cy="{legend_y - 5}" r="6" fill="{color}"/>',
            f'  <text class="legend" x="776" y="{legend_y}">{html.escape(model["display_name"])} — {money(model["cost_usd"])}</text>',
        ))
    if not models:
        lines.append('  <text class="note" x="48" y="150">No scored paid case studies are available yet.</text>')
    lines.extend((
        '  <text class="note" x="48" y="708">Local runs remain unpriced and are available in the dashboard table, but cannot receive a cost-efficiency axis.</text>',
        '  <text class="note" x="48" y="734">Source: CONTENT_SCORE.json and OPENROUTER_USAGE.jsonl in each case-study directory.</text>',
        '</svg>',
    ))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    data = compile_comparison(root)
    (root / "model-comparison.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (root / "model-comparison.html").write_text(dashboard_html(data), encoding="utf-8")
    (root / "model-comparison-radar.svg").write_text(radar_svg(data), encoding="utf-8")
    print(f"Compiled {len(data['models'])} scored model runs; {len(data['awaiting_score'])} await scoring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
