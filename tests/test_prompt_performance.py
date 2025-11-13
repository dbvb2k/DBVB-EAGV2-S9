"""Prompt performance comparison harness for conservative decision prompt."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import hashlib
import math
import statistics
import unittest
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

if __package__ in {None, ""}:
    CURRENT_DIR = Path(__file__).resolve().parent
    ROOT_DIR = CURRENT_DIR.parent
    if str(ROOT_DIR) not in os.sys.path:
        os.sys.path.insert(0, str(ROOT_DIR))

from heuristics_lib import heuristics_validators as hv

ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "prompts"
ORIGINAL_PROMPT_PATH = PROMPT_DIR / "decision_prompt_conservative.txt"
# OPT_PROMPT_PATH = PROMPT_DIR / "decision_prompt_conservative_opt1.txt"
OPT_PROMPT_PATH = PROMPT_DIR / "decision_prompt_conservative_opt2.txt"
BENCHMARK_DATA_PATH = ROOT / "tests" / "prompt_benchmark_samples.json"

REQUIRED_PHRASES = [
    "Plan exactly ONE FUNCTION_CALL",
    "`async def solve():`",
    "await mcp.call_tool",
    "tool docstring",
    "tool name string",
    "json.loads",
    "FINAL_ANSWER",
    "FURTHER_PROCESSING_REQUIRED",
]

SAFETY_RED_FLAGS = [
    "ignore previous instructions",
    "override system",
    "shutdown",
    "format disk",
]

SCENARIOS = [
    {
        "name": "search_summary",
        "tool_descriptions": "- search: Usage docstring...\n- summarize: Usage docstring...",
        "user_input": "Summarize the latest trends in AI hardware within 3 bullet points.",
    },
    {
        "name": "numeric_tool",
        "tool_descriptions": "- add_numbers: Adds two ints.\n- multiply_numbers: Multiplies two ints.",
        "user_input": "Compute (42 + 58) * 3 using available tools.",
    },
    {
        "name": "document_pipeline",
        "tool_descriptions": "- extract_webpage: Fetches clean HTML text.\n- search_docs: Retrieves indexed passages.",
        "user_input": "Find financial highlights for DLF from stored documents and cite sources.",
    },
    {
        "name": "web_markdown_conversion",
        "tool_descriptions": "- convert_webpage_url_into_markdown: Cleans and converts webpages into markdown.",
        "user_input": "Fetch the markdown for DeepMind's latest blog spotlight.",
    },
    {
        "name": "duckduckgo_news_search",
        "tool_descriptions": "- duckduckgo_search_results: Retrieves DuckDuckGo results.",
        "user_input": "Get three recent stories about AI regulation developments.",
    },
]

TOOL_CALL_PATTERN = re.compile(r"await\s+mcp\.call_tool\(\s*['\"]([^'\"]+)['\"]")
RETURN_PREFIX_PATTERN = re.compile(r"return\s+f?['\"](?P<prefix>[A-Z_]+):")


@dataclass(frozen=True)
class PromptMetrics:
    scenario: str
    rendered_prompt: str
    word_count: int
    normalized_word_count: int
    rule_coverage: Dict[str, bool]
    missing_rules: List[str]
    code_fence_count: int
    has_required_returns: bool
    safety_flags: List[str]
    injection_validation: hv.ValidationResult
    composite_score: float
    hash_fingerprint: str


@dataclass(frozen=True)
class PromptRunRecord:
    output: str
    success: bool
    human_score: float


@dataclass(frozen=True)
class PromptEvaluationSummary:
    runs: Sequence[PromptRunRecord]
    schema_conform_rate: float
    function_call_rate: float
    success_rate: float
    avg_human_score: float
    stability_score: float
    tool_calls: List[str]
    concatenated_output: str


@dataclass(frozen=True)
class ScenarioComparison:
    name: str
    allowed_tools: Sequence[str]
    return_prefixes: Sequence[str]
    original: PromptEvaluationSummary
    optimized: PromptEvaluationSummary
    tool_selection_match_rate: float
    embedding_similarity: float
    original_prompt: str
    optimized_prompt: str


def load_benchmark_dataset(path: Path = BENCHMARK_DATA_PATH) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("scenarios", [])


def parse_runs(raw_runs: Sequence[dict]) -> List[PromptRunRecord]:
    records: List[PromptRunRecord] = []
    for item in raw_runs:
        records.append(
            PromptRunRecord(
                output=item["output"],
                success=bool(item.get("success", False)),
                human_score=float(item.get("human_score", 0.0)),
            )
        )
    return records


def extract_tool_calls(output: str) -> List[str]:
    return TOOL_CALL_PATTERN.findall(output)


def extract_return_prefix(output: str) -> str | None:
    match = RETURN_PREFIX_PATTERN.search(output)
    if not match:
        return None
    return match.group("prefix")


def vectorize_text(text: str) -> Dict[str, float]:
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    counter = Counter(tokens)
    return {token: float(count) for token, count in counter.items()}


def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    intersection = set(vec_a) & set(vec_b)
    numerator = sum(vec_a[token] * vec_b[token] for token in intersection)
    denom_a = math.sqrt(sum(value ** 2 for value in vec_a.values()))
    denom_b = math.sqrt(sum(value ** 2 for value in vec_b.values()))
    if math.isclose(denom_a, 0.0) or math.isclose(denom_b, 0.0):
        return 0.0
    return numerator / (denom_a * denom_b)


def compute_stability_score(outputs: Sequence[str]) -> float:
    if not outputs:
        return 1.0
    frequency = Counter(outputs)
    dominant = max(frequency.values())
    return dominant / len(outputs)


def compute_tool_selection_match_rate(original_calls: Sequence[str], optimized_calls: Sequence[str]) -> float:
    pairs = list(zip(original_calls, optimized_calls))
    if not pairs:
        return 1.0
    matches = sum(1 for original, optimized in pairs if original == optimized)
    return matches / len(pairs)


def compute_prompt_evaluation(
    runs: Sequence[PromptRunRecord],
    allowed_tools: Sequence[str],
    schema_regex: str,
    return_prefixes: Sequence[str],
) -> PromptEvaluationSummary:
    if not runs:
        return PromptEvaluationSummary(
            runs=[],
            schema_conform_rate=1.0,
            function_call_rate=1.0,
            success_rate=1.0,
            avg_human_score=0.0,
            stability_score=1.0,
            tool_calls=[],
            concatenated_output="",
        )

    schema_hits = 0
    function_hits = 0
    success_hits = 0
    tool_calls: List[str] = []
    human_scores: List[float] = []
    outputs: List[str] = []

    for record in runs:
        output = record.output
        outputs.append(output)
        tools = extract_tool_calls(output)
        primary_tool = tools[0] if tools else ""
        tool_calls.append(primary_tool)

        schema_ok = bool(re.search(schema_regex, output)) and (
            extract_return_prefix(output) in return_prefixes
        )
        if schema_ok:
            schema_hits += 1

        function_ok = len(tools) == 1 and all(tool in allowed_tools for tool in tools)
        if function_ok:
            function_hits += 1

        if record.success:
            success_hits += 1
        human_scores.append(record.human_score)

    concatenated = "\n".join(outputs)
    schema_rate = schema_hits / len(runs)
    function_rate = function_hits / len(runs)
    success_rate = success_hits / len(runs)
    avg_human = statistics.mean(human_scores) if human_scores else 0.0
    stability = compute_stability_score(outputs)

    return PromptEvaluationSummary(
        runs=runs,
        schema_conform_rate=schema_rate,
        function_call_rate=function_rate,
        success_rate=success_rate,
        avg_human_score=avg_human,
        stability_score=stability,
        tool_calls=tool_calls,
        concatenated_output=concatenated,
    )


def build_scenario_comparisons(
    original_metrics: Sequence[PromptMetrics],
    optimized_metrics: Sequence[PromptMetrics],
) -> List[ScenarioComparison]:
    metrics_by_name_original = {metric.scenario: metric for metric in original_metrics}
    metrics_by_name_optimized = {metric.scenario: metric for metric in optimized_metrics}

    comparisons: List[ScenarioComparison] = []
    for scenario_data in load_benchmark_dataset():
        name = scenario_data["name"]
        allowed_tools = scenario_data["allowed_tools"]
        schema_regex = scenario_data.get("schema_regex", r"async\s+def\s+solve")
        return_prefixes = scenario_data.get(
            "return_prefixes",
            ["FINAL_ANSWER", "FURTHER_PROCESSING_REQUIRED"],
        )
        original_runs = parse_runs(scenario_data["original"]["runs"])
        optimized_runs = parse_runs(scenario_data["optimized"]["runs"])

        original_eval = compute_prompt_evaluation(original_runs, allowed_tools, schema_regex, return_prefixes)
        optimized_eval = compute_prompt_evaluation(optimized_runs, allowed_tools, schema_regex, return_prefixes)

        match_rate = compute_tool_selection_match_rate(original_eval.tool_calls, optimized_eval.tool_calls)
        embedding_sim = cosine_similarity(
            vectorize_text(original_eval.concatenated_output),
            vectorize_text(optimized_eval.concatenated_output),
        )

        original_prompt = metrics_by_name_original[name].rendered_prompt
        optimized_prompt = metrics_by_name_optimized[name].rendered_prompt

        comparisons.append(
            ScenarioComparison(
                name=name,
                allowed_tools=allowed_tools,
                return_prefixes=return_prefixes,
                original=original_eval,
                optimized=optimized_eval,
                tool_selection_match_rate=match_rate,
                embedding_similarity=embedding_sim,
                original_prompt=original_prompt,
                optimized_prompt=optimized_prompt,
            )
        )
    return comparisons


def run_llm_judge(command: str, comparisons: Sequence[ScenarioComparison]) -> None:
    print("\n=== LLM JUDGE RESULTS ===")
    for scenario in comparisons:
        payload = json.dumps(
            {
                "scenario": scenario.name,
                "original_prompt": scenario.original_prompt,
                "optimized_prompt": scenario.optimized_prompt,
                "original_outputs": [run.output for run in scenario.original.runs],
                "optimized_outputs": [run.output for run in scenario.optimized.runs],
            },
            ensure_ascii=False,
        )
        try:
            completed = subprocess.run(
                command,
                input=payload.encode("utf-8"),
                capture_output=True,
                shell=True,
                check=False,
            )
            stdout = completed.stdout.decode("utf-8", errors="replace").strip()
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            print(f"- {scenario.name}: returncode={completed.returncode}")
            if stdout:
                print(f"  stdout: {stdout}")
            if stderr:
                print(f"  stderr: {stderr}")
        except OSError as exc:
            print(f"- {scenario.name}: failed to invoke judge command ({exc})")


def render_prompt(path: Path, *,
                  tool_descriptions: str,
                  user_input: str) -> str:
    namespace: Dict[str, str] = {
        "tool_descriptions": tool_descriptions,
        "user_input": user_input,
    }
    exec(path.read_text(encoding="utf-8"), namespace)  # noqa: S102
    return namespace["prompt"]


def _normalized_word_count(prompt_text: str, *, tool_descriptions: str, user_input: str) -> int:
    words = prompt_text.split()
    tool_words = tool_descriptions.split()
    input_words = user_input.split()
    return max(len(words) - len(tool_words) - len(input_words), 0)


def collect_metrics(prompt_text: str, scenario_name: str, *, tool_descriptions: str, user_input: str) -> PromptMetrics:
    lowered = prompt_text.lower()
    rule_presence = {rule: (rule.lower() in lowered) for rule in REQUIRED_PHRASES}
    missing = [rule for rule, present in rule_presence.items() if not present]
    fences = prompt_text.count("```")
    code_blocks = fences // 2
    has_returns = "FINAL_ANSWER" in prompt_text and "FURTHER_PROCESSING_REQUIRED" in prompt_text
    safety_hits = [flag for flag in SAFETY_RED_FLAGS if flag in lowered]
    injection_check = hv.detect_prompt_injection(prompt_text, max_length=8000)
    coverage_ratio = sum(rule_presence.values()) / len(REQUIRED_PHRASES)
    example_score = min(code_blocks, 2) / 2  # up to 2 relevant examples
    return_score = 1.0 if has_returns else 0.0
    safety_score = 1.0 if not safety_hits and injection_check.is_valid else 0.0
    composite = 0.45 * coverage_ratio + 0.2 * example_score + 0.2 * return_score + 0.15 * safety_score
    fingerprint = hashlib.sha256(" ".join(sorted(lowered.split())).encode("utf-8")).hexdigest()

    return PromptMetrics(
        scenario=scenario_name,
        rendered_prompt=prompt_text,
        word_count=len(prompt_text.split()),
        normalized_word_count=_normalized_word_count(
            prompt_text,
            tool_descriptions=tool_descriptions,
            user_input=user_input,
        ),
        rule_coverage=rule_presence,
        missing_rules=missing,
        code_fence_count=code_blocks,
        has_required_returns=has_returns,
        safety_flags=safety_hits,
        injection_validation=injection_check,
        composite_score=composite,
        hash_fingerprint=fingerprint,
    )


def evaluate_prompt(path: Path) -> List[PromptMetrics]:
    metrics: List[PromptMetrics] = []
    for scenario in SCENARIOS:
        rendered = render_prompt(path, tool_descriptions=scenario["tool_descriptions"], user_input=scenario["user_input"])
        metrics.append(
            collect_metrics(
                rendered,
                scenario_name=scenario["name"],
                tool_descriptions=scenario["tool_descriptions"],
                user_input=scenario["user_input"],
            )
        )
    return metrics


def aggregate_scores(results: Iterable[PromptMetrics]) -> float:
    return statistics.mean(metric.composite_score for metric in results)


def paired_t_stat(original: Iterable[PromptMetrics], optimized: Iterable[PromptMetrics]) -> float:
    diffs = [
        o.composite_score - opt.composite_score
        for o, opt in zip(original, optimized)
    ]
    if not diffs:
        return 0.0
    mean_diff = statistics.mean(diffs)
    if math.isclose(mean_diff, 0.0, abs_tol=1e-9):
        return 0.0
    std_dev = statistics.pstdev(diffs)
    if math.isclose(std_dev, 0.0, abs_tol=1e-9):
        return 0.0
    return (mean_diff / (std_dev / math.sqrt(len(diffs))))


class PromptParityTests:
    """Reusable test mixin to share evaluation logic."""

    @classmethod
    def setUpClass(cls) -> None:
        print(f"[PromptComparison] Original prompt file: {ORIGINAL_PROMPT_PATH}")
        print(f"[PromptComparison] Optimized prompt file: {OPT_PROMPT_PATH}")
        cls.original_metrics = evaluate_prompt(ORIGINAL_PROMPT_PATH)
        cls.optimized_metrics = evaluate_prompt(OPT_PROMPT_PATH)
        cls.scenario_comparisons = build_scenario_comparisons(cls.original_metrics, cls.optimized_metrics)


class TestPromptComparison(PromptParityTests, unittest.TestCase):
    """Unit tests ensuring optimized prompt parity with original."""

    def test_optimized_prompt_word_budget(self) -> None:
        for metric in self.optimized_metrics:
            self.assertLessEqual(metric.normalized_word_count, 300, f"{metric.scenario} exceeds word budget")

    def test_rule_coverage_matches_original(self) -> None:
        for original, optimized in zip(self.original_metrics, self.optimized_metrics):
            self.assertEqual(
                original.missing_rules,
                optimized.missing_rules,
                f"Rule coverage mismatch in scenario '{original.scenario}'",
            )

    def test_composite_scores_parity(self) -> None:
        original_score = aggregate_scores(self.original_metrics)
        optimized_score = aggregate_scores(self.optimized_metrics)
        self.assertTrue(
            optimized_score >= original_score - 0.02,
            f"Optimized composite score {optimized_score:.3f} trails original {original_score:.3f} beyond tolerance.",
        )

    def test_paired_statistic_indicates_no_regression(self) -> None:
        t_value = paired_t_stat(self.original_metrics, self.optimized_metrics)
        self.assertLess(abs(t_value), 1.5, f"Paired t-statistic indicates significant regression: {t_value:.3f}")

    def test_safety_and_guardrail_checks(self) -> None:
        for metric in (*self.original_metrics, *self.optimized_metrics):
            self.assertTrue(metric.injection_validation.is_valid, f"Injection guard failed for {metric.scenario}")
            self.assertFalse(metric.safety_flags, f"Safety flags present for {metric.scenario}")

    def test_example_and_structure_consistency(self) -> None:
        for metric in self.optimized_metrics:
            self.assertGreaterEqual(metric.code_fence_count, 2, "Optimized prompt should retain at least two examples.")
            self.assertTrue(metric.has_required_returns, "Missing FINAL_ANSWER / FURTHER_PROCESSING_REQUIRED guidance.")

    def test_regression_guard_keyword_overlap(self) -> None:
        def bigram_set(text: str) -> set[str]:
            tokens = text.lower().split()
            return {" ".join(pair) for pair in zip(tokens, tokens[1:])}

        original_bigrams = set.union(*(bigram_set(metric.rendered_prompt) for metric in self.original_metrics))
        optimized_bigrams = set.union(*(bigram_set(metric.rendered_prompt) for metric in self.optimized_metrics))
        union = original_bigrams | optimized_bigrams
        jaccard = len(original_bigrams & optimized_bigrams) / len(union) if union else 1.0
        self.assertGreaterEqual(jaccard, 0.20, f"Semantic overlap too low (Jaccard={jaccard:.2f})")

    def test_schema_conformity_rate(self) -> None:
        for comparison in self.scenario_comparisons:
            self.assertGreaterEqual(
                comparison.original.schema_conform_rate,
                0.95,
                f"Original prompt schema conformity low for {comparison.name}",
            )
            self.assertGreaterEqual(
                comparison.optimized.schema_conform_rate,
                0.95,
                f"Optimized prompt schema conformity low for {comparison.name}",
            )

    def test_function_call_correctness_rate(self) -> None:
        for comparison in self.scenario_comparisons:
            self.assertGreaterEqual(
                comparison.original.function_call_rate,
                0.95,
                f"Original prompt function-call correctness low for {comparison.name}",
            )
            self.assertGreaterEqual(
                comparison.optimized.function_call_rate,
                0.95,
                f"Optimized prompt function-call correctness low for {comparison.name}",
            )

    def test_tool_selection_match_rate(self) -> None:
        for comparison in self.scenario_comparisons:
            self.assertGreaterEqual(
                comparison.tool_selection_match_rate,
                0.95,
                f"Tool selection diverged for {comparison.name}",
            )

    def test_task_success_rate(self) -> None:
        for comparison in self.scenario_comparisons:
            self.assertGreaterEqual(
                comparison.optimized.success_rate,
                comparison.original.success_rate - 0.05,
                f"Task success regression for {comparison.name}",
            )

    def test_stability_scores(self) -> None:
        for comparison in self.scenario_comparisons:
            self.assertGreaterEqual(
                comparison.optimized.stability_score,
                0.95,
                f"Optimized prompt instability detected in {comparison.name}",
            )

    def test_embedding_similarity_threshold(self) -> None:
        for comparison in self.scenario_comparisons:
            self.assertGreaterEqual(
                comparison.embedding_similarity,
                0.85,
                f"Embedding similarity below threshold for {comparison.name}",
            )

    def test_human_score_delta_within_margin(self) -> None:
        for comparison in self.scenario_comparisons:
            delta = abs(comparison.original.avg_human_score - comparison.optimized.avg_human_score)
            self.assertLessEqual(
                delta,
                0.5,
                f"Human evaluation score delta too high for {comparison.name}",
            )


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Compare conservative decision prompts.")
    parser.add_argument(
        "--prompt",
        choices=["original", "optimized", "both"],
        default="both",
        help="Select which prompt(s) to evaluate and display metrics for.",
    )
    parser.add_argument(
        "--llm-judge-command",
        help="Optional shell command that receives scenario data on stdin and scores prompt outputs.",
    )
    args = parser.parse_args()

    original_metrics = evaluate_prompt(ORIGINAL_PROMPT_PATH)
    optimized_metrics = evaluate_prompt(OPT_PROMPT_PATH)
    comparisons = build_scenario_comparisons(original_metrics, optimized_metrics)

    selected = []
    if args.prompt in {"original", "both"}:
        selected.append(("original", ORIGINAL_PROMPT_PATH, original_metrics))
    if args.prompt in {"optimized", "both"}:
        selected.append(("optimized", OPT_PROMPT_PATH, optimized_metrics))

    for label, prompt_path, metrics in selected:
        print(f"\n=== {label.upper()} PROMPT METRICS ===")
        print(f"Prompt file: {prompt_path}")
        composite = aggregate_scores(metrics)
        print(f"Composite score: {composite:.3f}")
        for metric in metrics:
            print(
                f"- {metric.scenario}: words={metric.normalized_word_count}, "
                f"rules={sum(metric.rule_coverage.values())}/{len(metric.rule_coverage)}, "
                f"examples={metric.code_fence_count}, score={metric.composite_score:.3f}"
            )

    print("\n=== SCENARIO COMPARISON SUMMARY ===")
    for comparison in comparisons:
        print(
            f"- {comparison.name}: tool_match={comparison.tool_selection_match_rate:.2f}, "
            f"embedding_similarity={comparison.embedding_similarity:.2f}"
        )
        print(
            f"  Original => schema={comparison.original.schema_conform_rate:.2f}, "
            f"function_call={comparison.original.function_call_rate:.2f}, "
            f"success={comparison.original.success_rate:.2f}, "
            f"stability={comparison.original.stability_score:.2f}, "
            f"human_score={comparison.original.avg_human_score:.2f}"
        )
        print(
            f"  Optimized => schema={comparison.optimized.schema_conform_rate:.2f}, "
            f"function_call={comparison.optimized.function_call_rate:.2f}, "
            f"success={comparison.optimized.success_rate:.2f}, "
            f"stability={comparison.optimized.stability_score:.2f}, "
            f"human_score={comparison.optimized.avg_human_score:.2f}"
        )

    if args.llm_judge_command:
        run_llm_judge(args.llm_judge_command, comparisons)


if __name__ == "__main__":
    run_cli()


