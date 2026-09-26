import logging
from typing import Any, Callable, Iterable, Mapping

from .config import EvaluatorNode
from .context import context_for_node
from .result_store import ResultStore, ResultView
from .result import EvaluationResult


logger = logging.getLogger(__name__)


NodeExecutor = Callable[[EvaluatorNode, ResultView], EvaluationResult]
NodeReporter = Callable[[EvaluatorNode, EvaluationResult], Iterable[Any]]


class EvaluationEngine:
    def __init__(
        self,
        nodes: Iterable[EvaluatorNode],
        result_store: ResultStore,
        executor: NodeExecutor,
        overwrite: bool = False,
        context: Mapping[str, Any] | None = None,
        reporter: NodeReporter | None = None,
        generate_reports: bool = True,
        report_only: bool = False,
    ):
        self.nodes = list(nodes)
        self.result_store = result_store
        self.executor = executor
        self.overwrite = overwrite
        self.context = dict(context or {})
        self.reporter = reporter
        self.generate_reports = bool(generate_reports)
        self.report_only = bool(report_only)
        self._nodes_by_id = self._validate_nodes(self.nodes)

    def execution_order(self) -> list[EvaluatorNode]:
        indegree = {node.node_id: 0 for node in self.nodes}
        dependents = {node.node_id: [] for node in self.nodes}
        for node in self.nodes:
            for dependency in node.depends_on:
                indegree[node.node_id] += 1
                dependents[dependency].append(node.node_id)

        ready = [node.node_id for node in self.nodes if indegree[node.node_id] == 0]
        ordered_ids = []
        while ready:
            node_id = ready.pop(0)
            ordered_ids.append(node_id)
            for dependent in dependents[node_id]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)

        if len(ordered_ids) != len(self.nodes):
            cyclic = [node_id for node_id, degree in indegree.items() if degree > 0]
            raise ValueError(f"Evaluator dependency cycle detected: {cyclic}")
        return [self._nodes_by_id[node_id] for node_id in ordered_ids]

    def run(self) -> list[str]:
        if self.report_only:
            return self.render_reports()

        completed = []
        execution_hashes = {}
        for node in self.execution_order():
            node_context = self._execution_context(node)
            node_config = node.as_dict()
            cache_identity = self.result_store.semantic_cache_identity(
                node_config, node_context
            )
            dependency_hashes = {
                dependency: execution_hashes[dependency]
                for dependency in node.depends_on
            }
            execution_hash = node.config_hash(
                dependency_hashes,
                context=node_context,
            )

            cached = False
            if (
                not self.overwrite
                and self.result_store.adopt_allowlisted_completion(
                    node.node_id, execution_hash, node.as_dict()
                )
            ):
                cached = True
                logger.warning(
                    "Adopted exactly allowlisted completed evaluator node %s "
                    "after cache identity migration",
                    node.node_id,
                )
            elif not self.overwrite:
                cached = self.result_store.is_complete(
                    node.node_id,
                    execution_hash,
                    config=node_config,
                    context=node_context,
                )
            if cached:
                logger.warning("Skipping cached evaluator node %s", node.node_id)
                if self._should_render_report(node):
                    self._render_report(
                        node, self.result_store.load_result(node.node_id)
                    )
                execution_hashes[node.node_id] = self.result_store.completion_hash(
                    node.node_id, execution_hash
                )
                completed.append(node.node_id)
                continue

            self.result_store.mark_running(
                node.node_id,
                execution_hash,
                node_config,
                reset_progress=self.overwrite,
                cache_identity=cache_identity,
            )
            logger.warning("Running evaluator node %s", node.node_id)
            try:
                result = self.executor(
                    node,
                    self.result_store.view(node.depends_on),
                )
                if not isinstance(result, EvaluationResult):
                    raise TypeError(
                        f"Evaluator '{node.node_id}' must return EvaluationResult."
                    )
                for kind in result.result_kinds:
                    getattr(self.result_store, f"save_{kind}")(
                        node.node_id, getattr(result, kind)
                    )
                metadata = {
                    **dict(result.metadata),
                    "result_kinds": result.result_kinds,
                    **{
                        f"{kind}_rows": len(getattr(result, kind))
                        for kind in result.result_kinds
                    },
                }
                self.result_store.mark_complete(
                    node.node_id,
                    execution_hash,
                    node_config,
                    metadata=metadata,
                    cache_identity=cache_identity,
                )
                self._render_report(node, result)
                execution_hashes[node.node_id] = self.result_store.completion_hash(
                    node.node_id, execution_hash
                )
            except Exception as error:
                self.result_store.mark_failed(
                    node.node_id,
                    execution_hash,
                    node_config,
                    error=repr(error),
                )
                raise
            completed.append(node.node_id)
        return completed

    def _execution_context(self, node: EvaluatorNode) -> dict[str, Any]:
        return context_for_node(self.context, node.node_id)

    def render_reports(self) -> list[str]:
        """Rebuild reports from completed node results without running inference."""
        completed = []
        for node in self.execution_order():
            if self._should_render_report(node):
                result = self.result_store.load_result(node.node_id)
                self._render_report(node, result)
            completed.append(node.node_id)
        return completed

    def _should_render_report(self, node: EvaluatorNode) -> bool:
        return bool(
            self.generate_reports
            and self.reporter is not None
            and node.report.get("enabled", True) is not False
        )

    def _render_report(self, node: EvaluatorNode, result: EvaluationResult) -> None:
        if not self._should_render_report(node):
            return
        try:
            paths = [str(path) for path in self.reporter(node, result)]
        except Exception as error:
            logger.warning(
                "Report generation failed for evaluator node %s: %s",
                node.node_id,
                error,
            )
            self._record_report(node.node_id, error=repr(error))
            return
        self._record_report(node.node_id, paths=paths)

    def _record_report(
        self,
        node_id: str,
        paths: Iterable[str] = (),
        error: str | None = None,
    ) -> None:
        try:
            self.result_store.record_report(node_id, paths=paths, error=error)
        except Exception as record_error:
            logger.warning(
                "Could not record report status for evaluator node %s: %s",
                node_id,
                record_error,
            )

    @classmethod
    def for_evaluators(
        cls,
        nodes: Iterable[EvaluatorNode],
        result_store: ResultStore,
        evaluator_factory: Callable[[EvaluatorNode, ResultView], Any],
        models_factory: Callable[[EvaluatorNode], tuple[list[Any], list[Any]]],
        targets_factory: Callable[[EvaluatorNode], tuple[list[Any], list[Any]]]
        | None = None,
        **kwargs: Any,
    ) -> "EvaluationEngine":
        """Build an engine whose only node action is calling Evaluator.evaluate."""

        evaluators = {}

        def get_evaluator(node: EvaluatorNode):
            if node.node_id not in evaluators:
                evaluators[node.node_id] = evaluator_factory(
                    node, result_store.view(node.depends_on)
                )
            return evaluators[node.node_id]

        def execute(node: EvaluatorNode, results: ResultView) -> EvaluationResult:
            evaluator = get_evaluator(node)
            resume = getattr(evaluator, "resume_from_progress", None)
            if targets_factory is not None and callable(resume):
                targets, concepts = targets_factory(node)
                restored = resume(targets, concepts)
                if restored is not None:
                    return restored
            models, concepts = models_factory(node)
            return evaluator.evaluate(models, concepts)

        def report(node: EvaluatorNode, result: EvaluationResult):
            evaluator = get_evaluator(node)
            return evaluator.render_report(result, result_store.node_dir(node.node_id))

        return cls(
            nodes,
            result_store,
            execute,
            reporter=report,
            **kwargs,
        )

    @staticmethod
    def _validate_nodes(nodes: list[EvaluatorNode]) -> dict[str, EvaluatorNode]:
        if not nodes:
            raise ValueError("At least one evaluator must be configured.")
        nodes_by_id = {}
        for node in nodes:
            if node.node_id in nodes_by_id:
                raise ValueError(f"Duplicate evaluator id '{node.node_id}'.")
            nodes_by_id[node.node_id] = node
        for node in nodes:
            missing = [dependency for dependency in node.depends_on if dependency not in nodes_by_id]
            if missing:
                raise ValueError(
                    f"Evaluator '{node.node_id}' has unknown dependencies: {missing}"
                )
            if node.node_id in node.depends_on:
                raise ValueError(f"Evaluator '{node.node_id}' cannot depend on itself.")
        return nodes_by_id
