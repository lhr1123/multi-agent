"""Feedback-driven Hungarian matching over a bounded task DAG."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import asdict

from models import ExecutionState, TaskRuntime
from services.assignment_service import assign_with_capacities, assign_subtasks_with_hungarian_core
from services.usage_tracker import BudgetExceeded, DeadlineExceeded


class ExecutionController:
    def __init__(self, tasks, agents, base_scores, config, tracker, feasible=None):
        self.tasks = tasks
        self.agents = agents
        self.scores = base_scores
        self.config = config
        self.tracker = tracker
        self.feasible = feasible or [[True] * len(agents) for _ in tasks]
        self.state = ExecutionState({t.id: TaskRuntime(t.id) for t in tasks})
        self.agent_outcomes = {agent.agent_id: {"successes": 0, "failures": 0} for agent in agents}
        if len(self.state.tasks) != len(tasks):
            raise ValueError("Duplicate task IDs")
        self.static = None
        if config.mode == "static":
            masked = [[s if self.feasible[i][j] else -1e9 for j, s in enumerate(row)]
                      for i, row in enumerate(base_scores)]
            self.static, _, _ = assign_subtasks_with_hungarian_core(masked, len(agents))

    def _components(self, i, j):
        agent = self.agents[j].agent_id
        history = self.state.tasks[self.tasks[i].id].attempts
        failures = sum(a["agent_id"] == agent and a["failure_kind"] == "capability" for a in history)
        calls = self.tracker.snapshot()["calls"]
        matching = [c for c in calls if c["operation"].endswith("|" + agent) and c["total_tokens"] is not None]
        operation_costs = {}
        for call in matching:
            operation_costs[call["operation"]] = operation_costs.get(call["operation"], 0) + call["total_tokens"]
        cost = (sum(operation_costs.values()) / len(operation_costs)
                if operation_costs else self.config.max_output_tokens)
        outcomes = self.agent_outcomes[agent]
        prior = self.config.reliability_priors.get(agent, 0.5)
        strength = self.config.reliability_prior_strength
        reliability = (strength * prior + outcomes["successes"]) / (
            strength + outcomes["successes"] + outcomes["failures"]
        )
        metadata = self.tasks[i].metadata or {}
        work_type = str(metadata.get("work_type", "solve")).lower()
        default_information_gain = {
            "solve": 1.0, "calculate": 1.0, "repair": 0.8,
            "criticize": 0.6, "verify": 0.6, "summarize": 0.2,
        }.get(work_type, 0.5)
        information_gain = metadata.get("information_gain", default_information_gain)
        try:
            information_gain = max(0.0, min(1.0, float(information_gain)))
        except (TypeError, ValueError):
            information_gain = default_information_gain
        diversity = 1.0 if not any(a["agent_id"] == agent for a in history) else 0.0
        return {"fit": max(0.0, min(1.0, self.scores[i][j] / 10)),
                "reliability": reliability,
                "cost": min(1.0, cost / (self.config.max_output_tokens * 4)),
                "repeat_failure": min(1.0, failures / max(1, self.config.max_attempts - 1)),
                "information_gain": information_gain,
                "diversity": diversity}

    @staticmethod
    def _required(task):
        return bool((task.metadata or {}).get("required", True))

    def _retry_feasible(self, task_index, agent_index):
        if self.config.mode != "rolling-hungarian":
            return True
        history = self.state.tasks[self.tasks[task_index].id].attempts
        if not history:
            return True
        last = history[-1]
        last_agent = last.get("agent_id")
        failure_kind = last.get("failure_kind")
        candidate = self.agents[agent_index].agent_id
        if failure_kind in {"transport", "format", "context", "tool"}:
            return candidate == last_agent
        if failure_kind == "capability":
            alternatives = any(
                other.agent_id != last_agent
                and self.feasible[task_index][other_index]
                and self.config.capacities.get(other.agent_id, self.config.max_parallel) > 0
                for other_index, other in enumerate(self.agents)
            )
            return candidate != last_agent if alternatives else True
        return True

    def _weights(self):
        return {"fit": self.config.fit_weight, "reliability": self.config.reliability_weight,
                "cost": -self.config.cost_weight, "repeat_failure": -self.config.failure_weight,
                "information_gain": self.config.information_gain_weight,
                "diversity": self.config.diversity_weight}

    def _raw_score(self, components):
        if not self.config.feedback:
            return components["fit"]
        return sum(self._weights()[key] * components[key] for key in self._weights())

    def _best_optional_utility(self):
        scores = []
        for i, task in enumerate(self.tasks):
            runtime = self.state.tasks[task.id]
            if self._required(task) or runtime.status in {"succeeded", "blocked", "skipped"}:
                continue
            if len(runtime.attempts) >= self.config.max_attempts or runtime.result.get("retryable") is False:
                continue
            if not all(self.state.tasks[d].status == "succeeded" for d in task.dependencies):
                continue
            for j, agent in enumerate(self.agents):
                if (self.feasible[i][j] and self._retry_feasible(i, j)
                        and self.config.capacities.get(agent.agent_id, self.config.max_parallel) > 0
                        and (self.static is None or self.static[i] == j)):
                    scores.append(self._raw_score(self._components(i, j)))
        return max(scores) if scores else None

    def _stop(self, reason):
        self.state.stop_reason = reason
        for task in self.tasks:
            runtime = self.state.tasks[task.id]
            if runtime.status in {"pending", "ready"} and any(
                d not in self.state.tasks or self.state.tasks[d].status != "succeeded" for d in task.dependencies
            ):
                runtime.status = "blocked"
        return self.state

    def run(self, execute):
        while True:
            if all(t.status == "succeeded" for t in self.state.tasks.values()):
                return self._stop("completed")
            required = [self.state.tasks[t.id] for t in self.tasks if self._required(t)]
            if required and all(runtime.status == "succeeded" for runtime in required):
                if self.config.early_stop_optional:
                    best_optional = self._best_optional_utility()
                    should_continue = (best_optional is not None
                                       and best_optional >= self.config.optional_utility_threshold)
                    self.state.termination = {
                        "required_complete": True,
                        "best_optional_utility": best_optional,
                        "optional_utility_threshold": self.config.optional_utility_threshold,
                        "decision": "continue_optional" if should_continue else "stop",
                    }
                    if not should_continue:
                        for task in self.tasks:
                            runtime = self.state.tasks[task.id]
                            if not self._required(task) and runtime.status in {"pending", "ready"}:
                                runtime.status = "skipped"
                        return self._stop("completed_early")
            try:
                self.tracker.check()
            except DeadlineExceeded:
                return self._stop("deadline_exceeded")
            exhausted = any(t.status == "failed" and len(t.attempts) >= self.config.max_attempts
                            for t in self.state.tasks.values())
            ready = []
            for i, task in enumerate(self.tasks):
                rt = self.state.tasks[task.id]
                if (rt.status in {"succeeded", "blocked"} or len(rt.attempts) >= self.config.max_attempts
                        or rt.result.get("retryable") is False):
                    continue
                if all(d in self.state.tasks and self.state.tasks[d].status == "succeeded" for d in task.dependencies):
                    rt.status = "ready"
                    ready.append(i)
            if not ready:
                return self._stop("attempts_exhausted" if exhausted else "blocked")
            # Bounded wave selection is an explicit heuristic, not budget-optimal matching.
            ready.sort(key=lambda i: (len(self.state.tasks[self.tasks[i].id].attempts), i))
            eligible = [i for i in ready if any(self.feasible[i][j] and
                        self._retry_feasible(i, j) and
                        self.config.capacities.get(a.agent_id, self.config.max_parallel) > 0 and
                        (self.static is None or self.static[i] == j)
                        for j, a in enumerate(self.agents))]
            batch = eligible[:self.config.max_parallel]
            if not batch:
                return self._stop("attempts_exhausted" if exhausted else "blocked")
            components = [[self._components(i, j) for j in range(len(self.agents))] for i in batch]
            raw = [[self._raw_score(c) for c in row] for row in components]
            # Required tasks: lexicographically maximize feasible cardinality, then score.
            bonus = 1 + 2 * len(batch) * max((abs(v) for row in raw for v in row), default=0)
            matrix = [[v + bonus for v in row] for row in raw]
            feasible = [[self.feasible[i][j] and self._retry_feasible(i, j)
                         and (self.static is None or self.static[i] == j)
                         for j in range(len(self.agents))] for i in batch]
            capacities = [self.config.capacities.get(a.agent_id, self.config.max_parallel) for a in self.agents]
            chosen = assign_with_capacities(matrix, capacities, feasible, self.config.policy,
                                           self.config.seed + len(self.state.rounds))
            pairs = [(i, j) for i, j in zip(batch, chosen) if j >= 0]
            trace = {"round": len(self.state.rounds), "candidate_ids": [self.tasks[i].id for i in batch],
                     "agent_ids": [a.agent_id for a in self.agents], "components": components,
                     "raw_scores": raw, "matching_scores": matrix, "feasible": feasible,
                     "capacities": capacities, "assignments": chosen,
                     "deferred_ids": [self.tasks[i].id for i in ready if i not in {p[0] for p in pairs}]}
            self.state.rounds.append(trace)
            if not pairs:
                return self._stop("blocked")
            for i, _ in pairs:
                self.state.tasks[self.tasks[i].id].status = "running"

            def invoke(i, j):
                task, agent = self.tasks[i], self.agents[j]
                deps = {d: self.state.tasks[d].result for d in task.dependencies}
                try:
                    return execute(task, agent, deps)
                except BudgetExceeded as exc:
                    return {"success": False, "error": str(exc), "failure_kind": "budget"}
                except DeadlineExceeded as exc:
                    return {"success": False, "error": str(exc), "failure_kind": "deadline"}
                except Exception as exc:
                    kind = "transport" if type(exc).__name__ in {
                        "TimeoutError", "APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError"
                    } else "execution"
                    return {"success": False, "error": str(exc), "failure_kind": kind}

            with ThreadPoolExecutor(max_workers=len(pairs)) as pool:
                futures = [pool.submit(copy_context().run, invoke, i, j) for i, j in pairs]
                outcomes = [future.result() for future in futures]
            for (i, j), result in zip(pairs, outcomes):
                rt = self.state.tasks[self.tasks[i].id]
                result["agent_id"] = self.agents[j].agent_id
                rt.status = "succeeded" if result.get("success") else "failed"
                rt.result = result
                rt.attempts.append({"agent_id": self.agents[j].agent_id, "success": bool(result.get("success")),
                                    "failure_kind": result.get("failure_kind", ""), "result": result})
                if result.get("success"):
                    self.agent_outcomes[self.agents[j].agent_id]["successes"] += 1
                elif result.get("failure_kind") == "capability":
                    self.agent_outcomes[self.agents[j].agent_id]["failures"] += 1
            if any(r.get("failure_kind") == "deadline" for r in outcomes):
                return self._stop("deadline_exceeded")
            if any(r.get("failure_kind") == "budget" for r in outcomes):
                return self._stop("budget_exhausted")
            if self.tracker.remaining_seconds() <= 0:
                return self._stop("deadline_exceeded")

    def to_dict(self):
        return asdict(self.state)
