"""
Workflow manager for multi-agent execution.
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from .actions import ActionRegistry, ActionType
from llm.llm_config import SUB_AGENT_MODEL, sub_agent_llm_model


class AgentCategory(Enum):
    TOOL = "tool"
    REASONING = "reasoning"
    TERMINATE = "terminate"


class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentPersona:
    agent_id: str
    agent_name: str
    category: AgentCategory
    description: str
    capabilities: List[str]


@dataclass
class WorkflowStep:
    step_id: str
    agent_id: str
    action_id: str
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    error: Optional[str] = None
    llm_response: Optional[str] = None
    token_usage: Dict[str, int] = field(default_factory=dict)


@dataclass
class Workflow:
    workflow_id: str
    task_description: str
    steps: List[WorkflowStep] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.PENDING
    results: Dict[str, Any] = field(default_factory=dict)


class WorkflowManager:
    """Workflow manager where agents decide whether to call tools."""

    AGENT_SYSTEM_PROMPTS = {
        "fileagent": "You are a file operation specialist.",
        "arxivagent": "You are an academic search specialist.",
        "bingagent": "You are a web search specialist.",
        "websiteagent": "You are a website parsing specialist.",
        "pythonagent": "You are a Python execution specialist.",
        "reasoningagent": "You are a logical reasoning specialist.",
        "criticagent": "You are a critical analysis specialist.",
        "reflectagent": "You are a reflection and improvement specialist.",
        "questionagent": "You are a question decomposition specialist.",
        "summerizeagent": "You are a summarization specialist.",
        "concludeagent": "You are a conclusion specialist.",
        "modifieragent": "You are an error fixing specialist.",
        "terminateagent": (
            "You are the final answer extractor. "
            "Read all intermediate evidence, resolve conflicts, and output the single best final answer. "
            "For math word problems, prefer a precise numeric final answer."
        ),
    }

    TOOL_AGENT_ACTIONS = {
        "fileagent": ["read_file", "write_file", "list_directory", "delete_file"],
        "arxivagent": ["arxiv_search"],
        "bingagent": ["bing_search"],
        "websiteagent": ["visit_website", "extract_content"],
        "pythonagent": ["execute_python"],
    }

    def __init__(
        self,
        client=None,
        model_name: str = SUB_AGENT_MODEL,
        persona_file: str = None,
    ):
        self.client = client or sub_agent_llm_model
        self.model_name = model_name

        self.persona_file = persona_file or self._get_default_persona_path()
        self.agent_personas: Dict[str, AgentPersona] = {}
        self.action_registry = ActionRegistry()
        self.current_workflow: Optional[Workflow] = None

        self._load_personas()

    def _get_default_persona_path(self) -> str:
        return str(Path(__file__).parent / "personas.jsonl")

    def _load_personas(self):
        try:
            with open(self.persona_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        persona_data = json.loads(line)
                    except json.JSONDecodeError:
                        # Keep loading other lines instead of failing the whole pool.
                        continue

                    category_str = persona_data.get("category", "tool")
                    category = AgentCategory(category_str)

                    caps = persona_data.get("capabilities", [])
                    if not caps and persona_data.get("capability"):
                        caps = persona_data.get("capability")

                    persona = AgentPersona(
                        agent_id=persona_data["agent_id"],
                        agent_name=persona_data["agent_name"],
                        category=category,
                        description=persona_data.get("description", ""),
                        capabilities=caps,
                    )
                    self.agent_personas[persona.agent_id] = persona
        except FileNotFoundError:
            print(f"Warning: persona file not found: {self.persona_file}")

    def get_agents_by_category(self, category: AgentCategory) -> List[AgentPersona]:
        return [p for p in self.agent_personas.values() if p.category == category]

    def get_tool_agents(self) -> List[AgentPersona]:
        return self.get_agents_by_category(AgentCategory.TOOL)

    def get_reasoning_agents(self) -> List[AgentPersona]:
        return self.get_agents_by_category(AgentCategory.REASONING)

    def get_terminate_agents(self) -> List[AgentPersona]:
        return self.get_agents_by_category(AgentCategory.TERMINATE)

    def get_agent_by_id(self, agent_id: str) -> Optional[AgentPersona]:
        return self.agent_personas.get(agent_id)

    def create_workflow(self, workflow_id: str, task_description: str) -> Workflow:
        workflow = Workflow(workflow_id=workflow_id, task_description=task_description)
        self.current_workflow = workflow
        return workflow

    def add_step(self, agent_id: str, action_id: str, input_data: Dict[str, Any]) -> WorkflowStep:
        if not self.current_workflow:
            raise ValueError("No active workflow.")
        step = WorkflowStep(
            step_id=f"step_{len(self.current_workflow.steps)}",
            agent_id=agent_id,
            action_id=action_id,
            input_data=input_data,
        )
        self.current_workflow.steps.append(step)
        return step

    def _validate_action_input(self, action_id: str, input_data: Dict[str, Any]) -> Optional[str]:
        action = self.action_registry.get_action(action_id)
        if action is None:
            return f"Action not found: {action_id}"
        missing = []
        for key in action.required_params:
            if key not in input_data:
                missing.append(key)
                continue
            value = input_data.get(key)
            if value is None:
                missing.append(key)
            elif isinstance(value, str) and not value.strip():
                missing.append(key)
            elif isinstance(value, (list, dict)) and len(value) == 0:
                missing.append(key)
        if missing:
            return f"Missing required params for `{action_id}`: {', '.join(missing)}"
        return None

    def _build_action_prompt(self, action_id: str, input_data: Dict[str, Any]) -> str:
        if action_id == "logical_reasoning":
            problem = input_data.get("problem", "")
            context = input_data.get("context", {})
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            return f"Problem:\n{problem}\n\nContext:\n{context_str}\n\nPlease reason and answer."
        if action_id == "terminate_task":
            result = input_data.get("result", "")
            summary = input_data.get("summary", "")
            return (
                f"Result:\n{result}\n\n"
                f"Summary:\n{summary}\n\n"
                "Task:\n"
                "1) Read all evidence and extract the true final answer.\n"
                "2) If evidence conflicts, choose the most self-consistent answer and explain briefly.\n"
                "3) For math problems, final_answer should be the final numeric value only.\n\n"
                "Return strict JSON only:\n"
                "{\n"
                '  "status": "ok|partial|failed",\n'
                '  "final_answer": "<final value only>",\n'
                '  "result_text": "<short rationale>",\n'
                '  "key_facts": [{"name":"", "value":"", "source":""}],\n'
                '  "confidence": 0.0\n'
                "}\n"
            )
        if action_id == "execute_python":
            return input_data.get("code", "")
        return "\n".join([f"{k}: {v}" for k, v in input_data.items()])

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return {
                "success": True,
                "content": response.choices[0].message.content,
                "token_usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e), "content": "", "token_usage": {}}

    def _safe_json_loads(self, content: str) -> Optional[Dict[str, Any]]:
        """Best-effort JSON parser for model output."""
        if content is None:
            return None
        text = content.strip()
        if not text:
            return None
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 2 and lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1]).strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            return None
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                return None
            try:
                parsed = json.loads(m.group(0))
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None

    def _normalize_structured_response(
        self,
        raw_content: str,
        fallback_text: str = "",
        fallback_status: str = "partial",
    ) -> Dict[str, Any]:
        """
        Normalize any agent output to a structured dict for robust downstream extraction.
        """
        parsed = self._safe_json_loads(raw_content or "")
        if parsed is None:
            text = (fallback_text or raw_content or "").strip()
            return {
                "status": fallback_status,
                "final_answer": text,
                "result_text": text,
                "key_facts": [],
                "confidence": 0.5,
            }

        status = str(parsed.get("status", "partial"))
        final_answer = str(parsed.get("final_answer", "") or "")
        result_text = str(parsed.get("result_text", "") or "")
        key_facts = parsed.get("key_facts", [])
        confidence = parsed.get("confidence", 0.5)

        if not isinstance(key_facts, list):
            key_facts = []
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        if not final_answer and result_text:
            final_answer = result_text
        if not result_text and final_answer:
            result_text = final_answer

        return {
            "status": status,
            "final_answer": final_answer,
            "result_text": result_text,
            "key_facts": key_facts,
            "confidence": confidence,
        }

    def _build_tool_decision_prompt(
        self,
        agent_id: str,
        task_action_id: str,
        task_input: Dict[str, Any],
    ) -> str:
        available_actions = self.TOOL_AGENT_ACTIONS.get(agent_id, [task_action_id])
        extra_rule = ""
        if agent_id == "websiteagent":
            extra_rule = (
                "Website tool rule: only set use_tool=true if you have a valid http/https URL. "
                "Never use placeholder/example domains. If no valid URL exists, set use_tool=false.\n"
            )
        return (
            "You are handling one subtask. Decide whether to call a tool.\n"
            f"{extra_rule}"
            f"Candidate action from planner: {task_action_id}\n"
            f"Available actions: {available_actions}\n"
            f"Task payload: {json.dumps(task_input, ensure_ascii=False)}\n\n"
            "Return strict JSON only:\n"
            "{\n"
            '  "use_tool": true or false,\n'
            '  "chosen_action": "<one available action when use_tool=true>",\n'
            '  "tool_input": { ... },\n'
            '  "final_answer": "<direct answer when use_tool=false or fallback>",\n'
            '  "reason": "<short reason>"\n'
            "}\n"
        )

    def _normalize_url(self, url: str) -> str:
        raw = str(url or "").strip()
        if not raw:
            return ""
        if raw.lower().startswith("www."):
            raw = "https://" + raw
        if not re.match(r"^https?://", raw, flags=re.IGNORECASE):
            return ""
        parsed = urllib.parse.urlparse(raw)
        if parsed.scheme.lower() not in {"http", "https"}:
            return ""
        if not parsed.netloc:
            return ""
        return urllib.parse.urlunparse(parsed)

    def _agent_decide_tool_call(
        self,
        agent: AgentPersona,
        step: WorkflowStep,
    ) -> Dict[str, Any]:
        """Ask the tool agent LLM to decide whether/how to call tools."""
        system_prompt = self.AGENT_SYSTEM_PROMPTS.get(step.agent_id, f"You are {agent.agent_name}.")
        user_prompt = self._build_tool_decision_prompt(step.agent_id, step.action_id, step.input_data)
        llm_result = self._call_llm(system_prompt, user_prompt)
        if not llm_result.get("success"):
            return {
                "ok": False,
                "error": llm_result.get("error", "Tool decision failed"),
                "tokens": llm_result.get("token_usage", {}),
            }

        decision = self._safe_json_loads(llm_result.get("content", ""))
        if decision is None:
            # Fallback: still let agent use planned action.
            fallback_use_tool = True
            if step.agent_id == "websiteagent":
                if not self._normalize_url((step.input_data or {}).get("url", "")):
                    fallback_use_tool = False
            decision = {
                "use_tool": fallback_use_tool,
                "chosen_action": step.action_id,
                "tool_input": step.input_data,
                "final_answer": llm_result.get("content", ""),
                "reason": "Decision JSON parse failed, fallback to planned action.",
            }

        if "use_tool" not in decision:
            decision["use_tool"] = True
        if decision.get("use_tool") and not decision.get("chosen_action"):
            decision["chosen_action"] = step.action_id
        if decision.get("use_tool") and not isinstance(decision.get("tool_input"), dict):
            decision["tool_input"] = step.input_data

        return {"ok": True, "decision": decision, "tokens": llm_result.get("token_usage", {})}

    def _agent_summarize_tool_output(
        self,
        agent: AgentPersona,
        action_id: str,
        tool_input: Dict[str, Any],
        tool_output: str,
    ) -> Dict[str, Any]:
        """Let the same agent produce final response from tool output."""
        system_prompt = self.AGENT_SYSTEM_PROMPTS.get(agent.agent_id, f"You are {agent.agent_name}.")
        user_prompt = (
            "You called a tool for a subtask. Use the tool output to answer.\n"
            f"Tool action: {action_id}\n"
            f"Tool input: {json.dumps(tool_input, ensure_ascii=False)}\n"
            f"Tool output:\n{tool_output}\n\n"
            "Return strict JSON only:\n"
            "{\n"
            '  "status": "ok|partial|failed",\n'
            '  "final_answer": "<concise answer>",\n'
            '  "result_text": "<longer explanation>",\n'
            '  "key_facts": [{"name":"", "value":"", "source":""}],\n'
            '  "confidence": 0.0\n'
            "}\n"
        )
        return self._call_llm(system_prompt, user_prompt)

    def _execute_tool_action(self, action_id: str, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute supported tool actions directly. Return None for LLM fallback."""
        if action_id == "execute_python":
            from utils.utils import execute_python_code

            result = execute_python_code(input_data.get("code", ""))
            if result.get("success"):
                output = result.get("output") or ""
                return_value = result.get("return_value") or ""
                return {
                    "success": True,
                    "content": f"Execution output:\n{output}\nReturn value: {return_value}",
                }
            return {"success": False, "content": result.get("error", "Python execution failed")}

        if action_id == "read_file":
            path = Path(input_data.get("path", ""))
            encoding = input_data.get("encoding", "utf-8")
            return {"success": True, "content": path.read_text(encoding=encoding)}

        if action_id == "write_file":
            path = Path(input_data.get("path", ""))
            path.parent.mkdir(parents=True, exist_ok=True)
            encoding = input_data.get("encoding", "utf-8")
            mode = input_data.get("mode", "w")
            with path.open(mode, encoding=encoding) as f:
                f.write(input_data.get("content", ""))
            return {"success": True, "content": f"Wrote file: {path}"}

        if action_id == "list_directory":
            path = Path(input_data.get("path", "."))
            lines = []
            for item in sorted(path.iterdir(), key=lambda p: p.name.lower()):
                lines.append(f"{'dir' if item.is_dir() else 'file'}: {item.name}")
            return {"success": True, "content": "\n".join(lines)}

        if action_id == "delete_file":
            path = Path(input_data.get("path", ""))
            if path.is_dir():
                return {"success": False, "content": f"Refusing to delete directory: {path}"}
            if not path.exists():
                return {"success": False, "content": f"File not found: {path}"}
            path.unlink()
            return {"success": True, "content": f"Deleted file: {path}"}

        if action_id == "visit_website":
            url = self._normalize_url(input_data.get("url", ""))
            if not url:
                return {"success": False, "content": "Invalid or missing URL for visit_website."}
            timeout = int(input_data.get("timeout", 30))
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            return {"success": True, "content": html[:5000]}

        if action_id == "extract_content":
            url = self._normalize_url(input_data.get("url", ""))
            if not url:
                return {"success": False, "content": "Invalid or missing URL for extract_content."}
            timeout = int(input_data.get("timeout", 30))
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            text = re.sub(r"<script[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
            text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return {"success": True, "content": text[:3000]}

        if action_id == "arxiv_search":
            query = input_data.get("query", "")
            max_results = int(input_data.get("max_results", 10))
            encoded = urllib.parse.quote(query)
            url = (
                "http://export.arxiv.org/api/query?"
                f"search_query=all:{encoded}&start=0&max_results={max_results}"
            )
            with urllib.request.urlopen(url, timeout=20) as resp:
                xml_content = resp.read().decode("utf-8", errors="ignore")

            root = ET.fromstring(xml_content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = []
            for entry in root.findall("atom:entry", ns):
                title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
                summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
                entries.append(f"Title: {title}\nSummary: {summary[:300]}")
            return {"success": True, "content": "\n\n".join(entries) if entries else "No arXiv results."}

        if action_id == "bing_search":
            # Public fallback without API key.
            query = input_data.get("query", "")
            max_results = int(input_data.get("max_results", 10))
            url = (
                "https://api.duckduckgo.com/?"
                + urllib.parse.urlencode(
                    {
                        "q": query,
                        "format": "json",
                        "no_redirect": "1",
                        "no_html": "1",
                    }
                )
            )
            with urllib.request.urlopen(url, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="ignore"))

            lines = []
            if payload.get("AbstractText"):
                lines.append(payload["AbstractText"])
            related = payload.get("RelatedTopics") or []
            count = 0
            for item in related:
                if isinstance(item, dict) and item.get("Text"):
                    lines.append(item["Text"])
                    count += 1
                elif isinstance(item, dict) and item.get("Topics"):
                    for sub in item.get("Topics", []):
                        if isinstance(sub, dict) and sub.get("Text"):
                            lines.append(sub["Text"])
                            count += 1
                            if count >= max_results:
                                break
                if count >= max_results:
                    break
            return {"success": True, "content": "\n".join(lines) if lines else "No search results."}

        return None

    def execute_workflow(self) -> Dict[str, Any]:
        if not self.current_workflow:
            return {"error": "No active workflow"}

        self.current_workflow.status = WorkflowStatus.RUNNING
        results: Dict[str, Any] = {}
        total_tokens = 0

        for step in self.current_workflow.steps:
            step.status = WorkflowStatus.RUNNING

            agent = self.get_agent_by_id(step.agent_id)
            action = self.action_registry.get_action(step.action_id)
            if agent is None:
                step.status = WorkflowStatus.FAILED
                step.error = f"Agent not found: {step.agent_id}"
                results[step.step_id] = {"error": step.error}
                continue
            if action is None:
                step.status = WorkflowStatus.FAILED
                step.error = f"Action not found: {step.action_id}"
                results[step.step_id] = {"error": step.error}
                continue

            print(f"\n[{step.step_id}] {agent.agent_name} -> {action.action_name}")

            is_tool_action = action.action_type in {
                ActionType.FILE_OPERATION,
                ActionType.SEARCH,
                ActionType.WEB_ACCESS,
                ActionType.CODE_EXECUTION,
            }

            if is_tool_action:
                # Tool usage should be decided by the assigned sub-agent (LLM), not hard-executed.
                decision_result = self._agent_decide_tool_call(agent, step)
                if not decision_result.get("ok"):
                    step.status = WorkflowStatus.FAILED
                    step.error = decision_result.get("error", "Tool decision failed")
                    step.token_usage = decision_result.get("tokens", {})
                    total_tokens += step.token_usage.get("total_tokens", 0)
                    results[step.step_id] = {"error": step.error}
                    continue

                decision_tokens = decision_result.get("tokens", {})
                total_tokens += decision_tokens.get("total_tokens", 0)
                decision = decision_result["decision"]
                available_actions = self.TOOL_AGENT_ACTIONS.get(step.agent_id, [step.action_id])

                if decision.get("use_tool"):
                    chosen_action = decision.get("chosen_action", step.action_id)
                    if chosen_action not in available_actions:
                        if step.action_id in available_actions:
                            chosen_action = step.action_id
                        else:
                            chosen_action = available_actions[0]
                    chosen_input = decision.get("tool_input", step.input_data)
                    if not isinstance(chosen_input, dict):
                        chosen_input = step.input_data

                    if chosen_action in {"visit_website", "extract_content"}:
                        normalized_url = self._normalize_url((chosen_input or {}).get("url", ""))
                        if not normalized_url:
                            response_content = (
                                str(decision.get("final_answer", "") or "").strip()
                                or "No valid URL found in task payload; skipped website access."
                            )
                            structured = self._normalize_structured_response(
                                json.dumps(
                                    {
                                        "status": "failed",
                                        "final_answer": response_content,
                                        "result_text": response_content,
                                        "key_facts": [],
                                        "confidence": 0.3,
                                    },
                                    ensure_ascii=False,
                                ),
                                fallback_text=response_content,
                            )
                            step.llm_response = response_content
                            step.token_usage = decision_tokens
                            step.status = WorkflowStatus.COMPLETED
                            step.output_data = {
                                "agent": agent.agent_name,
                                "action": chosen_action,
                                "response": response_content,
                                "input": chosen_input,
                                "tool_decision": decision,
                                "structured_response": structured,
                            }
                            results[step.step_id] = step.output_data
                            continue
                        chosen_input["url"] = normalized_url

                    validation_error = self._validate_action_input(chosen_action, chosen_input)
                    if validation_error:
                        step.status = WorkflowStatus.FAILED
                        step.error = validation_error
                        results[step.step_id] = {"error": validation_error}
                        continue

                    try:
                        tool_result = self._execute_tool_action(chosen_action, chosen_input)
                    except Exception as e:
                        tool_result = {"success": False, "content": str(e)}

                    if tool_result is None:
                        step.status = WorkflowStatus.FAILED
                        step.error = f"Unsupported tool action: {chosen_action}"
                        results[step.step_id] = {"error": step.error}
                        continue

                    if not tool_result.get("success"):
                        step.status = WorkflowStatus.FAILED
                        step.error = tool_result.get("content", "Tool execution failed")
                        results[step.step_id] = {"error": step.error}
                        continue

                    summarize = self._agent_summarize_tool_output(
                        agent=agent,
                        action_id=chosen_action,
                        tool_input=chosen_input,
                        tool_output=tool_result.get("content", ""),
                    )
                    if summarize.get("success"):
                        summary_tokens = summarize.get("token_usage", {})
                        total_tokens += summary_tokens.get("total_tokens", 0)
                        response_content = summarize.get("content", "")
                        structured = self._normalize_structured_response(
                            response_content,
                            fallback_text=tool_result.get("content", ""),
                            fallback_status="partial",
                        )
                        step.llm_response = response_content
                        step.token_usage = {
                            "prompt_tokens": decision_tokens.get("prompt_tokens", 0)
                            + summary_tokens.get("prompt_tokens", 0),
                            "completion_tokens": decision_tokens.get("completion_tokens", 0)
                            + summary_tokens.get("completion_tokens", 0),
                            "total_tokens": decision_tokens.get("total_tokens", 0)
                            + summary_tokens.get("total_tokens", 0),
                        }
                        step.status = WorkflowStatus.COMPLETED
                        step.output_data = {
                            "agent": agent.agent_name,
                            "action": chosen_action,
                            "response": response_content,
                            "input": chosen_input,
                            "tool_output": tool_result.get("content", ""),
                            "tool_decision": decision,
                            "structured_response": structured,
                        }
                        results[step.step_id] = step.output_data
                    else:
                        # Fallback to raw tool output if summarization fails.
                        response_content = tool_result.get("content", "")
                        structured = self._normalize_structured_response(
                            "",
                            fallback_text=response_content,
                            fallback_status="partial",
                        )
                        step.llm_response = response_content
                        step.token_usage = decision_tokens
                        step.status = WorkflowStatus.COMPLETED
                        step.output_data = {
                            "agent": agent.agent_name,
                            "action": chosen_action,
                            "response": response_content,
                            "input": chosen_input,
                            "tool_output": response_content,
                            "tool_decision": decision,
                            "structured_response": structured,
                        }
                        results[step.step_id] = step.output_data
                    continue

                # Agent decides to not call tools.
                response_content = decision.get("final_answer", "") or decision.get("reason", "")
                structured = self._normalize_structured_response(
                    json.dumps(
                        {
                            "status": "ok" if response_content else "partial",
                            "final_answer": response_content,
                            "result_text": response_content,
                            "key_facts": [],
                            "confidence": 0.6,
                        },
                        ensure_ascii=False,
                    ),
                    fallback_text=response_content,
                )
                step.llm_response = response_content
                step.token_usage = decision_tokens
                step.status = WorkflowStatus.COMPLETED
                step.output_data = {
                    "agent": agent.agent_name,
                    "action": action.action_name,
                    "response": response_content,
                    "input": step.input_data,
                    "tool_decision": decision,
                    "structured_response": structured,
                }
                results[step.step_id] = step.output_data
                continue

            # Non-tool actions keep direct LLM execution.
            validation_error = self._validate_action_input(step.action_id, step.input_data)
            if validation_error:
                step.status = WorkflowStatus.FAILED
                step.error = validation_error
                results[step.step_id] = {"error": validation_error}
                continue

            system_prompt = self.AGENT_SYSTEM_PROMPTS.get(
                step.agent_id,
                f"You are {agent.agent_name}.",
            )
            user_prompt = self._build_action_prompt(step.action_id, step.input_data)
            llm_result = self._call_llm(system_prompt, user_prompt)
            if llm_result.get("success"):
                step.llm_response = llm_result["content"]
                step.token_usage = llm_result.get("token_usage", {})
                total_tokens += step.token_usage.get("total_tokens", 0)
                step.status = WorkflowStatus.COMPLETED
                structured = self._normalize_structured_response(
                    llm_result["content"],
                    fallback_text=llm_result["content"],
                    fallback_status="partial",
                )
                step.output_data = {
                    "agent": agent.agent_name,
                    "action": action.action_name,
                    "response": llm_result["content"],
                    "input": step.input_data,
                    "structured_response": structured,
                }
                results[step.step_id] = step.output_data
            else:
                step.status = WorkflowStatus.FAILED
                step.error = llm_result.get("error", "LLM call failed")
                results[step.step_id] = {"error": step.error}

        self.current_workflow.status = WorkflowStatus.COMPLETED
        self.current_workflow.results = results
        return {
            "workflow_id": self.current_workflow.workflow_id,
            "results": results,
            "total_tokens": total_tokens,
        }

    def get_workflow_status(self) -> Dict[str, Any]:
        if not self.current_workflow:
            return {"status": "no_workflow"}
        return {
            "workflow_id": self.current_workflow.workflow_id,
            "task_description": self.current_workflow.task_description,
            "status": self.current_workflow.status.value,
            "steps_count": len(self.current_workflow.steps),
            "results": self.current_workflow.results,
        }

    def reset_workflow(self):
        self.current_workflow = None

    def list_all_agents(self) -> List[Dict[str, Any]]:
        return [
            {
                "agent_id": p.agent_id,
                "agent_name": p.agent_name,
                "category": p.category.value,
                "description": p.description,
                "capabilities": p.capabilities,
            }
            for p in self.agent_personas.values()
        ]
