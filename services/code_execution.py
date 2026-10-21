"""Isolated Python process with a real execution timeout (not a security sandbox)."""

import json
import subprocess
import sys


_RUNNER = """
import contextlib, io, json, sys, traceback
code = json.loads(sys.stdin.read())
stdout, stderr = io.StringIO(), io.StringIO()
namespace = {}
try:
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        exec(compile(code, '<agent>', 'exec'), namespace)
    value = namespace.get('result', namespace.get('answer'))
    payload = dict(success=True, output=stdout.getvalue(), error=stderr.getvalue(),
                   return_value=str(value) if value is not None else None)
except BaseException:
    payload = dict(success=False, output=stdout.getvalue(), error=traceback.format_exc(), return_value=None)
sys.stdout.write(json.dumps(payload))
"""


def execute_python(code, timeout=30):
    from services.result_parsing import strip_code_fence

    try:
        process = subprocess.run(
            [sys.executable, "-I", "-c", _RUNNER], input=json.dumps(strip_code_fence(code)),
            text=True, encoding="utf-8", capture_output=True, timeout=timeout,
        )
        if process.returncode != 0:
            return {"success": False, "error": process.stderr, "output": "", "return_value": None}
        return json.loads(process.stdout)
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Python execution timed out", "output": "", "return_value": None}
    except (ValueError, OSError) as exc:
        return {"success": False, "error": str(exc), "output": "", "return_value": None}
