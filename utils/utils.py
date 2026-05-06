"""
Python代码执行工具
"""
import subprocess
import sys
import io
import traceback
import contextlib
from typing import Dict, Any, Optional


def execute_python_code(code: str, timeout: int = 30) -> Dict[str, Any]:
    """
    执行Python代码并返回结果
    
    Args:
        code: 要执行的Python代码
        timeout: 超时时间（秒）
    
    Returns:
        {
            "success": bool,
            "output": str,  # 执行结果
            "error": str,   # 错误信息（如果有）
            "return_value": str  # 返回值（如果有）
        }
    """
    # 清理代码（移除markdown代码块标记）
    code = code.strip()
    if code.startswith("```python"):
        code = code[10:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()
    
    # 创建受限的执行环境
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    local_vars = {}
    
    try:
        # 执行代码
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(code, {'__builtins__': __builtins__}, local_vars)
        
        # 获取输出
        stdout_value = stdout_capture.getvalue()
        stderr_value = stderr_capture.getvalue()
        
        # 检查是否有返回值
        return_value = None
        if 'result' in local_vars:
            return_value = str(local_vars['result'])
        elif 'answer' in local_vars:
            return_value = str(local_vars['answer'])
        
        return {
            "success": True,
            "output": stdout_value,
            "error": stderr_value,
            "return_value": return_value
        }
        
    except Exception as e:
        stderr_value = stderr_capture.getvalue()
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        
        return {
            "success": False,
            "output": stdout_capture.getvalue(),
            "error": error_msg,
            "return_value": None
        }


def execute_python_with_input(code: str, inputs: list = None, timeout: int = 30) -> Dict[str, Any]:
    """
    执行需要输入的Python代码
    
    Args:
        code: Python代码
        inputs: 输入值列表
        timeout: 超时时间
    
    Returns:
        执行结果字典
    """
    code = code.strip()
    if code.startswith("```python"):
        code = code[10:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()
    
    # 如果有输入需求，自动提供
    if inputs:
        # 使用input()模拟输入
        input_iter = iter(inputs)
        def mock_input(prompt=""):
            try:
                return next(input_iter)
            except StopIteration:
                return ""
        
        # 保存原始input函数
        original_input = __builtins__.get('input')
        __builtins__['input'] = mock_input
        
        try:
            result = execute_python_code(code, timeout)
            return result
        finally:
            # 恢复原始input
            if original_input:
                __builtins__['input'] = original_input
            else:
                del __builtins__['input']
    else:
        return execute_python_code(code, timeout)


def safe_eval(expression: str, context: Dict[str, Any] = None) -> Any:
    """
    安全地计算数学表达式
    
    Args:
        expression: 数学表达式字符串
        context: 变量上下文
    
    Returns:
        计算结果
    """
    if context is None:
        context = {}
    
    # 只允许安全的数学运算
    allowed_names = {
        'abs': abs, 'min': min, 'max': max, 'sum': sum,
        'int': int, 'float': float, 'round': round,
        'pow': pow, 'divmod': divmod
    }
    
    # 尝试安全计算
    try:
        # 替换常见运算符
        expression = expression.replace('×', '*').replace('÷', '/')
        expression = expression.replace('−', '-')
        
        result = eval(expression, {"__builtins__": allowed_names}, context)
        return result
    except Exception as e:
        return f"计算错误: {str(e)}"


if __name__ == "__main__":
    # 测试
    test_code = """
total = 5 * 4225558
guest_taken = total - 14
result = guest_taken
print(f"答案: {result}")
"""
    
    result = execute_python_code(test_code)
    print(f"成功: {result['success']}")
    print(f"输出: {result['output']}")
    print(f"返回值: {result['return_value']}")
    if result['error']:
        print(f"错误: {result['error']}")
