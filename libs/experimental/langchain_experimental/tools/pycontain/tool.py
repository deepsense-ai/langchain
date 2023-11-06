"""A tool for running python code in a ds_pycontain REPL.
See: https://github.com/deepsense-ai/ds-pycontain/
"""

import asyncio
from typing import Any, Optional, Type

from ds_pycontain.python_dockerized_repl import PythonContainerREPL
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.base import BaseTool

from langchain_experimental.tools.python.tool import sanitize_input


class PythonInputs(BaseModel):
    query: str = Field(description="python code snippet to run")


class PycontainREPLTool(BaseTool):
    """A tool for running python code in a dockerized REPL in exec mode."""

    name: str = "pycontain_repl"
    description: str = (
        "A Python shell. Use this to execute python commands. "
        "Input should be a valid python command. "
        "If you want to see the output of a value, you should print it out "
        "with `print(...)`."
    )
    # default is hard to provide here, user has to provide it
    python_repl: PythonContainerREPL = Field()
    sanitize_input: bool = True
    args_schema: Type[BaseModel] = PythonInputs

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool."""
        if self.sanitize_input:
            query = sanitize_input(query)
        result = self.python_repl.exec(query)
        return result

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self.run, query)
        return result


class PycontainAstREPLTool(PycontainREPLTool):
    """A tool for running python code in a dockerized REPL in eval mode."""

    name: str = "pycontain_repl_ast"
    description: str = (
        "A Python shell. Use this to execute python commands. "
        "Input should be a valid python command. "
        "When using this tool, sometimes output is abbreviated - "
        "make sure it does not look abbreviated before using it in your answer."
    )

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool."""
        if self.sanitize_input:
            query = sanitize_input(query)
        result = self.python_repl.eval(query)
        return result
