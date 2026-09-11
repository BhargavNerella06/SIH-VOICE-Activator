"""
command - deterministic, rule-based command interpreter. See
command.interpreter for the full interface, schema, and design notes.
"""
from .interpreter import CommandInterpreter, CommandResult

__all__ = ["CommandInterpreter", "CommandResult"]
