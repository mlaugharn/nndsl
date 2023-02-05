from dataclasses import dataclass
from typing import List
from lark.tree import Meta
from lark import Transformer, ast_utils, v_args, Discard

@dataclass
class _Ast(ast_utils.Ast):
    pass

@dataclass
class Flowchart(_Ast, ast_utils.WithMeta):
    meta: Meta
    flow_type: str
    flow_statements: list

@dataclass
class FlowStep(_Ast):
    var: str
    num: int

@dataclass
class _NEWLINE(_Ast):
    pass

@dataclass
class SrcLabel(_Ast):
    label: str

@dataclass
class DstLabel(_Ast):
    label: str

@dataclass
class FlowStatement(_Ast):
    src: FlowStep
    slabel: SrcLabel
    dst: FlowStep
    dlabel: DstLabel


@dataclass
class FlowStatements(_Ast, ast_utils.AsList):
    statements: List[FlowStatement]

class ToAst(Transformer):
    @v_args(inline=True)
    def start(self, x):
        return x
    def INT(self, tok):
        "Convert the value of `tok` from string to int, while maintaining line number & column."
        return tok.update(value=int(tok))
    def NEWLINE(self, tok):
        return Discard