from lark import ast_utils, Lark
import dsl_ast
import sys

dsl_ast_module = sys.modules['dsl_ast']
with open('grammar.lark', 'r') as f:
    grammar = f.read()


def mk_compact(ast):
    edges = []
    for flow_statement in ast.flow_statements.statements:
        s = flow_statement.src.var.value, flow_statement.src.num.value
        d = flow_statement.dst.var.value, flow_statement.dst.num.value
        edges.append((s, d))
    return edges

def inc_compact(compact):
    new = []
    for flow_statement in compact:
        statement = []
        for flow_step in flow_statement:
            step = (flow_step[0], flow_step[1] + 1)
            statement.append(step)
        new.append(tuple(statement))
    return new

class DslInterpreter:
    parser = Lark(grammar)
    t = ast_utils.create_transformer(dsl_ast_module, dsl_ast.ToAst())
    
    def apply(self, first_script, then_script, times=1):
        out = []
        first = self.t.transform(self.parser.parse(first_script))
        then = self.t.transform(self.parser.parse(then_script))
        
        fi = mk_compact(first)
        th = mk_compact(then)
        while times:
            out.extend(fi)
            out.extend(th)
            fi = inc_compact(fi)
            th = inc_compact(th)
            times -= 1
        return out