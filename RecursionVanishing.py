from abc import abstractmethod
from typing import Generic
from typing import TypeVar


class Symbol:
    @abstractmethod
    def is_terminal(self):
        pass

    @abstractmethod
    def get_name(self):
        pass

    def __eq__(self, other):
        return self.is_terminal() == other.is_terminal() and self.get_name() == other.get_name()

    def __hash__(self):
        return str.__hash__(("__TERMINAL_" if self.is_terminal() else "") + self.get_name())

    def __repr__(self):
        return self.__str__()

    def is_empty(self):
        return False


class NonTerminal(Symbol):
    def __init__(self, name : str):
        self.name = name

    def is_terminal(self):
        return False

    def __str__(self):
        return "<" + self.get_name() + ">"

    def get_name(self):
        return self.name


class Terminal(Symbol):
    def __init__(self, name: str):
        self.name = name

    def is_terminal(self):
        return True

    def __str__(self):
        return self.get_name()

    def get_name(self):
        return self.name


class Eps(Terminal):
    def __init__(self):
        pass

    def is_terminal(self):
        return True

    def is_empty(self):
        return True

    def __str__(self):
        return self.get_name()

    def get_name(self):
        return "{e}"


class Rule:
    def __init__(self, lhs: NonTerminal, *rhs: Symbol):
        self.lhs = lhs
        self.rhs = list(rhs)

    def __str__(self):
        return str(self.lhs) + " --> " + ' '.join(map(str, self.rhs))

    def __repr__(self):
        return self.__str__()


class Grammar:

    class Terminals:
        def __init__(self, *symbols: Terminal):
            self.symbols = list(symbols)

    class NonTerminals:
        def __init__(self, *symbols: NonTerminal):
            self.symbols = list(symbols)

    class Rules:
        def __init__(self, *rules: Rule):
            self.rules = list(rules)

    def __init__(self, terminals: Terminals, non_terminals: NonTerminals, start: NonTerminal, rules: Rules):
        self.terminals = terminals.symbols
        self.non_terminals = non_terminals.symbols
        self.start = start
        self.rules = rules.rules

    def __str__(self):
        terminals = "terminals: " + str(self.terminals) + "\n"
        non_terminals = "non-terminals: " + str(self.non_terminals) + "\n"
        start = "start: " + str(self.start) + "\n"
        rules = "rules:\n" + "\n".join(map(str, self.rules))
        return terminals + non_terminals + start + rules

    def __repr__(self):
        return str(self)

    def filter_by_non_terms(self, non_terms: set):
        tmp_non_terminals = []
        for non_terminal in self.non_terminals:
            if non_terminal in non_terms:
                tmp_non_terminals.append(non_terminal)
        self.non_terminals = tmp_non_terminals
        tmp_rules = []
        for rule in self.rules:
            if rule.lhs in tmp_non_terminals:
                all_lhs_alive = True
                for rhs_arg in rule.rhs:
                    if not rhs_arg.is_terminal() and rhs_arg not in tmp_non_terminals:
                        all_lhs_alive = False
                if all_lhs_alive:
                    tmp_rules.append(rule)
        self.non_terminals = tmp_non_terminals
        self.rules = tmp_rules


def find_vanishing_iteration(vanish: set, grammar: Grammar) -> set:
    vanish_tmp = vanish.copy()
    for rule in grammar.rules:
        van = True
        for sym in rule.rhs:
            if not sym.is_terminal() and sym not in vanish:
                van = False
                break
        if van:
            vanish_tmp.add(rule.lhs)
    return vanish_tmp


def find_vanishing(grammar: Grammar) -> set:
    vanish = set()
    vanish_tmp = set()
    for rule in grammar.rules:
        if len(rule.rhs) == 1 and rule.rhs[0].is_empty():
            vanish_tmp.add(rule.lhs)
    while len(vanish) != len(vanish_tmp):
        vanish, vanish_tmp = vanish_tmp, find_vanishing_iteration(vanish_tmp, grammar)
    return vanish


T = TypeVar('T')


class Graph(Generic[T]):

    def __init__(self, nodes: list):
        self.nodes = nodes
        self.edges = [set() for _ in range(len(nodes))]
        self.used_buffer = [False] * len(nodes)

    def get_item(self, index: int) -> T:
        return self.nodes[index]

    def get_index(self, item: T) -> int:
        return self.nodes.index(item)

    def add_edge(self, u: int, v: int):
        self.edges[u].add(v)

    def in_cycle(self, s: int, t=None) -> bool:
        if t is not None:
            if t == s:
                return True
        else:
            self.used_buffer = [False] * len(self.nodes)
            t = s
        if self.used_buffer[s]:
            return False
        self.used_buffer[s] = True
        for to in self.edges[s]:
            if self.in_cycle(to, t):
                return True
        return False


def build_recurse_graph(grammar: Grammar) -> Graph[NonTerminal]:
    graph = Graph(grammar.non_terminals)
    for rule in grammar.rules:
        for term in rule.rhs:
            if not term.is_terminal():
                graph.add_edge(graph.get_index(rule.lhs), graph.get_index(term))
    return graph


def find_recursive(grammar: Grammar) -> list:
    graph = build_recurse_graph(grammar)
    recursive = []
    for non_term in grammar.non_terminals:
        if graph.in_cycle(graph.get_index(non_term)):
            recursive.append(non_term)
    return recursive


def grammar_expressions():
    plus = Terminal('+')
    aster = Terminal('*')
    const = Terminal('C')
    brace_op = Terminal('(')
    brace_cl = Terminal(')')
    terminals = Grammar.Terminals(plus, aster, const, brace_op, brace_cl, Eps())
    expr = NonTerminal('expr')
    sum_t = NonTerminal('sum')
    mult = NonTerminal('mult')
    non_terminals = Grammar.NonTerminals(expr, sum_t, mult)
    start = expr
    rule1 = Rule(sum_t, expr, plus, expr)
    rule2 = Rule(expr, sum_t)
    rule3 = Rule(expr, const)
    rule4 = Rule(mult, expr, aster, expr)
    rule5 = Rule(expr, mult)
    rule6 = Rule(expr, brace_op, expr, brace_cl)
    rule7 = Rule(expr, Eps())
    rules = Grammar.Rules(rule1, rule2, rule3, rule4, rule5, rule6, rule7)
    return Grammar(terminals, non_terminals, start, rules)


if __name__ == '__main__':
    g = grammar_expressions()
    # normalize_grammar(g)
    print(find_recursive(g))
    print(find_vanishing(g))
    print(g)
