
from abc import abstractmethod
from typing import Generic, List
from typing import TypeVar
from collections import defaultdict
from copy import deepcopy


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
    def __init__(self, name: str):
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
        super().__init__("{e}")

    def is_terminal(self):
        return True

    def is_empty(self):
        return True

    def __str__(self):
        return self.get_name()


class Rule:
    def __init__(self, lhs: NonTerminal, *rhs: Symbol):
        self.lhs = lhs
        self.rhs = list(rhs)

    def __str__(self):
        return str(self.lhs) + " --> " + ' '.join(map(str, self.rhs))

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.lhs == other.lhs and all(x[0] == x[1] for x in zip(self.rhs, other.rhs))


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


def find_alive_iteration(alive: set, grammar: Grammar):
    alive_tmp = alive.copy()
    for rule in grammar.rules:
        valid = True
        for rhs_arg in rule.rhs:
            if not rhs_arg.is_terminal() and rhs_arg not in alive:
                valid = False
        if valid:
            alive_tmp.add(rule.lhs)
    return alive_tmp


def find_alive(grammar: Grammar):
    alive = set()
    alive_tmp = find_alive_iteration(alive, grammar)
    while alive != alive_tmp:
        alive, alive_tmp = alive_tmp, find_alive_iteration(alive_tmp, grammar)
    return alive


def find_reachable_iteration(reachable: set, grammar: Grammar):
    reachable_tmp = reachable.copy()
    for rule in grammar.rules:
        if rule.lhs in reachable:
            for rhs_arg in rule.rhs:
                if not rhs_arg.is_terminal():
                    reachable_tmp.add(rhs_arg)
    return reachable_tmp


def find_reachable(grammar: Grammar):
    reachable = {grammar.start}
    tmp_reachable = find_reachable_iteration(reachable, grammar)
    while reachable != tmp_reachable:
        reachable, tmp_reachable = tmp_reachable, find_reachable_iteration(tmp_reachable, grammar)
    return reachable


def normalize_grammar(grammar: Grammar):
    alive = find_alive(grammar)
    grammar.filter_by_non_terms(alive)
    reachable = find_reachable(grammar)
    grammar.filter_by_non_terms(reachable)


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


def any_chains(grammar: Grammar):
    return any(not rule.lhs.is_terminal() and len(rule.rhs) == 1 and not rule.rhs[0].is_terminal() for rule in grammar.rules)


def delete_chains(grammar: Grammar):
    while any_chains(grammar):
        for rule in grammar.rules.copy():
            if not rule.lhs.is_terminal() and len(rule.rhs) == 1 and not rule.rhs[0].is_terminal():
                for rule2 in grammar.rules:
                    if rule == rule2:
                        continue
                    if rule2.lhs == rule.lhs:
                        rule2.lhs = rule.rhs[0]
                    for i in range(len(rule2.rhs)):
                        if rule2.rhs[i] == rule.lhs:
                            rule2.rhs[i] = rule.rhs[0]
                if rule.lhs == grammar.start:
                    grammar.start = rule.rhs[0]
                grammar.rules.remove(rule)


def is_eps_rule(rule: Rule):
    return len(rule.rhs) == 1 and rule.rhs[0].is_empty()


def any_eps_rules(rules: List[Rule]):
    return not all(not is_eps_rule(rule) for rule in rules)


def delete_eps_rules(grammar: Grammar):
    rules = grammar.rules.copy()
    while any_eps_rules(rules):
        tmp_rules = []
        for rule in rules:
            if is_eps_rule(rule):
                for rule2 in rules:
                    if not is_eps_rule(rule2):
                        for i in range(2**rule2.rhs.count(rule.lhs)):
                            tmp_rhs = []
                            for symbol in rule2.rhs:
                                if symbol == rule.lhs:
                                    if i % 2:
                                        tmp_rhs.append(symbol)
                                    i //= 2
                                else:
                                    tmp_rhs.append(symbol)
                            tmp_rules.append(Rule(rule2.lhs, *tmp_rhs))
        rules = tmp_rules
    grammar.rules = rules


def is_recursive(grammar: Grammar, symbol: NonTerminal) -> bool:
    for rule in grammar.rules:
        if rule.lhs == symbol == rule.rhs[0]:
            return True
    return False


def make_hatched(symbol: NonTerminal) -> NonTerminal:
    return NonTerminal(symbol.name + "'")


def eliminate_recursions(grammar: Grammar):
    numerated = defaultdict(int)
    for i in range(len(grammar.non_terminals)):
        numerated[grammar.non_terminals[i]] = i
    cnt_hatched = 0
    for non_term_i in grammar.non_terminals.copy():
        for non_term_j in sorted(grammar.non_terminals, key=lambda x: numerated[x]):
            if numerated[non_term_j] >= numerated[non_term_i]:
                break
            for rule in grammar.rules.copy():
                if rule.lhs == non_term_i and rule.rhs[0] == non_term_j:
                    grammar.rules.remove(rule)
                    for rule_subs in grammar.rules.copy():
                        if rule_subs.lhs == non_term_j:
                            rhs = rule_subs.rhs + rule.rhs[1:]
                            grammar.rules.append(Rule(rule.lhs, *rhs))
        if is_recursive(grammar, non_term_i):
            hatched = make_hatched(non_term_i)
            for rule in grammar.rules.copy():
                if rule.lhs == non_term_i:
                    if rule.rhs[0] == non_term_i:
                        grammar.rules.append(Rule(hatched, *rule.rhs[1:], hatched))
                        grammar.rules.append(Rule(hatched, *rule.rhs[1:]))
                        grammar.rules.remove(rule)
                    else:
                        grammar.rules.append(Rule(non_term_i, *rule.rhs, hatched))
            grammar.non_terminals.append(hatched)
            cnt_hatched += 1
            numerated[hatched] = -cnt_hatched
    grammar.non_terminals.sort(key=lambda x: numerated[x])


def greibah(grammar: Grammar):
    for i in range(len(grammar.non_terminals) - 1, -1, -1):
        for j in range(i + 1, len(grammar.non_terminals)):
            for rule in grammar.rules.copy():
                if rule.lhs == grammar.non_terminals[i] and rule.rhs[0] == grammar.non_terminals[j]:
                    for rule_subs in grammar.rules.copy():
                        if rule_subs.lhs == grammar.non_terminals[j]:
                            grammar.rules.append(Rule(rule.lhs, *rule_subs.rhs, *rule.rhs[1:]))
                    grammar.rules.remove(rule)


def has_ambiguous(grammar: Grammar) -> bool:
    types = set()
    for rule in grammar.rules:
        if (rule.lhs, rule.rhs[0]) in types:
            return True
        types.add((rule.lhs, rule.rhs[0]))
    return False


def factorization(grammar: Grammar):
    while has_ambiguous(grammar):
        groups = defaultdict(list)
        for rule in grammar.rules:
            groups[(rule.lhs, rule.rhs[0])].append(rule.rhs[1:] if len(rule.rhs[1:]) else [Eps()])
        tmp_rules = []
        for group, branches in groups.items():
            if len(branches) > 1:
                hatched = make_hatched(group[0])
                tmp_rules.append(Rule(*group, hatched))
                for branch in branches:
                    tmp_rules.append(Rule(hatched, *branch))
            else:
                tmp_rules.append(Rule(*group, *(branches[0] if branches[0] != [Eps()] else [])))
        grammar.rules = tmp_rules


class Parser:
    grammar: Grammar

    def __init__(self, grammar: Grammar):
        self.grammar = deepcopy(grammar)
        delete_chains(self.grammar)
        delete_eps_rules(self.grammar)
        eliminate_recursions(self.grammar)
        greibah(self.grammar)

    def parse(self, inp: List[Terminal]) -> bool:
        return self.parse_recurse(inp, [self.grammar.start])

    def parse_recurse(self, inp: List[Terminal], current: List[Symbol]) -> bool:
        if (len(inp) == 0) != (len(current) == 0):
            return False
        if len(inp) == 0:
            return True
        if current[0].is_terminal():
            if current[0] == inp[0]:
                return self.parse_recurse(inp[1:], current[1:])
            else:
                return False
        for rule in self.grammar.rules:
            if rule.lhs == current[0]:
                if self.parse_recurse(inp, rule.rhs + current[1:]):
                    return True
        return False


########################################################################################################################
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
########################################################################################################################

def grammar_regex():
    S = NonTerminal('S')
    A = NonTerminal('A')
    P = NonTerminal('P')
    non_terminals = Grammar.NonTerminals(S, A, P)
    or_ = Terminal('|')
    and_ = Terminal(',')
    mult_ = Terminal('*')
    chr_ = Terminal('chr')
    nil_ = Terminal('nil')
    terminals = Grammar.Terminals(or_, and_, mult_, chr_, nil_)
    start = S
    rule1 = Rule(S, S, or_, S)
    rule2 = Rule(S, A)
    rule3 = Rule(A, A, and_, A)
    rule4 = Rule(A, P)
    rule5 = Rule(P, P, mult_)
    rule6 = Rule(P, chr_)
    rule7 = Rule(P, nil_)
    rules = Grammar.Rules(rule1, rule2, rule3, rule4, rule5, rule6, rule7)
    return Grammar(terminals, non_terminals, start, rules)


def grammar_epsilons():
    a = Terminal('a')
    b = Terminal('b')
    c = Terminal('c')
    d = Terminal('d')
    terminals = Grammar.Terminals(a)
    A = NonTerminal('A')
    B = NonTerminal('B')
    non_terminals = Grammar.NonTerminals(A, B)
    rule1 = Rule(A, a)
    rule2 = Rule(B, b)
    rule3 = Rule(A, B, c)
    rule4 = Rule(B, A, d)
    rules = Grammar.Rules(rule1, rule2, rule3, rule4)
    start = A
    return Grammar(terminals, non_terminals, start, rules)


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
    g = grammar_regex()
    or_ = Terminal('|')
    and_ = Terminal(',')
    mult_ = Terminal('*')
    chr_ = Terminal('chr')
    nil_ = Terminal('nil')
    parser = Parser(g)
    print(g)
    print(parser.grammar)
    print(parser.parse([chr_, or_, chr_, mult_]))
