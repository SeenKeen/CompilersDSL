from abc import abstractmethod


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


def grammar_expressions():
    plus = Terminal('+')
    aster = Terminal('*')
    const = Terminal('C')
    brace_op = Terminal('(')
    brace_cl = Terminal(')')
    terminals = Grammar.Terminals(plus, aster, const, brace_op, brace_cl)
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
    rules = Grammar.Rules(rule1, rule2, rule3, rule4, rule5, rule6)
    return Grammar(terminals, non_terminals, start, rules)


if __name__ == '__main__':
    g = grammar_expressions()
    normalize_grammar(g)
    print(g)
