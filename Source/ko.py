from strings_with_arrows import *

NUMBER = '০১২৩৪৫৬৭৮৯0123456789'

######################
###     ERRORS     ###
######################

class Error:
    def __init__(self, posStart, posEnd, errorName, details):
        self.posStart = posStart
        self.posEnd = posEnd
        self.errorName = errorName
        self.details = details

    def asString(self):
        result = f'{self.errorName}: {self.details}\n'
        result += f'ফাইলঃ {self.posStart.fn}, লাইনঃ {self.posStart.ln + 1}'
        result += '\n\n' + string_with_arrows(self.posStart.ftxt, self.posStart, self.posEnd)
        return result


class IllegalCharError(Error):
    def __init__(self, posStart, posEnd, details):
        super().__init__(posStart, posEnd, 'দুঃখিত, ভুল অক্ষর', details)


class InvalidSyntaxError(Error):
    def __init__(self, posStart, posEnd, details=''):
        super().__init__(posStart, posEnd, 'দুঃখিত, লিখতে ভুল হয়েছে', details)

######################
###   POSITIONS    ###
######################
class Position:
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt

    def advance(self, currentChar = None):
        self.idx += 1
        self.col += 1

        if currentChar == '\n':
            self.ln += 1
            self.col = 0

        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)



######################
###     TOKENS     ###
######################

KO_INT		= 'সংখ্যা'
KO_FLOAT    = 'দশমিক সংখ্যা'
KO_PLUS     = 'যোগ'
KO_MINUS    = 'বিয়োগ'
KO_MUL      = 'গুন'
KO_DIV      = 'ভাগ'
KO_LPAREN   = 'বাম ব্রাকেট'
KO_RPAREN   = 'ডান ব্রাকেট'
KO_EOF      = 'ফাইল শেষ'

class Token:
    def __init__(self, type_, value=None, posStart=None, posEnd=None):
        self.type = type_
        self.value = value

        if posStart:
            self.posStart = posStart.copy()
            self.posEnd = posStart.copy()
            self.posEnd.advance()

        if posEnd:
            self.posEnd = posEnd

    def __repr__(self):
        if self.value:
            return f'{self.type}:{self.value}'
        else:
            return f'{self.type}'



######################
###      LEXER     ###
######################

class Lexer:
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.currentChar = None
        self.advance()

    def advance(self):
        self.pos.advance(self.currentChar)
        self.currentChar = self.text[self.pos.idx] if self.pos.idx <len(self.text) else None

    def makeTokens(self):
        tokens = []

        while self.currentChar != None:
            if self.currentChar in ' \t':
                self.advance()
            elif self.currentChar in NUMBER:
                tokens.append(self.makeNumber())
            elif self.currentChar == '+':
                tokens.append(Token(KO_PLUS, posStart=self.pos))
                self.advance()
            elif self.currentChar == '-':
                tokens.append(Token(KO_MINUS, posStart=self.pos))
                self.advance()
            elif self.currentChar == '*':
                tokens.append(Token(KO_MUL, posStart=self.pos))
                self.advance()
            elif self.currentChar == '/':
                tokens.append(Token(KO_DIV, posStart=self.pos))
                self.advance()
            elif self.currentChar == '(':
                tokens.append(Token(KO_LPAREN, posStart=self.pos))
                self.advance()
            elif self.currentChar == ')':
                tokens.append(Token(KO_RPAREN, posStart=self.pos))
                self.advance()
            else:
                posStart = self.pos.copy()
                char = self.currentChar
                self.advance()
                return [], IllegalCharError(posStart, self.pos, "'" + char + "'")

        tokens.append(Token(KO_EOF, posStart=self.pos))
        return tokens, None

    def makeNumber(self):
        numStr = ''
        dotCount = 0
        posStart = self.pos.copy()
        while self.currentChar != None and self.currentChar in NUMBER + '.':
            if self.currentChar == '.':
                if dotCount == 1: break
                dotCount += 1
                numStr += '.'
            else:
                numStr += self.currentChar
            self.advance()

        if dotCount == 0:
            return Token(KO_INT, int(numStr), posStart, self.pos)
        else:
            return Token(KO_FLOAT, float(numStr), posStart, self.pos)


######################
###      NODES     ###
######################


class NumberNode:
    def __init__(self, tok):
        self.tok = tok

    def __repr__(self):
        return f'{self.tok}'

class BinOpNode:
    def __init__(self, leftNode, opTok, rightNode):
        self.leftNode = leftNode
        self.opTok = opTok
        self.rightNode = rightNode

    def __repr__(self):
        return f'({self.leftNode}, {self.opTok}, {self.rightNode})'


class UnaryOpNode:
    def __init__(self, opTok, node):
        self.opTok = opTok
        self.node = node

    def __repr__(self):
        return f'({self.opTok}, {self.node})'


######################
### PARSER RESULT  ###
######################


class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None

    def register(self, res):
        if isinstance(res, ParseResult):
            if res.error: self.error = res.error
            return res.node
        return res

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        self.error = error
        return self


######################
###     PARSER     ###
######################


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tokIdx = -1
        self.advance()

    def advance(self, ):
        self.tokIdx += 1
        if self.tokIdx < len(self.tokens):
            self.currentTok = self.tokens[self.tokIdx]
        return self.currentTok

    def parse(self):
        res = self.expr()
        if not res.error and self.currentTok.type != KO_EOF:
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd, "সম্ভবত '+', '-', '*' বা '/' হবে "
            ))
        return res

    ###################################

    def factor(self):
        res = ParseResult()
        tok = self.currentTok

        if tok.type in (KO_PLUS, KO_MINUS):
            res.register(self.advance())
            factor = res.register(self.factor())
            if res.error:
                return res
            return res.success(UnaryOpNode(tok, factor))

        elif tok.type in (KO_INT, KO_FLOAT):
            res.register(self.advance())
            return res.success(NumberNode(tok))

        elif tok.type == KO_LPAREN:
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error: return res
            if self.currentTok.type == KO_RPAREN:
                res.register(self.advance())
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(
                    self.currentTok.posStart, self.currentTok.posEnd,
                    "সম্ভবত ')' হবে!"
                ))

        return res.failure(InvalidSyntaxError(
            tok.posStart, tok.posEnd,
            "সম্ভবত সংখ্যা অথবা দশমিক সংখ্যা হবে !"
        ))

    def term(self):
        return self.binOp(self.factor, (KO_MUL, KO_DIV))

    def expr(self):
        return self.binOp(self.term, (KO_PLUS, KO_MINUS))

    ###################################

    def binOp(self, func, ops):
        res = ParseResult()
        left = res.register(func())
        if res.error: return res

        while self.currentTok.type in ops:
            opTok = self.currentTok
            res.register(self.advance())
            right = res.register(func())
            if res.error: return res
            left = BinOpNode(left, opTok, right)

        return res.success(left)


######################
###       RUN      ###
######################

def run(fn, text):
    lexer = Lexer(fn, text)
    tokens, error = lexer.makeTokens()
    if error: return None, error

    # Generate Abstract syntax Tree
    parser = Parser(tokens)
    ast = parser.parse()

    return ast.node, ast.error