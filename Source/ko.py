from strings_with_arrows import *

NUMBER = '০১২৩৪৫৬৭৮৯'
LETTERS = 'ঁংঃঅআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরল঴঵শষসহ়ঽািীুূৃৄেৈোৌ্ৎৗড়ঢ়য়ৠৰৱ৹৺৻'
LETTERS_NUMBERS = LETTERS + NUMBER


def En2BnNum(enNum):
    strNum = str(enNum)
    for i in range(0, len(strNum)):
        if strNum[i] == '1':
            strNum = strNum.replace(strNum[i], '১')
        elif strNum[i] == '2':
            strNum = strNum.replace(strNum[i], '২')
        elif strNum[i] == '3':
            strNum = strNum.replace(strNum[i], '৩')
        elif strNum[i] == '4':
            strNum = strNum.replace(strNum[i], '৪')
        elif strNum[i] == '5':
            strNum = strNum.replace(strNum[i], '৫')
        elif strNum[i] == '6':
            strNum = strNum.replace(strNum[i], '৬')
        elif strNum[i] == '7':
            strNum = strNum.replace(strNum[i], '৭')
        elif strNum[i] == '8':
            strNum = strNum.replace(strNum[i], '৮')
        elif strNum[i] == '9':
            strNum = strNum.replace(strNum[i], '৯')
        elif strNum[i] == '0':
            strNum = strNum.replace(strNum[i], '০')
        else:
            pass
    return strNum


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

class ExpectedCharError(Error):
    def __init__(self, posStart, posEnd, details):
        super().__init__(posStart, posEnd, 'দুঃখিত, কাঙ্ক্ষিত অক্ষর', details)


class InvalidSyntaxError(Error):
    def __init__(self, posStart, posEnd, details=''):
        super().__init__(posStart, posEnd, 'দুঃখিত, লিখতে ভুল হয়েছে', details)


class RTError(Error):
    def __init__(self, posStart, posEnd, details, context):
        super().__init__(posStart, posEnd, 'Runtime Error', details)
        self.context = context

    def asString(self):
        result = self.generateTraceback()
        result += f'{self.errorName}: {self.details}'
        result += '\n\n' + string_with_arrows(self.posStart.ftxt, self.posStart, self.posEnd)
        return result

    def generateTraceback(self):
        result = ''
        pos = self.posStart
        ctx = self.context

        while ctx:
            result = f'  File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.displayName}\n' + result
            pos = ctx.parentEntryPos
            ctx = ctx.parent

        return 'Traceback (most recent call last):\n' + result


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

    def advance(self, currentChar=None):
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

KO_INT = 'সংখ্যা'
KO_FLOAT = 'দশমিক সংখ্যা'
KO_IDENTIFIER = 'শনাক্তকারী'
KO_KEYWORD = 'কিওয়ার্ড'
KO_PLUS = 'যোগ'
KO_MINUS = 'বিয়োগ'
KO_MUL = 'গুন'
KO_DIV = 'ভাগ'
KO_POW = 'পাওয়ার'
KO_EQ = 'সমান'
KO_MOD = 'ভাগশেষ'
KO_LPAREN = 'বাম ব্রাকেট'
KO_RPAREN = 'ডান ব্রাকেট'
KO_EE = 'সমান সমান'
KO_NE = 'সমান নয়'
KO_LT = 'ছোট'
KO_GT = 'বড়'
KO_LTE = 'ছোট অথবা সমান'
KO_GTE = 'বড় অথবা সমান'
KO_EOF = 'ফাইল শেষ'


KEYWORDS = [
    'ধরি',
    'এবং',
    'অথবা',
    'নয়',
    'যদি',
    'তাহলে',
    'অথবা যদি',
    'নাহলে',
    'লুপ',
    'থেকে',
    'বৃদ্ধি',
    'যখন'
]


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

    def matches(self, type_, value):
        return self.type == type_ and self.value == value

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
        self.currentChar = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def makeTokens(self):
        tokens = []

        while self.currentChar != None:
            if self.currentChar in ' \t':
                self.advance()
            elif self.currentChar in NUMBER:
                tokens.append(self.makeNumber())
            elif self.currentChar in LETTERS:
                tokens.append(self.makeIdentifier())
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
            elif self.currentChar == '^':
                tokens.append(Token(KO_POW, posStart=self.pos))
                self.advance()
            elif self.currentChar == '(':
                tokens.append(Token(KO_LPAREN, posStart=self.pos))
                self.advance()
            elif self.currentChar == ')':
                tokens.append(Token(KO_RPAREN, posStart=self.pos))
                self.advance()
            elif self.currentChar == '!':
                token, error = self.makeNotEquals()
                if error: return [], error
                tokens.append(token)
            elif self.currentChar == '=':
                tokens.append(self.makeEquals())
            elif self.currentChar == '<':
                tokens.append(self.makeLessThan())
            elif self.currentChar == '>':
                tokens.append(self.makeGreaterThan())
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

            numStr += self.currentChar
            self.advance()

        if dotCount == 0:
            return Token(KO_INT, int(numStr), posStart, self.pos)
        else:
            return Token(KO_FLOAT, float(numStr), posStart, self.pos)

    def makeIdentifier(self):
        idStr = ''
        posStart = self.pos.copy()

        while self.currentChar != None and self.currentChar in LETTERS_NUMBERS + '_':
            idStr += self.currentChar
            self.advance()

        tokType = KO_KEYWORD if idStr in KEYWORDS else KO_IDENTIFIER
        return Token(tokType, idStr, posStart, self.pos)

    def makeNotEquals(self):
        posStart = self.pos.copy()
        self.advance()

        if self.currentChar == '=':
            self.advance()
            return Token(KO_NE, posStart=posStart, posEnd=self.pos), None

        self.advance()
        return None, ExpectedCharError(posStart, self.pos, "'=' (পরে '!')")

    def makeEquals(self):
        tokType = KO_EQ
        posStart = self.pos.copy()
        self.advance()

        if self.currentChar == '=':
            self.advance()
            tokType = KO_EE

        return Token(tokType, posStart=posStart, posEnd=self.pos)

    def makeLessThan(self):
        tokType = KO_LT
        posStart = self.pos.copy()
        self.advance()

        if self.currentChar == '=':
            self.advance()
            tokType = KO_LTE

        return Token(tokType, posStart=posStart, posEnd=self.pos)

    def makeGreaterThan(self):
        tokType = KO_GT
        posStart = self.pos.copy()
        self.advance()

        if self.currentChar == '=':
            self.advance()
            tokType = KO_GTE

        return Token(tokType, posStart=posStart, posEnd=self.pos)


######################
###      NODES     ###
######################


class NumberNode:
    def __init__(self, tok):
        self.tok = tok
        self.posStart = self.tok.posStart
        self.posEnd = self.tok.posEnd

    def __repr__(self):
        return f'{self.tok}'


class VarAccessNode:
    def __init__(self, varNameTok):
        self.varNameTok = varNameTok

        self.posStart = self.varNameTok.posStart
        self.posEnd = self.varNameTok.posEnd


class VarAssignNode:
    def __init__(self, varNameTok, valueNode):
        self.varNameTok = varNameTok
        self.valueNode = valueNode

        self.posStart = self.varNameTok.posStart
        self.posEnd = self.valueNode.posEnd


class BinOpNode:
    def __init__(self, leftNode, opTok, rightNode):
        self.leftNode = leftNode
        self.opTok = opTok
        self.rightNode = rightNode

        self.posStart = self.leftNode.posStart
        self.posEnd = self.rightNode.posEnd

    def __repr__(self):
        return f'({self.leftNode}, {self.opTok}, {self.rightNode})'


class UnaryOpNode:
    def __init__(self, opTok, node):
        self.opTok = opTok
        self.node = node
        self.posStart = self.opTok.posStart
        self.posEnd = node.posEnd

    def __repr__(self):
        return f'({self.opTok}, {self.node})'


class IfNode:
    def __init__(self, cases, elseCase):
        self.cases = cases
        self.elseCase = elseCase

        self.posStart = self.cases[0][0].posStart
        self.posEnd = (self.elseCase or self.cases[len(self.cases) - 1][0]).posEnd


class ForNode:
    def __init__(self, varNameTok, startValueNode, endValueNode, stepValueNode, bodyNode):
        self.varNameTok = varNameTok
        self.startValueNode = startValueNode
        self.endValueNode = endValueNode
        self.stepValueNode = stepValueNode
        self.bodyNode = bodyNode

        self.posStar = self.varNameTok.posStart
        self.posEnd = self.bodyNode.posEnd


class WhileNode:
    def __init__(self, conditionNode, bodyNode):
        self.conditionNode = conditionNode
        self.bodyNode = bodyNode

        self.posStart = self.conditionNode.posStart
        self.posEnd = self.bodyNode.posEnd

######################
### PARSER RESULT  ###
######################


class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.advanceCount = 0

    def registerAdvancement(self):
        self.advanceCount += 1

    def register(self, res):
        self.advanceCount += res.advanceCount
        if res.error: self.error = res.error
        return res.node

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        if not self.error or self.advanceCount == 0:
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
                self.currentTok.posStart, self.currentTok.posEnd, "সম্ভবত '+', '-', '*', '/', '^', '==', '!=', '<', '>', <=', '>=', 'এবং' বা 'অথবা' হবে। "
            ))
        return res

    ###################################

    def ifExpr(self):
        res = ParseResult()
        cases = []
        elseCase = None

        if not self.currentTok.matches(KO_KEYWORD, 'যদি'):
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                f"সম্ভবত 'যদি' হবে। "
            ))

        res.registerAdvancement()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res

        if not self.currentTok.matches(KO_KEYWORD, 'তাহলে'):
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                f"সম্ভবত 'তাহলে' হবে।'"
            ))

        res.registerAdvancement()
        self.advance()

        expr = res.register(self.expr())
        if res.error: return res
        cases.append((condition, expr))

        while self.currentTok.matches(KO_KEYWORD, 'অথবা যদি'):
            res.registerAdvancement()
            self.advance()

            condition = res.register(self.expr())
            if res.error: return res

            if not self.currentTok.matches(KO_KEYWORD, 'তাহলে'):
                return res.failure(InvalidSyntaxError(
                    self.currentTok.posStart, self.currentTok.posEnd,
                    f"সম্ভবত 'তাহলে' হবে।"
                ))

            res.registerAdvancement()
            self.advance()

            expr = res.register(self.expr())
            if res.error: return res
            cases.append((condition, expr))

        if self.currentTok.matches(KO_KEYWORD, 'নাহলে'):
            res.registerAdvancement()
            self.advance()

            elseCase = res.register(self.expr())
            if res.error: return res

        return res.success(IfNode(cases, elseCase))

    def forExpr(self):
        res = ParseResult()

        if not self.currentTok.matches(KO_KEYWORD, 'লুপ'):
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                f"সম্ভবত 'লুপ' হবে"
            ))

        res.registerAdvancement()
        self.advance()

        if self.currentTok.type != KO_IDENTIFIER:
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                f"সম্ভবত identifier হবে"
            ))

        varName = self.currentTok
        res.registerAdvancement()
        self.advance()

        if self.currentTok.type != KO_EQ:
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                f"সম্ভবত '=' হবে"
            ))

        res.registerAdvancement()
        self.advance()

        startValue = res.register(self.expr())
        if res.error: return res

        if not self.currentTok.matches(KO_KEYWORD, 'থেকে'):
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                f"সম্ভবত 'থেকে' হবে"
            ))

        res.registerAdvancement()
        self.advance()

        endValue = res.register(self.expr())
        if res.error: return res

        if self.currentTok.matches(KO_KEYWORD, 'বৃদ্ধি'):
            res.registerAdvancement()
            self.advance()

            stepValue = res.register(self.expr())
            if res.error: return res
        else:
            stepValue = None

        if not self.currentTok.matches(KO_KEYWORD, 'তাহলে'):
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                f"সম্ভবত 'তাহলে' হবে"
            ))

        res.registerAdvancement()
        self.advance()

        body = res.register(self.expr())
        if res.error: return res

        return res.success(ForNode(varName, startValue, endValue, stepValue, body))

    def whileExpr(self):
        res = ParseResult()

        if not self.currentTok.matches(KO_KEYWORD, 'যখন'):
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                f"সম্ভবত 'যখন' হবে"
            ))

        res.registerAdvancement()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res

        if not self.currentTok.matches(KO_KEYWORD, 'তাহলে'):
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                f"সম্ভবত 'তাহলে' হবে"
            ))

        res.registerAdvancement()
        self.advance()

        body = res.register(self.expr())
        if res.error: return res

        return res.success(WhileNode(condition, body))

    def atom(self):
        res = ParseResult()
        tok = self.currentTok

        if tok.type in (KO_INT, KO_FLOAT):
            res.registerAdvancement()
            self.advance()
            return res.success(NumberNode(tok))

        elif tok.type == KO_IDENTIFIER:
            res.registerAdvancement()
            self.advance()
            return res.success(VarAccessNode(tok))

        elif tok.type == KO_LPAREN:
            res.registerAdvancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            if self.currentTok.type == KO_RPAREN:
                res.registerAdvancement()
                self.advance()
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(
                    self.currentTok.posStart, self.currentTok.posEnd,
                    "সম্ভবত ')' হবে!"
                ))

        elif tok.matches(KO_KEYWORD, 'যদি'):
            ifExpr = res.register(self.ifExpr())
            if res.error: return res
            return res.success(ifExpr)

        elif tok.matches(KO_KEYWORD, 'লুপ'):
            forExpr = res.register(self.forExpr())
            if res.error: return res
            return res.success(forExpr)

        elif tok.matches(KO_KEYWORD, 'যখন'):
            whileExpr = res.register(self.whileExpr())
            if res.error: return res
            return res.success(whileExpr)

        return res.failure(InvalidSyntaxError(
            tok.posStart, tok.posEnd,
            "সম্ভবত সংখ্যা, দশমিক সংখ্যা, শনাক্তকারী, +, - অথবা (  হবে !"
        ))

    def power(self):
        return self.binOp(self.atom, (KO_POW,), self.factor)

    def factor(self):
        res = ParseResult()
        tok = self.currentTok

        if tok.type in (KO_PLUS, KO_MINUS):
            res.registerAdvancement()
            self.advance()
            factor = res.register(self.factor())
            if res.error:
                return res
            return res.success(UnaryOpNode(tok, factor))

        return self.power()

    def term(self):
        return self.binOp(self.factor, (KO_MUL, KO_DIV))

    def arithExpr(self):
        return self.binOp(self.term, (KO_PLUS, KO_MINUS))

    def compExpr(self):
        res = ParseResult()

        if self.currentTok.matches(KO_KEYWORD, 'নয়'):
            opTok = self.currentTok
            res.registerAdvancement()
            self.advance()

            node = res.register(self.compExpr())
            if res.error: return res
            return res.success(UnaryOpNode(opTok, node))

        node = res.register(self.binOp(self.arithExpr, (KO_EE, KO_NE, KO_LT, KO_GT, KO_LTE, KO_GTE)))

        if res.error:
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                "সম্ভবত সংখ্যা, দশমিক সংখ্যা, শনাক্তকারী, +, -, ( অথবা 'নয়' হবে !"
            ))

        return res.success(node)

    def expr(self):
        res = ParseResult()

        if self.currentTok.matches(KO_KEYWORD, 'ধরি'):
            res.registerAdvancement()
            self.advance()

            if self.currentTok.type != KO_IDENTIFIER:
                return res.failure(InvalidSyntaxError(
                    self.currentTok.posStart, self.currentTok.posEnd,
                    "সম্ভবত নির্দেশক হবে"
                ))

            varName = self.currentTok
            res.registerAdvancement()
            self.advance()

            if self.currentTok.type != KO_EQ:
                return res.failure(InvalidSyntaxError(
                    self.currentTok.posStart, self.currentTok.posEnd,
                    "সম্ভবত '=' হবে ।"
                ))

            res.registerAdvancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            return res.success(VarAssignNode(varName, expr))

        node = res.register(self.binOp(self.compExpr, ((KO_KEYWORD, "এবং"),(KO_KEYWORD, "অথবা"))))

        if res.error:
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                "সম্ভবত 'ধরি', সংখ্যা, দশমিক সংখ্যা, শনাক্তকারী, '+', '-' অথবা '(' হবে ।"
            ))

        return res.success(node)

    ###################################

    def binOp(self, func_a, ops, func_b=None):
        if func_b == None:
            func_b = func_a
        res = ParseResult()
        left = res.register(func_a())
        if res.error: return res

        while self.currentTok.type in ops or (self.currentTok.type, self.currentTok.value) in ops:
            opTok = self.currentTok
            res.registerAdvancement()
            self.advance()
            right = res.register(func_b())
            if res.error: return res
            left = BinOpNode(left, opTok, right)

        return res.success(left)


######################
### RUNNTIME RESULT ##
######################


class RTResult:
    def __init__(self):
        self.value = None
        self.error = None

    def register(self, res):
        if res.error: self.error = res.error
        return res.value

    def success(self, value):
        self.value = value
        return self

    def failure(self, error):
        self.error = error
        return self


######################
###     VALUE      ###
######################


class Number:
    def __init__(self, value):
        self.value = value
        self.setPos()
        self.setContext()

    def setPos(self, posStart=None, posEnd=None):
        self.posStart = posStart
        self.posEnd = posEnd
        return self

    def setContext(self, context=None):
        self.context = context
        return self

    def addedTo(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).setContext(self.context), None

    def subbedBy(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).setContext(self.context), None

    def multedBy(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).setContext(self.context), None

    def divedBy(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(
                    other.posStart, other.posEnd, 'দুঃখিত, শূন্য দিয়ে ভাগ করা যায় না ! ', self.context
                )

            return Number(self.value / other.value).setContext(self.context), None

    def powedBy(self, other):
        if isinstance(other, Number):
            return Number(self.value ** other.value).setContext(self.context), None

    def getComparisonEq(self, other):
        if isinstance(other, Number):
            return Number(int(self.value == other.value)).setContext(self.context), None

    def getComparisonNe(self, other):
        if isinstance(other, Number):
            return Number(int(self.value != other.value)).setContext(self.context), None

    def getComparisonLt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value < other.value)).setContext(self.context), None

    def getComparisonGt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value > other.value)).setContext(self.context), None

    def getComparisonLte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value <= other.value)).setContext(self.context), None

    def getComparisonGte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value >= other.value)).setContext(self.context), None

    def andedBy(self, other):
        if isinstance(other, Number):
            return Number(int(self.value and other.value)).setContext(self.context), None

    def oredBy(self, other):
        if isinstance(other, Number):
            return Number(int(self.value or other.value)).setContext(self.context), None

    def notted(self):
        return Number(1 if self.value == 0 else 0).setContext(self.context), None

    def copy(self):
        copy = Number(self.value)
        copy.setPos(self.posStart, self.posEnd)
        copy.setContext(self.context)
        return copy

    def isTrue(self):
        return self.value != 0

    def __repr__(self):
        return str(self.value)


######################
###     CONTEXT    ###
######################


class Context:
    def __init__(self, displayName, parent=None, parentEntryPos=None):
        self.displayName = displayName
        self.parent = parent
        self.parentEntryPos = parentEntryPos
        self.symbolTable = None


######################
###  SYMBOL TABLE  ###
######################

class SymbolTable:
    def __init__(self):
        self.symbols = {}
        self.parent = None

    def get(self, name):
        value = self.symbols.get(name, None)
        if value == None and self.parent:
            return self.parent.get(name)
        return value

    def set(self, name, value):
        self.symbols[name] = value

    def remove(self, name):
        del self.symbols[name]


######################
###  INTERPRETER   ###
######################


class Interpreter:
    def visit(self, node, context):
        methodName = f'visit{type(node).__name__}'
        method = getattr(self, methodName, self.noVisitMethod)
        return method(node, context)

    def noVisitMethod(self, node, context):
        raise Exception(f'No visit{type(node).__name__} method defined')

    ###################################

    def visitNumberNode(self, node, context):
        return RTResult().success(
            Number(node.tok.value).setContext(context).setPos(node.posStart, node.posEnd)
        )

    def visitVarAccessNode(self, node, context):
        res = RTResult()
        varName = node.varNameTok.value
        value = context.symbolTable.get(varName)

        if not value:
            return res.failure(RTError(
                node.posStart, node.posEnd,
                f"'{varName}' পূর্বে উল্লেখ করা হয় নি ।",
                context
            ))

        value = value.copy().setPos(node.posStart, node.posEnd)
        return res.success(value)

    def visitVarAssignNode(self, node, context):
        res = RTResult()
        varName = node.varNameTok.value
        value = res.register(self.visit(node.valueNode, context))
        if res.error: return res

        context.symbolTable.set(varName, value)
        return res.success(value)

    def visitBinOpNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.leftNode, context))
        if res.error: return res
        right = res.register(self.visit(node.rightNode, context))
        if res.error: return res

        if node.opTok.type == KO_PLUS:
            result, error = left.addedTo(right)
        elif node.opTok.type == KO_MINUS:
            result, error = left.subbedBy(right)
        elif node.opTok.type == KO_MUL:
            result, error = left.multedBy(right)
        elif node.opTok.type == KO_DIV:
            result, error = left.divedBy(right)
        elif node.opTok.type == KO_POW:
            result, error = left.powedBy(right)
        elif node.opTok.type == KO_EE:
            result, error = left.getComparisonEq(right)
        elif node.opTok.type == KO_NE:
            result, error = left.getComparisonNe(right)
        elif node.opTok.type == KO_LT:
            result, error = left.getComparisonLt(right)
        elif node.opTok.type == KO_GT:
            result, error = left.getComparisonGt(right)
        elif node.opTok.type == KO_LTE:
            result, error = left.getComparisonLte(right)
        elif node.opTok.type == KO_GTE:
            result, error = left.getComparisonGte(right)
        elif node.opTok.matches(KO_KEYWORD, 'এবং'):
            result, error = left.andedBy(right)
        elif node.opTok.matches(KO_KEYWORD, 'অথবা'):
            result, error = left.oredBy(right)

        if error:
            return res.failure(error)
        else:
            return res.success(result.setPos(node.posStart, node.posEnd))

    def visitUnaryOpNode(self, node, context):
        res = RTResult()
        number = res.register(self.visit(node.node, context))
        if res.error: return res

        error = None

        if node.opTok.type == KO_MINUS:
            number, error = number.multedBy(Number(-1))
        elif node.opTok.matches(KO_KEYWORD, 'নয়'):
            number, error = number.notted()

        if error:
            return res.failure(error)
        else:
            return res.success(number.setPos(node.posStart, node.posEnd))

    def visitIfNode(self, node, context):
        res = RTResult()

        for condition, expr in node.cases:
            conditionValue = res.register(self.visit(condition, context))
            if res.error: return res

            if conditionValue.isTrue():
                exprValue = res.register(self.visit(expr, context))
                if res.error: return res
                return res.success(exprValue)

        if node.elseCase:
            elseValue = res.register(self.visit(node.elseCase, context))
            if res.error: return res
            return res.success(elseValue)

        return res.success(None)

    def visitForNode(self, node, context):
        res = RTResult()

        startValue = res.register(self.visit(node.startValueNode, context))
        if res.error: return res

        endValue = res.register(self.visit(node.endValueNode, context))
        if res.error: return res

        if node.stepValueNode:
            stepValue = res.register(self.visit(node.stepValueNode, context))
            if res.error: return res
        else:
            stepValue = Number(1)

        i = startValue.value

        if stepValue.value >= 0:
            condition = lambda: i < endValue.value
        else:
            condition = lambda: i > endValue.value

        while condition():
            context.symbolTable.set(node.varNameTok.value, Number(i))
            i += stepValue.value

            res.register(self.visit(node.bodyNode, context))
            if res.error: return res

        return res.success(None)

    def visitWhileNode(self, node, context):
        res = RTResult()

        while True:
            condition = res.register(self.visit(node.conditionNode, context))
            if res.error: return res

            if not condition.isTrue(): break
            res.register(self.visit(node.bodyNode, context))
            if res.error: return res

        return res.success(None)


######################
###       RUN      ###
######################

globalSymbolTable = SymbolTable()
globalSymbolTable.set("null", Number(0))
globalSymbolTable.set("মিথ্যা", Number(0))
globalSymbolTable.set("সত্য", Number(1))


def run(fn, text):
    lexer = Lexer(fn, text)
    tokens, error = lexer.makeTokens()
    if error: return None, error

    # Generate Abstract syntax Tree
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error

    # Run program
    interpreter = Interpreter()
    context = Context('<program>')
    context.symbolTable = globalSymbolTable
    result = interpreter.visit(ast.node, context)

    return En2BnNum(result.value), result.error