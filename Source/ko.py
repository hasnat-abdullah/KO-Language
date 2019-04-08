

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
        return result


class IllegalCharError(Error):
    def __init__(self, posStart, posEnd, details):
        super().__init__(posStart, posEnd, 'দুঃখিত, ভুল অক্ষর', details)


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

    def advance(self, currentChar):
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

class Token:
    def __init__(self, type_, value=None):
        self.type = type_
        self.value = value

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
                tokens.append(Token(KO_PLUS))
                self.advance()
            elif self.currentChar == '-':
                tokens.append(Token(KO_MINUS))
                self.advance()
            elif self.currentChar == '*':
                tokens.append(Token(KO_MUL))
                self.advance()
            elif self.currentChar == '/':
                tokens.append(Token(KO_DIV))
                self.advance()
            elif self.currentChar == '(':
                tokens.append(Token(KO_LPAREN))
                self.advance()
            elif self.currentChar == ')':
                tokens.append(Token(KO_RPAREN))
                self.advance()
            else:
                posStart = self.pos.copy()
                char = self.currentChar
                self.advance()
                return [], IllegalCharError(posStart, self.pos, "'" + char + "'")

        return tokens, None

    def makeNumber(self):
        numStr = ''
        dotCount = 0

        while self.currentChar != None and self.currentChar in NUMBER + '.':
            if self.currentChar == '.':
                if dotCount == 1: break
                dotCount += 1
                numStr += '.'
            else:
                numStr += self.currentChar
            self.advance()

        if dotCount == 0:
            return Token(KO_INT, int(numStr))
        else:
            return Token(KO_FLOAT, float(numStr))

    #######################################
    # RUN
    #######################################

def run(fn, text):
    lexer = Lexer(fn, text)
    tokens, error = lexer.makeTokens()

    return tokens, error