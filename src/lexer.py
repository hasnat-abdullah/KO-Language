import re

class Lexer(object):
    def __init__(self,source_code):
        self.source_code = source_code

    def tokenize(self):
        tokens = []

        source_code = self.source_code.split()

        source_index = 0

        while source_index < len(source_code):
            word = source_code[source_index]

            # variable identification
            if word == "সংখ্যা": tokens.append(["VAR_DECLERATION", word])

            # identifier token recognizer
            elif re.match('[a-z]',word) or re.match('[A-Z]',word) or re.match('[\u0980-\u09E5]',word) or re.match('[\u09F0-\u09FF]',word):
                if word[len(word)-1] == ';':
                    tokens.append(['IDENTIFIER',word[0:len(word)-1]])
                else:
                    tokens.append(['IDENTIFIER',word])

            # Integer identifier
            elif re.match('[০-৯]',word):
                if word[len(word)-1] == ';':
                    tokens.append(['INTEGER',word[0:len(word)-1]])
                else:
                    tokens.append(['INTEGER',word])

            # Operator recognizer
            elif word in "+-=/*":
                tokens.append(['OPERATOR', word])

            # ending statement
            if word[len(word) - 1] == ';':
                tokens.append(['STATEMENT_END', ';'])
            source_index += 1

        print(tokens)

        return tokens