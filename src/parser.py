from Objects.varObject import VariableObject

class Parser(object):
    def __init__(self, tokens):
        self.tokens = tokens
        self.token_index = 0
        self.transpiled_code = ""

    def parse(self):
        while self.token_index <len(self.tokens):
            token_type = self.tokens[self.token_index][0]
            token_value = self.tokens[self.token_index][1]

            # variable detection
            if token_type == "VAR_DECLERATION" and token_value == "সংখ্যা":
                self.parse_variable_decleration(self.tokens[self.token_index:len(self.tokens)])

            self.token_index +=1
        print(self.transpiled_code)

    def parse_variable_decleration(self,token_stream):
        tokens_checked = 0
        name = ""
        operator = ""
        value = ""

        for token in range(0,len(token_stream)):
            token_type = token_stream[tokens_checked][0]
            token_value = token_stream[tokens_checked][1]

            if token_type == "STATEMENT_END":
                break

            if token == 1 and token_type == "IDENTIFIER":
                name = token_value
            elif token ==1 and token_type != "IDENTIFIER":
                print("দুঃখিত! ভেরিয়েবলের নাম ভুল হয়েছে। '" + token_value+"'")
                quit()

            elif token == 2 and token_type == "OPERATOR":
                operator = token_value
            elif token == 2 and token_type != "OPERATOR":
                print("দুঃখিত! অপারেটর নাম ভুল হয়েছে। এটা '=' হবে।")
                quit()

            elif token == 3 and token_type in ['IDENTIFIER','STRING','INTEGER']:
                value = token_value
            elif token == 3 and token_type not in ['IDENTIFIER','STRING','INTEGER']:
                print("দুঃখিত! ভেরিয়েবলের মান ভুল হয়েছে। '" + token_value+"'")
                quit()

            tokens_checked += 1

        varObj = VariableObject()
        self.transpiled_code += varObj.transpile(name, operator, value)

        self.token_index += tokens_checked


