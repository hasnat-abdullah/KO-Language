import lexer
import parser

def main():
    content = ""
    with open('test.ko', 'r') as file:
        content = file.read()

    ###########
    ## Lexer ##
    ###########

    lex = lexer.Lexer(content)
    tokens = lex.tokenize()

    ############
    ## Parser ##
    ############
    parse = parser.Parser(tokens)
    objs = parse.parse()


main()