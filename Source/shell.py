import ko

from Source.ko import En2BnNum

while True:
    text = input('ko > ')
    result, error = ko.run('Working File', text)

    if error: print(error.asString())
    else: print(result)