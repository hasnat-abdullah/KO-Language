import ko

while True:
    text = input('ko > ')
    result, error = ko.run('Working File', text)

    if error: print(error.as_string())
    elif result: print(result)