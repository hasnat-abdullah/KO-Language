import ko

while True:
    text = input('ko > ')
    if text.strip() == "": continue
    result, error = ko.run('<stdin>', text)

    if error:
        print(error.as_string())
    elif result:
        try:
            if len(result.elements) == 1:
                print(repr(result.elements[0]))
            else:
                print(repr(result))
        except Exception as e:
            None
