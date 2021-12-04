def gen_sentences(path):
    with open(path, encoding="utf8") as file:
        for line in file:
            if line != "\n":
                line = line.strip("\n")
                yield line.split(" ")


def case_sarcastically(text):
    text = list(text)
    counter = 0
    #replace with boolean?
    for i in range(len(text)):
        if(text[i].upper()) != (text[i].lower()):
            if counter % 2 == 0:
                text[i] = text[i].lower()
                counter += 1
            else:
                text[i] = text[i].upper()
                counter += 1
    return ''.join(text)


def prefix(s, n):
    if n > len(s):
        raise ValueError("number is too big")
    elif n < 0:
        raise ValueError("number is negative")
    elif n == 0:
        raise ValueError("number cannot be 0")
    return s[:n]


def suffix(s, n):
    if n > len(s):
        raise ValueError("number is too big")
    elif n < 0:
        raise ValueError("number is negative")
    elif n == 0:
        raise ValueError("number cannot be 0")
    return s[-n:]


def sorted_chars(s):
    s = sorted(set(s))
    return s
