

def first_name(inp: str, bar: int = 1) -> str:
    return inp.split(' ')[0]


foo: int = 1

first_name(foo)

first_name('a b')
