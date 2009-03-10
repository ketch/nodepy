def permutations(str):
    if len(str) <=1:
        yield str
    else:
        for perm in permutations(str[1:]):
            for i in range(len(perm)+1):
                yield perm[:i] + str[0:1] + perm[i:]
