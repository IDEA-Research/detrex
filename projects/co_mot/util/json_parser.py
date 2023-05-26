import json
import sys

def parse(str, key):
    str_dict = json.loads(str)
    val = str_dict[key]
    if type(val)==list:
        return ",".join(val)
    else:
        return val

if __name__ == '__main__':
    parse(sys.argv[1], sys.argv[2])