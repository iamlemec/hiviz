# main

from .testing import run_test
from . import testing

# run test
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python -m hiviz <test_name>')
        sys.exit(1)

    # run desired test
    _, gen_name = sys.argv
    gen = getattr(testing, gen_name)
    run_test(gen)
