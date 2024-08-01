import time


def cost_time(func):

    def fun(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        print(f'func {func.__name__} cost time:{time.perf_counter() - t:.8f} s')
        return result

    return fun


@cost_time
def test():
    print('func start')
    time.sleep(2)
    print('func end')


if __name__ == '__main__':
    test()
