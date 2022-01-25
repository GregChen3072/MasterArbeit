import multiprocessing


def f(x):
    return x*x


def calc_stuff():
    pass


if __name__ == '__main__':
    with multiprocessing.Pool(5) as p:
        print(p.map(f, [1, 2, 3]))


offset = 1
# setup output lists
output1 = list()
output2 = list()
output3 = list()

for j in range(0, 10):
    # calc individual parameter value
    parameter = j * offset
    # call the calculation
    out1, out2, out3 = calc_stuff(parameter=parameter)

    # put results into correct output list
    output1.append(out1)
    output2.append(out2)
    output3.append(out3)

pool = multiprocessing.Pool(4)
out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))
