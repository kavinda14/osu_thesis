from multiprocessing import Pool

if __name__ == "__main__":
    
    global_thing = 0
    def do_something(a, b, global_thing):
        global_thing += 1
        return a+b 

    input_list = list()
    for i in range(5):
        sub_list = [i, i+1, global_thing]
        input_list.append(sub_list)

    pool = Pool()
    results = pool.starmap(do_something, input_list)
    pool.close()
    pool.join()

    print(results)
    print(global_thing)