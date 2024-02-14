from collections import defaultdict
import time
import torch


class Timer:
    times = defaultdict(list)

    @staticmethod
    def echo():
        print("------------TIMER------------")

        for func in Timer.times.keys():
            if func != 'timing_val.<locals>.wrapper':
                print(func, " ", round(100*sum(Timer.times[func])/sum(
                    Timer.times["VisualizationDemo.run_on_image"]), 1), "%")

        print("total time:", sum(
            Timer.times["VisualizationDemo.run_on_image"]))


def timing_val(func):
    def wrapper(*arg, **kw):
        '''source: http://www.daniweb.com/code/snippet368.html'''
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        res = func(*arg, **kw)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        runtime = t2 - t1
        name = func.__qualname__
        Timer.times[name].append(runtime)
        return res
    return wrapper
