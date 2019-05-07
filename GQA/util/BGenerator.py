import threading
import queue

class BGenerator(threading.Thread):
    def __init__(self, generator, max_prefetch=1):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self

class background_generator:
    def __init__(self, max_prefetch=1):
        self.max_prefetch = max_prefetch

    def __call__(self, generator):
        def bg_generator(*args, **kwargs):
            return BGenerator(generator(*args, **kwargs), max_prefetch=self.max_prefetch)
        return bg_generator

