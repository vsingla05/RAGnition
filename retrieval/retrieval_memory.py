# retrieval/retrieval_memory.py

class RetrievalMemory:

    def __init__(self):
        self.history = []

    def add(self, item):
        self.history.append(item)

    def last(self):
        return self.history[-1] if self.history else None