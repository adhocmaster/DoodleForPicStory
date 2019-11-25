from abc import ABC, abstractmethod

class BatchStrategy(ABC):


    @abstractmethod
    def getBatch(self, generator, batchIndex):
        pass

