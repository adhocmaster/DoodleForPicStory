from abc import ABC, abstractmethod

class Strategy(ABC):


    @abstractmethod
    def getBatch(self, generator, batchIndex):
        pass

