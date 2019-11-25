from dataProcessors.StrategyRandomClassRandomSample import StrategyRandomClassRandomSample
from dataProcessors.StrategyPseudoRandomClassRandomSample import StrategyPseudoRandomClassRandomSample
from dataProcessors.GenerationStrategyType import GenerationStrategyType


class BatchStrategyFactory:

    def create(self, strategyType:GenerationStrategyType):
        method = getattr(self, 'getStrategy' + strategyType.name, f'{strategyType.name} does not have any implementation')
        return method()

    
    def getStrategyRandomClassRandomSample(self):
        return StrategyRandomClassRandomSample()
        
        
    def getStrategyPseudoRandomClassRandomSample(self):
        return StrategyPseudoRandomClassRandomSample()
        