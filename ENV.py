import numpy as np

class ENVIRONMENT:
    def __init__(self,SensorNum,ActionNum,ActionChoice ):
        self.actionsize = ActionNum
        self.sensornum = SensorNum
        self.actionchoice = ActionChoice

    def creatSensor(self,Power):
        stateInput = np.zeros(self.sensornum)
        for i in range(self.sensornum):
            stateInput[i] = Power
        stateInput = stateInput.reshape(1,-1)
        return stateInput

    def getReward(self,stateInput,actionInput):
        Power = stateInput[0][0]
        Action = list(actionInput).index(1)
        Action = self.actionchoice[Action]
        temp = abs(Power + Action)//1

        reward = 0
        if  not temp:
            reward = 10
        if temp:
            reward = -1
        print("reward is:%d" %reward)
        return reward,Action
