import pandas as pd
import  numpy as np

Loss = [1,2,3,4,5,6,7,8,9]
Loss = np.array(Loss)

data = pd.DataFrame(Loss,columns=['loss'])

data.to_csv("D:\YuanZihong\loss.csv")