#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from random import uniform


# In[39]:


dados = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data", header= None, prefix = "V")


# In[44]:


dados.head()


# In[ ]:





# In[49]:


for x in range(208):
    if(dados.iat[x,60]=="M"):
        pcolor='red'
    else:
        pcolor = 'blue'
    dataRow = dados.iloc[x,0:60]
    dataRow.plot(color=pcolor)
plot.xlabel("Attribute Index")
plot.ylabel(("Attribute Values"))
plot.show()


# In[53]:


dataRow2=dados.iloc[1,0:60]
dataRow3=dados.iloc[2,0:60]
plot.scatter(dataRow2, dataRow3)
plot.xlabel("2nd Attribute")
plot.ylabel(("3rd Attribute"))
plot.show()


# In[55]:


dataRow21 = dados.iloc[20,0:60]
plot.scatter(dataRow2, dataRow21)
plot.xlabel("2nd Attribute")
plot.ylabel(("21st Attribute"))
plot.show()


# In[57]:


target = []
for x in range(208):
    if(dados.iat[x,60]=="M"):
        target.append(1.0)
    else:
        target.append(0.0)
dataRow4 = dados.iloc[0:208,35]
plot.scatter(dataRow4, target)
plot.xlabel("Attribute Value")
plot.ylabel("Target Value")
plot.show()


# In[59]:


target2 = []
for i in range(208):
#assign 0 or 1 target value based on "M" or "R" labels
# and add some dither
    if dados.iat[i,60] == "M":
        target2.append(1.0 + uniform(-0.1, 0.1))
    else:
        target2.append(0.0 + uniform(-0.1, 0.1))
#plot 35th attribute with semi-opaque points
dataRow5 = dados.iloc[0:208,35]
plot.scatter(dataRow5, target2, alpha=0.5, s=120)
plot.xlabel("Attribute Value")
plot.ylabel("Target Value")
plot.show()


# In[63]:


corMat=pd.DataFrame(dados.corr())
plot.pcolor(corMat)
plot.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




