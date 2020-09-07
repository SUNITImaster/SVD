# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 00:36:35 2020
This is my attemp to solve the Gamblers problem example 4.3 Sutton, using value iteration

@author: Suniti 

"""

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot

#initializing input variables
p_h=0.4 #prob of head
p_t=1-p_h #prob of tail
no_of_iter = 100 #max no. of value iterations to be made
tol=0.0000001 #tolerance value for accepting convergence

#declaring a policy construct
pol1=np.ones(100)
pol=np.zeros((100,100))
q=np.zeros((100,100))
state=np.zeros(101)
state[100]=100

#initializing the value array
#val=np.array([(i+0.0000)/100 for i in range(101)])
val=np.array([0.00 for i in range(101)])
val[0]=0.0
val[100]=1.0


#initializing a uniformly random policy that explores each action with equal probability
for i in range(1,100,1):
    state[i]=i
    for j in range(1,100,1):
        if j<=i:
            pol[i,j]=1/i
        else:
            pol[i,j]=0
        

#print(pol)

print("initialized val is :")
print(val)

# Value Iteration block 

for iteration in range(no_of_iter):
    vold=val.copy()
    for s in range(1,100,1):#This is an inplace value update algorithm
        print("start of state "+np.str(s))
        for a in range(1,s+1,1):
            q[s,a]=p_t*val[s-a]+0.4*val[min(s+a,100)]
            #print(v[s,a])
        val[s]= max(q[s,:])
        print("value of state "+str(s)+" is "+str(val[s]))
        pol1[s]=np.argmax(q[s,:])
        #print("value for state "+str(s)+" is :"+str(val[s]))
        
    diff=sum(abs(vold-val))
    print("difference in value functions after iteration "+str(iteration)+"is below :")
    
    print(max([abs(val[s]-vold[s]) for s in range(1,100,1)]))
    if(diff<tol):
        print("tolerance met so exiting the value iteration loop !")
        break
     
valf=val[1:100]
fig=make_subplots(rows=1,cols=2)
df=pd.DataFrame(pd.concat([pd.Series(state),pd.Series(pol1)],axis=1))
df.columns=["state","optimalaction"]
fig.add_trace(go.Scatter(x=df["state"],y=df["optimalaction"],name="optimal policy"),row=1,col=1)



df1=pd.DataFrame(pd.concat([pd.Series(state),pd.Series(valf)],axis=1))
df1.columns=["state","optimalvalue"]
fig.add_trace(go.Scatter(x=df1["state"],y=df1["optimalvalue"],name="optimal value function"),row=1,col=2)
fig.update_layout(height=600,width=1200,title="final converged Optimal policy and value funcion")
plot(fig)  

print("Printed the optimal value function")
