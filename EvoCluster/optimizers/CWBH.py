# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:19:49 2016

@author: hossam
"""
import random
import numpy
import math
from .._solution import solution
import time

def WBH(objf,lb,ub,dim,SearchAgents_no,Max_iter,k,points, metric):


    #dim=30
    #SearchAgents_no=50
    #lb=-100
    #ub=100
    #Max_iter=500
      
    # initialize position vector and score for the leader
#-----------BAT INITIALIZATION-------------------------------#
    n=SearchAgents_no;      # Population size
    #lb=-50
    #ub=50
    
    N_gen=Max_iter  # Number of generations
    
    A=0.9;      # Loudness  (constant or decreasing)
    r=0.9;      # Pulse rate (constant or decreasing)
    
    Qmin=0         # Frequency minimum
    Qmax=10         # Frequency maximum
    
    
    d=dim           # Number of dimensions 
    
    # Initializing arrays
    Q=numpy.zeros(n)  # Frequency
    v=numpy.zeros((n,d))  # Velocities
    Convergence_curve=[]
    
    # Initialize the population/solutions
    Sol=numpy.random.rand(n,d)*(ub-lb)+lb
    labelsPred=numpy.zeros((n,len(points)))
    Fitness=numpy.zeros(n)

    S=numpy.zeros((n,d))
    S=numpy.copy(Sol)
#--------WHALE INITIALIZATION---------------------------#
    Leader_pos=numpy.zeros(dim)
    Leader_score=float("inf")  #change this to -inf for maximization problems
    
    
    #Initialize the positions of search agents
    Positions = numpy.zeros((SearchAgents_no, dim))
    
    Positions=numpy.random.uniform(0,1,(SearchAgents_no,dim)) *(ub-lb)+lb
    #labelsPred=numpy.zeros((SearchAgents_no,len(points)))
    
    #Initialize convergence
    convergence_curve=numpy.zeros(Max_iter)
    
    
    ############################
    s=solution()

    print("WBH is optimizing  \""+objf.__name__+"\"")    

    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################


    #-------BAT PART--------------------------#
    
    #Evaluate initial random solutions
    for i in range(0,n):
        startpts = numpy.reshape(Sol[i,:], (k,(int)(dim/k)))
        if objf.__name__ == 'SSE' or objf.__name__ == 'SC' or objf.__name__ == 'DI':
            fitnessValue, labelsPredValues=objf(startpts, points, k, metric) 
        else:
            fitnessValue, labelsPredValues=objf(startpts, points, k) 
        Fitness[i] = fitnessValue
        labelsPred[i,:] = labelsPredValues

    
    # Find the initial best solution
    fmin = min(Fitness)
    I=numpy.argmin(Fitness)
    best=Sol[I,:]
    bestLabelsPred=labelsPred[I,:]
       
    # Main loop
    for t in range (0,N_gen): 
        
        # Loop over all bats(solutions)
        for i in range (0,n):
          Q[i]=Qmin+(Qmin-Qmax)*random.random()
          v[i,:]=v[i,:]+(Sol[i,:]-best)*Q[i]
          S[i,:]=Sol[i,:]+v[i,:]
          
          # Check boundaries
          Sol=numpy.clip(Sol,lb,ub)

    
          # Pulse rate
          if random.random()>r:
              S[i,:]=best+0.001*numpy.random.randn(d)
          
    
          # Evaluate new solutions
          startpts = numpy.reshape(S[i,:], (k,(int)(dim/k)))

          if objf.__name__ == 'SSE' or objf.__name__ == 'SC' or objf.__name__ == 'DI':
              fitnessValue, labelsPredValues=objf(startpts, points, k, metric) 
          else:
              fitnessValue, labelsPredValues=objf(startpts, points, k) 
              
          Fnew = fitnessValue
          LabelsPrednew = labelsPredValues
          
          # Update if the solution improves
          #if ((Fnew != numpy.inf) and (Fnew<=Fitness[i]) and (random.random()<A) ):
          if ((Fnew != numpy.inf) and (Fnew<=Fitness[i]) ):
                Sol[i,:]=numpy.copy(S[i,:])
                Fitness[i]=Fnew
                labelsPred[i,:]=LabelsPrednew
           
    
          # Update the current best solution
          if Fnew != numpy.inf and Fnew<=fmin:
                best=numpy.copy(S[i,:])
                fmin=Fnew
                bestLabelsPred=LabelsPrednew

          #----------WHALE PART-------------------------#
          Positions=best
            
          Positions[i,:]=numpy.clip(Positions[i,:], lb, ub)
   
          fitness = fitnessValue
          labelsPred[i,:] = labelsPredValues
          
          # Update the leader
          if fitness<Leader_score: # Change this to > for maximization problem
              Leader_score=fitness  
              #Leader_score=fitness; # Update alpha
              Leader_pos=Positions[i,:].copy() # copy current whale position into the leader position
              #Leader_pos=best
              Leader_labels=labelsPred[i,:].copy() # copy current whale position into the leader position
                

        '''
        for i in range(0,SearchAgents_no):
            
            # Return back the search agents that go beyond the boundaries of the search space
            
            #Positions[i,:]=checkBounds(Positions[i,:],lb,ub)            
            Positions[i,:]=numpy.clip(Positions[i,:], lb, ub)

            
            # Calculate objective function for each search agent
            startpts = numpy.reshape(Positions[i,:], (k,(int)(dim/k)))

            if objf.__name__ == 'SSE' or objf.__name__ == 'SC' or objf.__name__ == 'DI':
                fitnessValue, labelsPredValues=objf(startpts, points, k, metric) 
            else:
                fitnessValue, labelsPredValues=objf(startpts, points, k) 
                
            fitness = fitnessValue
            labelsPred[i,:] = labelsPredValues
            
            # Update the leader
            if fitness<Leader_score: # Change this to > for maximization problem
                Leader_score=fitness; # Update alpha
                Leader_pos=Positions[i,:].copy() # copy current whale position into the leader position
                Leader_labels=labelsPred[i,:].copy() # copy current whale position into the leader position
            '''
            
        
        
        a=2-t*((2)/Max_iter); # a decreases linearly fron 2 to 0 in Eq. (2.3)
        
        # a2 linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
        a2=-1+t*((-1)/Max_iter);
        
        # Update the Position of search agents 
        for i in range(0,SearchAgents_no):
            r1=random.random() # r1 is a random number in [0,1]
            r2=random.random() # r2 is a random number in [0,1]
            
            A=2*a*r1-a  # Eq. (2.3) in the paper
            C=2*r2      # Eq. (2.4) in the paper
            
            
            b=1;               #  parameters in Eq. (2.5)
            l=(a2-1)*random.random()+1   #  parameters in Eq. (2.5)
            
            p = random.random()        # p in Eq. (2.6)
            
            for j in range(0,dim):
                
                if p<0.5:
                    if abs(A)>=1:
                        rand_leader_index = math.floor(SearchAgents_no*random.random());
                        X_rand = Positions[rand_leader_index, :]
                        D_X_rand=abs(C*X_rand[j]-Positions[i,j]) 
                        Positions[i,j]=X_rand[j]-A*D_X_rand      
                        
                    elif abs(A)<1:
                        D_Leader=abs(C*Leader_pos[j]-Positions[i,j]) 
                        Positions[i,j]=Leader_pos[j]-A*D_Leader      
                    
                    
                elif p>=0.5:
                  
                    distance2Leader=abs(Leader_pos[j]-Positions[i,j])
                    # Eq. (2.5)
                    Positions[i,j]=distance2Leader*math.exp(b*l)*math.cos(l*2*math.pi)+Leader_pos[j]
                    
      
        
        convergence_curve[t]=Leader_score
        if (t%1==0):
               print(['At iteration '+ str(t)+ ' the best fitness is '+ str(Leader_score)]);
        t=t+1
    
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence_curve
    s.optimizer="WBH"   
    s.objfname=objf.__name__
    s.best = Leader_score
    s.bestIndividual = Leader_pos
    s.labelsPred = numpy.array(Leader_labels, dtype=numpy.int64)
    
    return s




# -*- coding: utf-8 -*-
"""
Created on Thu May 26 02:00:55 2016

@author: hossam
"""

    
'''
def BAT(objf,lb,ub,dim,N,Max_iteration, k, points, metric):
    
    n=N;      # Population size
    #lb=-50
    #ub=50
    
    N_gen=Max_iteration  # Number of generations
    
    A=0.5;      # Loudness  (constant or decreasing)
    r=0.5;      # Pulse rate (constant or decreasing)
    
    Qmin=0         # Frequency minimum
    Qmax=2         # Frequency maximum
    
    
    d=dim           # Number of dimensions 
    
    # Initializing arrays
    Q=numpy.zeros(n)  # Frequency
    v=numpy.zeros((n,d))  # Velocities
    Convergence_curve=[];
    
    # Initialize the population/solutions
    Sol=numpy.random.rand(n,d)*(ub-lb)+lb
    labelsPred=numpy.zeros((n,len(points)))
    Fitness=numpy.zeros(n)

    S=numpy.zeros((n,d))
    S=numpy.copy(Sol)
    
    
    # initialize solution for the final results   
    s=solution()
    print("BAT is optimizing  \""+objf.__name__+"\"")    
    
    # Initialize timer for the experiment
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    #Evaluate initial random solutions
    for i in range(0,n):
        startpts = numpy.reshape(Sol[i,:], (k,(int)(dim/k)))
        if objf.__name__ == 'SSE' or objf.__name__ == 'SC' or objf.__name__ == 'DI':
            fitnessValue, labelsPredValues=objf(startpts, points, k, metric) 
        else:
            fitnessValue, labelsPredValues=objf(startpts, points, k) 
        Fitness[i] = fitnessValue
        labelsPred[i,:] = labelsPredValues
    
    
    # Find the initial best solution
    fmin = min(Fitness)
    I=numpy.argmin(Fitness)
    best=Sol[I,:]
    bestLabelsPred=labelsPred[I,:]
       
    # Main loop
    for t in range (0,N_gen): 
        
        # Loop over all bats(solutions)
        for i in range (0,n):
          Q[i]=Qmin+(Qmin-Qmax)*random.random()
          v[i,:]=v[i,:]+(Sol[i,:]-best)*Q[i]
          S[i,:]=Sol[i,:]+v[i,:]
          
          # Check boundaries
          Sol=numpy.clip(Sol,lb,ub)

    
          # Pulse rate
          if random.random()>r:
              S[i,:]=best+0.001*numpy.random.randn(d)
          
    
          # Evaluate new solutions
          startpts = numpy.reshape(S[i,:], (k,(int)(dim/k)))

          if objf.__name__ == 'SSE' or objf.__name__ == 'SC' or objf.__name__ == 'DI':
              fitnessValue, labelsPredValues=objf(startpts, points, k, metric) 
          else:
              fitnessValue, labelsPredValues=objf(startpts, points, k) 
              
          Fnew = fitnessValue
          LabelsPrednew = labelsPredValues
          
          # Update if the solution improves
          if ((Fnew != numpy.inf) and (Fnew<=Fitness[i]) and (random.random()<A) ):
                Sol[i,:]=numpy.copy(S[i,:])
                Fitness[i]=Fnew
                labelsPred[i,:]=LabelsPrednew
           
    
          # Update the current best solution
          if Fnew != numpy.inf and Fnew<=fmin:
                best=numpy.copy(S[i,:])
                fmin=Fnew
                bestLabelsPred=LabelsPrednew
                
        #update convergence curve
        Convergence_curve.append(fmin)        

        if (t%1==0):
            print(['At iteration '+ str(t)+ ' the best fitness is '+ str(fmin)])
    
    
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=Convergence_curve
    s.optimizer="BAT"   
    s.objfname=objf.__name__
    s.labelsPred = numpy.array(bestLabelsPred, dtype=numpy.int64)
    s.bestIndividual = best
    
    
    
    return s

'''
