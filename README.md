### Overview
This aim of the project is to optimize electroforming testing for nanodevices (memristors) using Reinforcement Learning.  

### Main tasks:	 

- First build a toy model so that we gather as much information from the tests on the model.  

- List all the parameters that are going to be used. These are the input and output parameters (possibly include system hyperparameters as well).   

- Perform compliance testing on a number of selected solutions (main target: RL set-ups; other possibilities: standard ANNs or rules-based approaches).  

- Assess whether toy model has enough discriminatory capability to distinguish between the different outcomes taking into consideration the input parameters 

- Once the toy model works in simulation, transfer & test on real devices and observe.  

- Launch a downloadable module available for community to use. 

### Progress

#### Allstates: 
Consists of all state transition models. Makes use of gaussian probablity distribution and sigma probablity distribution 
#### MDP: 
By taking the one state stransition model i.e (state I), Markov decision process is applyed to determine the next state for a given applied voltage. it returns next state with the probablity.

