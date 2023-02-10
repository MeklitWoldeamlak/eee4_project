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
#### State transitions
In this case the model is a probabilistic model which means each state transitions are represented in probabilistic form; the likelihood the DUT goes from a given state it to any other state or stays in the same state. No real data is collected of how this probability state transitions will look like. Hence, begin defining these probability state transitions by taking into account  of what is known at the moment of definition and making reasonable assumptions.   
Having a function for every voltage below is a general idea of how the probability of transition will look like.
#### MDP: 
By taking the one state stransition model i.e (state I), Markov decision process is applyed to determine the next state for a given applied voltage. it returns next state with the probablity.

 
<img src="https://github.com/MeklitWoldeamlak/eee4_project/blob/master/Data/im1.jpg"  width=40% height=40%>
a)	State transition diagram from state I

<img src="https://github.com/MeklitWoldeamlak/eee4_project/blob/master/Data/im2.jpg"  width=40% height=40%>
b) probability of transition from state I

<img src="https://github.com/MeklitWoldeamlak/eee4_project/blob/master/Data/im3.jpg"  width=40% height=40%>
Given that we have a number of actions in a list(a1,a2, a3,…)  the transition diagram can also look like this

#### Variables Involved  
<img src="https://github.com/MeklitWoldeamlak/eee4_project/blob/master/Data/im.jpg"  width=40% height=40%>

#### Software file
•	The main () function and algorithm is included
•	Used a simple interface just like with the real hardware (Needs to be compatible with the real interface
•	Make sure  we don’t use info that is not available to us with the hardware
•	It doesn’t know what version of the model is being used in the hardware

#### Hardware file
•	Represents the actual FPGA(Arc2 system in our case)
•	Software version of the hardware which has the same constraints and behave in similar way
•	Only talk to it via the functions like apply_voltage ( )
•	It’s characteristics functions can’t be accessed 

#### Process
- After writing algorithm against the hardware file, run algorithm over many iteration/ devices and get a result
- Change the model of hardware for another with a slightly different characteristics and run algorithm again and see if the algorithms copes (learns the new model as well as the old one)
- This will show us how robust the algorithm is, checks out assumptions about the interface 
- It will help generate a baseline set of results
- This can be done several times then when we apply it to the real hardware we will do so with the confidence 

#### Assumptions 

1. The electroforming probabilities characteristics in real hardware does not change over time on a single device/ This can be also from device to device in a wafer  
2. The electroforming probabilities characteristics will change with the physical series resistor
3. To go to higher resistance state requires a positive voltage
4. To go to lower resistance state requires a negative voltage
5. Applying no voltage(0V) to the device results in no change in the device
6. The chances of a device failing increases with voltage
7. The probability transitions are modelled to follow a gaussian distribution except the transition to a failed state or when it is remaining in the same state

    - transition to a Fail state is modelled to follow symmetric normal distribution or symmetric sigmoid function
    - The probablity of remaining in the same state is modelled to so that it has higher value around zero and decreseas as we gp further away. The probablity will be 1 - sum(all other probablity transitioms)
8. The Vmax is set to 5V and Vmin is -5V. Applying a voltage ranging between these two value is resonable enough for electroforming 

#### Update on Assumptions (Feb 09)

1. Revisit later in the future as we discover more data but for now stick with the assumption
2.  Assume there is no need for inclusion of series resistor. The electroforming probabilities characteristics will not depend on the series resistor. Add the capability later on   
9. Assuming model is fully observable (possible to read state after each action), we can apply series of voltage(action) after first action. ie there are no hidden state
10. We assume the model follows stationary state: the transition probabilities will remain the same and it does not change based on history. For example, looking at transitions
 I--> II and I -> I --> II, the transition probabilities (bold arrow) will be the same.

##### Question on the initial state, is there a specific state that all devices begin with before electroforming? Or they are all random?


 