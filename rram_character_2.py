import numpy as np
import json
import time
import argparse
import logging
import random
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import arc2_simulator2 as arc2
from numpy import asarray
from numpy import savetxt
import numpy as gfg
from collections import Counter
import pandas as pd
        
SAMPLE_VOLTAGE_IN_MV = 5
DEFAULT_NUMBER_DEVICES = 1000
DEFAULT_NUMBER_WAFERS = 1
DEFAULT_MAX_ATTEMPT =20
MODEL_INDEX=5
STEP_VOLTAGES=50
DEFAULT_ALGORITHM = "epsilon_c"
GAMMA = 0.7 #discount factor
ALPHA = 0.9 #learning factor

class Arc2Tester(ABC):
    """This is the deliverable"""
    def run(self,hardware: arc2.Arc2HardwareSimulator) -> list:
        """Run test on hardware"""  
        _report = []
        _time_record=[]
       # _transition_record= np.array(np.zeros([3,4,20]))
        while True:
            n=0 #time step
            # get current state of device from the hardware
            _current_state = hardware.get_current_device_state()
            # select the state we want this to be, random but could be read from file
            # and don't choose the current state
            _possible_target_states = set(range(arc2.NUM_NON_FAIL_STATES))
            _possible_target_states.remove(_current_state)
            _target_state = random.choice(list(_possible_target_states))
            # determine the voltage to apply
            for i in range(args.max_attempts):
                n+=1
                _action=self.get_action(_current_state,_target_state)
                _voltage =_action['voltage']
                _pulse_duration = _action['pulse_duration']
                # apply to hardware a number of times representing pulse length
                for _ in range(_pulse_duration):
                    hardware.apply_voltage(_voltage)
                _new_state = hardware.get_current_device_state()
                
                self.update(_current_state,_target_state,_action,_new_state)
                if _new_state ==_target_state or arc2.STATES[_new_state]=="FAIL":
                    break
                 
                _current_state=_new_state
            _time_record=n    
            _action_index =  int((_voltage+5)/0.5)
           # _transition_record[_current_state][_new_state][_action_index-1]+=1
            _report.append(
                {
                    'current_state': arc2.STATES[_current_state],
                    'target_state': arc2.STATES[_target_state],
                    'actual_state': arc2.STATES[_new_state],
                    'to state I': _new_state == 0,
                    'to state II': _new_state == 1,
                    'to state III': _new_state == 2,
                    'voltage_applied': _voltage,
                    'voltage_pulse': _pulse_duration,
                    'success': _target_state == _new_state,
                    #Í'table': _transition_record,
                    'time_step':_time_record
                }
            )
           
            # check to see if we have finished the wafer
            if not hardware.move_to_next_device():
                break
                
        self.q_table()
        
        # fig, ax = plt.subplots()
        # x=["{:.1f}".format(d['voltage_applied']) for d in _report]
        # count = Counter(sorted(x))
        # df = pd.DataFrame.from_dict(count, orient='index')
        # df.plot(ax=ax,kind='bar')
        # ax.set_ylabel('Frequency')
        # ax.set_xlabel('Voltage')
        # ax.legend(labels=['Epsilon greedy'])
        # plt.show()
        # print(sorted(x))
        return _report
    
    @abstractmethod
    def get_action(self, current_state, target_state) -> dict:
        raise NotImplementedError("Must override get_action")
    def update(self, old_state:int, target_state:int, action:dict, new_state:int ):
        pass
    def q_table(self):
        pass

class RandomVoltageArc2Tester(Arc2Tester):
    
    """First attelet's just pick a voltage randomly"""
    def __init__(self):
        super().__init__()
    
    def get_action(self, current_state, target_state) -> dict:
        _voltage= random.uniform(arc2.MIN_VOLTAGE,arc2.MAX_VOLTAGE)
        return {'voltage':_voltage, 
                'pulse_duration': 1}
    
    def update(self, old_state:int, target_state:int, action:dict, new_state:int ):
        pass
    def q_table(self):
        pass
class RandomVoltageWithRangeKnowledgeArc2Tester(Arc2Tester):
    """Addition to First attempt - let's just pick a voltage randomly but with using already
    known knowlege of hardware device"""
    def __init__(self):
        super().__init__()

    def get_action(self, current_state, target_state) -> dict:
        _voltage = random.randrange(arc2.MIN_VOLTAGE*10,arc2.MAX_VOLTAGE*10,5)/10 
        if current_state == 0:
            _voltage = random.randrange(0,arc2.MAX_VOLTAGE*10,5)/10 
        elif current_state == 2:
            _voltage= -random.randrange(0,arc2.MAX_VOLTAGE*10,5)/10 
        return {'voltage':_voltage, 
                'pulse_duration': 1}
    def update(self, old_state:int, target_state:int, action:dict, new_state:int ):
        pass
    def q_table(self):
        pass

class ExperiencedUserTester(Arc2Tester):
    """Second attempt - a knowledge user has determined a good set of values
    
    A user has determined, having electroformed many devices over the years,
    a good set of voltages to use depending on what we are trying to do
    """
    def __init__(self,model:int):
        super().__init__()
        self.model=model
        self.lookup_matrix=np.zeros((arc2.NUM_STATES,arc2.NUM_STATES))
        with open("mdp_param.json") as f:
            mdp_param = json.load(f)
        model_param=mdp_param[self.model]
        self.lookup_matrix[0][1]=model_param[0][0]['mean']
        self.lookup_matrix[0][2]=model_param[0][1]['mean']
        self.lookup_matrix[1][0]=model_param[1][0]['mean']
        self.lookup_matrix[1][2]=model_param[1][1]['mean']
        self.lookup_matrix[2][0]=model_param[2][0]['mean']
        self.lookup_matrix[2][1]=model_param[2][1]['mean']
            
    def get_action(self, current_state, target_state) -> dict:
        return {'voltage': self.lookup_matrix[current_state][target_state],
                'pulse_duration':1}


class EpsilonGreedyTester(Arc2Tester):
    """ Third attempt- mix exploration with simple exploitation 
    Use a greedy epsilon method. start with lots of random exploration to generate an expected 
    reward table and change slowly to exploitation using a simple reward based approach 
    Reward:
    +1 if device didn't fail during electroform
    +10 if device succesfully electroformed 
    -20 if device failed after electroforming 
    """
    def __init__(self, max_attempts:int,voltage_step:int, gamma:float):
        super().__init__()
        self._voltage_inc=(arc2.MAX_VOLTAGE - arc2.MIN_VOLTAGE)/float(voltage_step) #float voltage increment (0.5)
        self._voltage_step= voltage_step+1 # total number of actions(20)
        self._voltages= [arc2.MIN_VOLTAGE+i*self._voltage_inc for i in range(self._voltage_step)] #actual value of volatges
        self._expected_reward_table = np.zeros((arc2.NUM_NON_FAIL_STATES,
                                                arc2.NUM_NON_FAIL_STATES,
                                                self._voltage_step))
        self._epsilon = 1
        self._gamma= gamma
        self._exploitation = 0
        self._exploration = 0
        self._action_value_est= 0
        
    def get_action(self, current_state, target_state) -> dict:
        
        _actions= self._voltages
        #for i in range(args.max_attempts):
             
        p = np.random.random()
        if p < self._epsilon:
            self._exploration += 1
            if current_state == 0:
                _voltage_index = random.randrange((self._voltage_step-1)/2,self._voltage_step)
            elif current_state == 1 and target_state ==0:
                _voltage_index= random.randrange(0,(self._voltage_step-1)/2)
            elif current_state == 1 and target_state ==2:
                _voltage_index= random.randrange((self._voltage_step-1)/2,self._voltage_step)
            elif current_state == 2:
                _voltage_index= random.randrange(0,(self._voltage_step-1)/2)
            _voltage=self._voltages[_voltage_index] 
            #_voltage_index = np.random.choice(self._voltage_step)
            #_voltage=self._voltages[_voltage_index]
        else:
            self._exploitation+= 1
            _voltage_index = np.argmax([a for a in self._expected_reward_table[current_state][target_state]])
            _voltage=self._voltages[_voltage_index]       
        
        return {'voltage': _voltage,
                'pulse_duration':1,
                'voltage_index':_voltage_index
        }
           
        
    def update(self, old_state:int, target_state:int, action:dict, new_state:int ):
        """update the expected reward table
        add 1 to self_exploration every time we explore(take single action)
        """
        
       # _index= int(action['voltage']/self._voltage_inc)
        _index= action['voltage_index']
        if new_state==target_state :
            self._expected_reward_table[old_state][target_state][_index]+=10
        elif new_state==3 :#fail
            self._expected_reward_table[old_state][target_state][_index]+=-20
        else:
            self._expected_reward_table[old_state][target_state][_index]+=1
        
        #favour exploitation a little bit more   
        self._epsilon *= 0.9999
    def q_table(self):
        Q=self._expected_reward_table
        _mean_voltage=np.zeros((arc2.NUM_NON_FAIL_STATES,arc2.NUM_NON_FAIL_STATES))
        for i in range(arc2.NUM_NON_FAIL_STATES):
            for j in range(arc2.NUM_NON_FAIL_STATES):
                _mean_v_index = np.argmax([a for a in Q[i][j]])
                _mean_voltage[i,j]=self._voltages[_mean_v_index]
                if i==j:
                    _mean_voltage[i,j]=0        
        print('Number of exploration=', self._exploration) 
        print('Number of exploitation=', self._exploitation) 
        print('Final epsilon=', self._epsilon)
        print(Q)
        print(_mean_voltage)          
class EpsilonGreedyCautious(Arc2Tester):
    """Update of EpsilonGreedy with a bit more caution made 
    by starting from lower voltages first and then progressing to higher 
    voltages as we explore
    """
    def __init__(self, max_attempts:int,voltage_step:int, gamma:float):
        super().__init__()
        self.range_subtractor=4
        self._voltage_inc=(arc2.MAX_VOLTAGE - arc2.MIN_VOLTAGE)/float(voltage_step) #float voltage increment (0.5)
        self._voltage_step= voltage_step+1 # total number of actions(20)
        self._voltages= [arc2.MIN_VOLTAGE+i*self._voltage_inc for i in range(self._voltage_step)] #actual value of volatges
        self._expected_reward_table = np.zeros((arc2.NUM_NON_FAIL_STATES,
                                                arc2.NUM_NON_FAIL_STATES,
                                                self._voltage_step))
       
        self._epsilon = 1
        self._gamma= gamma
        self._exploitation = 0
        self._exploration = 0
        self._action_value_est= 0
        
    def get_action(self, current_state, target_state) -> dict:
        
        _actions= self._voltages
        #for i in range(args.max_attempts):
             
        p = np.random.random()
        if p < self._epsilon:
            self._exploration += 1
            if current_state == 0:
                _voltage_index = random.randrange((self._voltage_step-1)/2,self._voltage_step-self.range_subtractor)
            elif current_state == 1 and target_state ==0:
                _voltage_index= random.randrange(0+self.range_subtractor,(self._voltage_step-1)/2)
            elif current_state == 1 and target_state ==2:
                _voltage_index= random.randrange((self._voltage_step-1)/2,self._voltage_step-self.range_subtractor)
            elif current_state == 2:
                _voltage_index= random.randrange(0+self.range_subtractor,(self._voltage_step-1)/2)
            _voltage=self._voltages[_voltage_index] 
        else:
            self._exploitation+= 1
            _voltage_index = np.argmax([a for a in self._expected_reward_table[current_state][target_state]])
            _voltage=self._voltages[_voltage_index]       
        
        return {'voltage': _voltage,
                'pulse_duration':1,
                'voltage_index':_voltage_index
        }     
        
    def update(self, old_state:int, target_state:int, action:dict, new_state:int ):
        """update the expected reward table
        add 1 to self_exploration every time we explore(take single action)
        """
        
       # _index= int(action['voltage']/self._voltage_inc)
        _index= action['voltage_index']
        if new_state==target_state :
            self._expected_reward_table[old_state][target_state][_index]+=10
        elif new_state==3 :#fail
            self._expected_reward_table[old_state][target_state][_index]+=-20
        else:
            self._expected_reward_table[old_state][target_state][_index]+=1
        
        #favour exploitation a little bit more   
        self._epsilon *= 0.999
        if  self._epsilon < 0.8 and self._epsilon > 0.6:
            self.range_subtractor=2
        elif  self._epsilon < 0.6:
            self.range_subtractor=0
    def q_table(self):
        Q=self._expected_reward_table
        _mean_voltage=np.zeros((arc2.NUM_NON_FAIL_STATES,arc2.NUM_NON_FAIL_STATES))
        for i in range(arc2.NUM_NON_FAIL_STATES):
            for j in range(arc2.NUM_NON_FAIL_STATES):
                _mean_v_index = np.argmax([a for a in Q[i][j]])
                _mean_voltage[i,j]=self._voltages[_mean_v_index]
                if i==j:
                    _mean_voltage[i,j]=0  
         
        # save to csv file
        # np.savetxt('data.csv', Q.flatten(), delimiter=',')
        # loaded_array = np.loadtxt('data.csv')
        # loaded_matrix_3d = np.reshape(loaded_array, ((arc2.NUM_NON_FAIL_STATES,
        #                                         arc2.NUM_NON_FAIL_STATES,
        #                                         self._voltage_step)))
        # with open('data.csv', 'a') as f_out:
        #     f_out.write(Q[1])
        
        # with open('outfile.txt','wb') as f:
        #     for line in Q:
        #         np.savetxt(f, line, fmt='%.2f')
        print('Number of exploration=', self._exploration) 
        print('Number of exploitation=', self._exploitation) 
        print('Final epsilon=', self._epsilon)
        print(Q)
        print(_mean_voltage)  
         
def plot_hardware_distribution(hardware: arc2.Arc2HardwareSimulator):
    """Plot the hardware distribution for each state"""
    _num_samples = (arc2.MAX_VOLTAGE-arc2.MIN_VOLTAGE)/(SAMPLE_VOLTAGE_IN_MV/1000.0)
    _voltage = np.linspace(arc2.MIN_VOLTAGE,arc2.MAX_VOLTAGE,num=int(_num_samples))
    # plot actual using private data in hardware that we shouldn't
    # normally have access to
    for i, _func in enumerate(hardware._state_transitions):
        _ax = plt.subplot(2,2,i+1)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
       #_ax.set_facecolor("#000000")
        _ax.set_title(arc2.STATES[i])
        _ax.set_xlim([arc2.MIN_VOLTAGE, arc2.MAX_VOLTAGE])
        _ax.set_xlabel('Voltage')
        _ax.set_ylabel('Probability Density')
        _probs = np.array([_func(v) for v in _voltage])
        for j in range(arc2.NUM_STATES):
            _ax.plot(_voltage,_probs[:,j],label=f"{arc2.STATES[i]} to {arc2.STATES[j]}")
            print('\n')
        _ax.legend()
    plt.show()

def main(args):
    
    start = time.time()

    "Use a loop to run number of times representing running your algorithm on many wafers"
    _stat_report=[]
    _yield_lists=[]
    _failed_lists=[]
    _time_list=[]
    for i in range (args.number_wafers):
        """Simulate a test module applying voltages to a number of devices in a wafer"""
        fail_cons=[1.001,1.0013,1.001,1.00105,1.0012,1.0012]
        with open("mdp_param.json") as f:
            mdp_param = json.load(f)
        _arc2_hardware = arc2.Arc2HardwareSimulator(args.number_devices,mdp_param[MODEL_INDEX], fail_cons[MODEL_INDEX])
        if args.algorithm_to_use == "random":
            _arc_tester = RandomVoltageArc2Tester()
        elif args.algorithm_to_use == "randomwithrange":
            _arc_tester = RandomVoltageWithRangeKnowledgeArc2Tester()
        elif args.algorithm_to_use == "expuser":
            _arc_tester = ExperiencedUserTester(MODEL_INDEX)
        elif args.algorithm_to_use == "epsilon":
            _arc_tester = EpsilonGreedyTester(args.max_attempts,STEP_VOLTAGES, GAMMA)
        elif args.algorithm_to_use == "epsilon_c":
            _arc_tester = EpsilonGreedyCautious(args.max_attempts,STEP_VOLTAGES, GAMMA)
        else:
            raise RuntimeError("Unknown algorithm!")
        _report = _arc_tester.run(_arc2_hardware)
        if args.plot_hardware_dist:
            plot_hardware_distribution(_arc2_hardware)
        _successful_electroform = sum([d['success'] for d in _report])
        _state_I = sum([d['to state I'] for d in _report])
        _state_II = sum([d['to state II'] for d in _report])
        _failed_devices = sum([d['actual_state']=="FAIL" for d in _report])
        _yield=100*(_successful_electroform/ args.number_devices)
        
        
        if args.additional_info:
            print("Num of devices in state I : ", _state_I)
            print("Num of devices in state II : ", _state_II)
            print("Successful electroform in wafer number ",i+1,'is' ,_successful_electroform)
            print("Failed devices in wafer number ",i+1,'is', _failed_devices)
            print("The First Pass Yield(FPY) in wafer number  ",i+1,'is',_yield ,'% \n')
        
        _stat_report.append(_successful_electroform)
        _yield_lists.append(_yield)
        _failed_lists.append(_failed_devices)
        _time_list.append(sum([d['time_step'] for d in _report])/args.number_devices)
    _time_mean=np.mean(_time_list)
    _yield_mean=np.mean(_yield_lists) 
    _failed_mean=np.mean(_failed_lists)
    _success_mean= np.mean(_stat_report) 
    _success_std= np.std(_stat_report) 
    
    print("Devices in wafer: ", args.number_devices)
    print("Devices tested: ", len(_report))
    print('The average successful electroform for ',args.number_wafers,' wafers is','%.2f' % _success_mean,'and', '%.2f' % _success_std ,'std')
    print("Average Yield for",args.number_wafers,"wafers is",'%.2f' % _yield_mean,'%, ','%.2f' % np.std(_yield_lists),'%','std') 
    print('Average successful forming time','%.2f' % _time_mean,'%.2f' % np.std(_time_list),'std')
    print("Average failed devices for",'%.2f' % args.number_wafers,"wafers is",'%.2f' %( _failed_mean/10),'%, ','%.2f' % (np.std(_failed_lists)/10),'%','std')
    tt=(time.time()-start)/60
    print('It took', tt, 'minutes.')

def parser_setup(parser):
    """Setup command line arguments"""
    group_input = parser.add_argument_group("input options")
    group_input.add_argument(
        "-a", "--algorithm-to-use",
        help="Choose algorithm to use.",
        action="store",
        type=str,
        default=DEFAULT_ALGORITHM
    )
    group_input.add_argument(
        "-n", "--number-devices",
        help="Number of memristor devices in the wafer.",
        action="store",
        type=int,
        default=DEFAULT_NUMBER_DEVICES
    )
    group_input.add_argument(
        "-w", "--number-wafers",
        help="Number of wafer.",
        action="store",
        type=int,
        default= DEFAULT_NUMBER_WAFERS
    )
    group_input.add_argument(
        "-m", "--max-attempts",
        help="Maxiumum number of time steps to perform on a single device.",
        action="store",
        type=int,
        default= DEFAULT_MAX_ATTEMPT
    )

    group_output = parser.add_argument_group("output options")
    group_output.add_argument(
        "--plot-hardware-dist",
        help="Plot the hardware distribution.",
        action="store_true"
    )
    group_output = parser.add_argument_group("output options")
    group_output.add_argument(
        "--additional-info",
        help="Additional information(number of devices in each state, yield of each wafer..)",
        action="store_true"
    )
    group_output.add_argument(
        "--log-level",
        help="Specify the logging level.",
        action="store",
        type=str,
        choices=list(logging._nameToLevel.keys()),
        default=logging.ERROR,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser_setup(parser)
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)
    try:
        main(args)
    except BaseException as e:
        print(f"Exception: Failed to run module: {e} ({type(e).__name__})")

