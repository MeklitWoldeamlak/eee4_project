import numpy as np
import argparse
import logging
import random
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import arc2_simulator2 as arc2

SAMPLE_VOLTAGE_IN_MV = 5
DEFAULT_NUMBER_DEVICES = 1000
DEFAULT_NUMBER_WAFERS = 2
DEFAULT_MAX_ATTEMPT =10
STEP_VOLTAGES=20
DEFAULT_ALGORITHM = "random"
GAMMA = 0.7 #discount factor
ALPHA = 0.9 #learning factor

class Arc2Tester(ABC):
    """This is the deliverable"""
    
    def run(self,hardware: arc2.Arc2HardwareSimulator) -> list:
        """Run test on hardware"""
        _report = []
        _transition_record= np.array(np.zeros([3,4,20]))
        while True:
            # get current state of device from the hardware
            _current_state = hardware.get_current_device_state()
            # select the state we want this to be, random but could be read from file
            # and don't choose the current state
            _possible_target_states = set(range(arc2.NUM_NON_FAIL_STATES))
            _possible_target_states.remove(_current_state)
            _target_state = random.choice(list(_possible_target_states))
            # determine the voltage to apply
            for i in range(args.max_attempts):
                _action=self.get_action(_current_state,_target_state)
                _voltage =_action['voltage']
                _pulse_duration = _action['pulse_duration']
                # apply to hardware a number of times representing pulse length
                for _ in range(_pulse_duration):
                    hardware.apply_voltage(_voltage)
                _new_state = hardware.get_current_device_state()
                if _new_state ==_target_state or arc2.STATES[_new_state]=="FAIL":
                    break
               # self.update(_current_state,_target_state,_action,_new_state)
                _current_state=_new_state
                
            #_action_index =  int((_voltage+5)/0.5)
            #_transition_record[_current_state][_new_state][_action_index]+=1
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
                    'table': _transition_record
                }
            )
           
            # check to see if we have finished the wafer
            if not hardware.move_to_next_device():
                break
        return _report
    
    

    @abstractmethod
    def get_action(self, current_state, target_state) -> dict:
        raise NotImplementedError("Must override get_action")
    def update(self, old_state:int, target_state:int, action:dict, new_state:int ):
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
    #random.uniform(arc2.MIN_VOLTAGE,arc2.MAX_VOLTAGE), 1


class ExperiencedUserTester(Arc2Tester):
    """Second attempt - a knowledge user has determined a good set of values
    
    A user has determined, having electroformed many devices over the years,
    a good set of voltages to use depending on what we are trying to do
    """
    def __init__(self):
        super().__init__()
        self._lookup_voltage_cheat = [
            [ 0.0,  2.0, 4.0, 0.0],
            [-3.0,  0.0, 3.0, 0.0],
            [-4.0, -2.0, 0.0, 0.0],
            [ 0.0,  0.0, 0.0, 0.0]]

    def get_action(self, current_state, target_state) -> dict:
        return {'voltage': self._lookup_voltage_cheat[current_state][target_state],
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
        _voltage_inc=(arc2.MAX_VOLTAGE - arc2.MIN_VOLTAGE)/float(voltage_step) #float voltage increment (0.5)
        self._voltage_step= voltage_step+1 # total number of actions(20)
        self._voltages= [arc2.MIN_VOLTAGE+i*_voltage_inc for i in range(self._voltage_step)] #actual value of volatges
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
        for i in range(args.max_attempts):
             
            p = np.random.random()
            if p < self._epsilon:
                _voltage_index = np.random.choice(self._voltage_step)
            else:
                _voltage_index = np.argmax([a for a in self._expected_reward_table[current_state][target_state]])
            x = _actions[_voltage_index]
            action={'voltage': self._voltages[_voltage_index],
                'pulse_duration':1}
            
        
        return {'voltage': self._voltages[_voltage_index],
                'pulse_duration':1,
                'voltage_index':_voltage_index}
           
        
    def update(self, old_state:int, target_state:int, action:dict, new_state:int ):
        """update the expected reward table
        add 1 to self_exploration every time we explore(take single action)
        """
        self._exploration += 1
        _index= int(action['voltage']/_voltage_inc)
        if new_state==target_state :
            self._expected_reward_table[old_state][target_state][_index]+=10
        elif new_state==3 :#fail
            self._expected_reward_table[old_state][target_state][_index]-=20
        else:
            self._expected_reward_table[old_state][target_state][_index]+=1
            
            
       ## self._action_value_est = (1 - 1.0 / self._exploration)*self._action_value_est + 1.0 / self._exploration * action['voltage']
        ## self._expected_reward_table[old_state][new_state][_voltage_index]=self._action_value_est
        
        #favour exploitation a little bit more   
        self._epsilon *= self._gamma
                
        
def plot_hardware_distribution(hardware: arc2.Arc2HardwareSimulator):
    """Plot the hardware distribution for each state"""
    _num_samples = (arc2.MAX_VOLTAGE-arc2.MIN_VOLTAGE)/(SAMPLE_VOLTAGE_IN_MV/1000.0)
    _voltage = np.linspace(arc2.MIN_VOLTAGE,arc2.MAX_VOLTAGE,num=int(_num_samples))
    # plot actual using private data in hardware that we shouldn't
    # normally have access to
    for i, _func in enumerate(hardware._state_transitions):
        _ax = plt.subplot(2,2,i+1)
        _ax.set_title(arc2.STATES[i])
        _ax.set_xlim([arc2.MIN_VOLTAGE, arc2.MAX_VOLTAGE])
        _probs = np.array([_func(v) for v in _voltage])
        for j in range(arc2.NUM_STATES):
            _ax.plot(_voltage,_probs[:,j],label=f"{arc2.STATES[i]} to {arc2.STATES[j]}")
        _ax.legend()
    plt.show()



def main(args):
    
    "Use a loop to run number of times representing running your algorithm on many wafers"
    _stat_report=[]
    for i in range (args.number_wafers):
        """Simulate a test module applying voltages to a number of devices in a wafer"""
        _arc2_hardware = arc2.Arc2HardwareSimulator(args.number_devices)
        if args.algorithm_to_use == "random":
            _arc_tester = RandomVoltageArc2Tester()
        elif args.algorithm_to_use == "randomwithrange":
            _arc_tester = RandomVoltageWithRangeKnowledgeArc2Tester()
        elif args.algorithm_to_use == "expuser":
            _arc_tester = ExperiencedUserTester()
        elif args.algorithm_to_use == "epsilon":
            _arc_tester = EpsilonGreedyTester(args.max_attempts,STEP_VOLTAGES, GAMMA)
        else:
            raise RuntimeError("Unknown algorithm!")
        if args.plot_hardware_dist:
            plot_hardware_distribution(_arc2_hardware)
        _report = _arc_tester.run(_arc2_hardware)
        _successful_electroform = sum([d['success'] for d in _report])
        _state_I = sum([d['to state I'] for d in _report])
        _state_II = sum([d['to state II'] for d in _report])
        _failed_devices = sum([d['actual_state']=="FAIL" for d in _report])
    # _target=[d['target_state'] for d in _report]
    # _current=[d['current_state'] for d in _report]
    # _new=[d['actual_state'] for d in _report]
        
        if args.list_states:
            print("Num of devices in state I : ", _state_I)
            print("Num of devices in state II : ", _state_II)
        
        print("Devices in wafer: ", args.number_devices)
        print("Devices tested: ", len(_report))
        #print("State of Devices : ", _current)
        #print(" New State of Devices : ", _new)
        #print("Voltage : ", [d['voltage_applied'] for d in _report])
        print("Successful electroform in wafer number ",i+1,'is' ,_successful_electroform)
        print("Failed devices in wafer number ",i+1,'is', _failed_devices)
        print("The First Pass Yield(FPY) in wafer number  ",i+1,'is', 100*(_successful_electroform/ args.number_devices),'% \n')
        table= [d['table'] for d in _report]
        #print(table[-1])
        
        _stat_report.append(_successful_electroform)
    _mean= np.mean(_stat_report)   
    _std= np.std(_stat_report) 
    print('The average  and standard deviation of Successful electroform in a number of wafer are ',_mean,'and',  _std ,'respectively')


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
        "--list-states",
        help="Number of devices in each state after electroform.",
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

