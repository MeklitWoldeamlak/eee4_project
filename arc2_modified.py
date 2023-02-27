import numpy as np
import random

# Information/knowledge about the arc2
# that we know for sure and can be directly used
MAX_VOLTAGE = 5.0
MIN_VOLTAGE = -5.0
STATES = ["I", "II", "III", "FAIL"]
NUM_STATES = len(STATES)
NUM_NON_FAIL_STATES = NUM_STATES-1
A=1.5
B= 6.5
FAIL_STD_DISCOUNT= 0.0006


# Information about the device that we
# should not have access to except from
# observation and behaviour of through
# experimentation or gained knowledge
_DEVICE_FAIL_DEVIATION = 1.0
_TO_FAIL_STATE_TPS_PARAMS ={'max_vol':5,'fail_std':1}
_NON_FAIL_STATE_TPS_PARAMS = [
        [
            {'mean': 2.0, 'stdev': 0.75},    # I to II
            {'mean': 4.0, 'stdev': 0.75}    # I to III
        ],
        [
            {'mean': -3.0, 'stdev': 1.0},   # II to I
            {'mean':  3.0, 'stdev': 1.0}    # II to III
            
        ],
        [
            {'mean': -4.0, 'stdev': 0.75},   # III to I
            {'mean': -2.0, 'stdev': 0.75}    # III to II
            
        ]
    ]

def _normal_dist(prob: np.float32, mean: np.float32, std: np.float32):
    def _norm(x: np.float32) -> np.float32:
        return prob * (1.0/(std*(2.0*np.pi)**(0.5))) * np.exp(-0.5*((x-mean)/std)**2)
    return _norm


def _symmetic_normal_dist(prob: np.float32, abs_mean: np.float32, std: np.float32):
    _upper = _normal_dist(prob, abs_mean, std)
    _lower = _normal_dist(prob, -1.0*abs_mean, std)
    def _sym_norm(x: np.float32) -> np.float32:
        return (_upper(x) + _lower(x))/2
    return _sym_norm

def _symmetric_sigmoid_dist(prob: np.float32, a: np.float32, b: np.float32):
    def _sigmoid(x: np.float32) -> np.float32:
        z1 = np.exp(a*x+b)
        z2= np.exp(-a*x+b)        
        return (prob/ (1 + z1)) + (prob/ (1 + z2))
    return _sigmoid

def _transition_probability(current_state: int, model_params: list,fail_param:dict):
    if len(model_params) != 2:
        raise RuntimeError("Invalid configuration!")
    #prob=1.2
    #if current_state==1:
      #  prob=1.5
    _prob_functions = [_normal_dist(1,params['mean'],params['stdev']) for params in model_params]
    _prob_functions.append(_symmetic_normal_dist(0.5,fail_param['max_vol'],fail_param['fail_std']))
    def _transition(voltage) -> np.array:
        _probabilities = np.array([f(voltage) for f in _prob_functions])
        _sum_prob = np.sum(_probabilities)
        if _sum_prob > 1.0:
            raise RuntimeError("Invalid model specification!")
        return np.insert(_probabilities,current_state,1.0-_sum_prob)
    return _transition


class Arc2HardwareSimulator:
    """Models the actual hardware with a variable probability distribution
    
    * In this model, the probability distribution vary across devices in the wafer
    """
    def __init__(self,number_wafers:int, number_devices: int):
        self._number_devices = number_devices
        self._number_wafers=number_wafers
        self._fail_pram=_TO_FAIL_STATE_TPS_PARAMS
        self.mdp_param= _NON_FAIL_STATE_TPS_PARAMS
        self._vary_constant=0.5
        self._current_device = 0
        self._current_wafer = 0
        self._state_transitions = [_transition_probability(state,params, self._fail_pram)
                                    for state,params in enumerate(self.mdp_param)]
        self._state_transitions.append(lambda x: [0.0, 0.0,0.0, 1.0])
        self._device_state = [random.randrange(NUM_NON_FAIL_STATES) for _ in range(number_devices)]#change to 0 after confirming 
        

    def get_current_device_state(self):
        return self._device_state[self._current_device]

    def apply_voltage(self,voltage: np.float32):
        _current_state = self._device_state[self._current_device]
        _state_transition_probabilities = self._state_transitions[_current_state](voltage)
        _next_state = np.random.choice(list(range(NUM_STATES)),p=_state_transition_probabilities)
        self._device_state[self._current_device] = _next_state

    def move_to_next_device(self) -> bool:
        self._current_device += 1
        self._fail_pram['fail_std']=1-(self._current_device*FAIL_STD_DISCOUNT)
        self._state_transitions = [_transition_probability(state,params, self._fail_pram)
                                     for state,params in enumerate(self.mdp_param)]
        self._state_transitions.append(lambda x: [0.0, 0.0,0.0, 1.0])
      
        return self._current_device < self._number_devices
    
    def tested_wafer(self) -> int:
        return self._current_wafer  
    
    def move_to_next_wafer(self):
        self._current_device = 0
        self._current_wafer += 1
        p = np.random.random()
        if p < self._vary_constant:
            for i in range(len(self.mdp_param)):
                for j in range(len(self.mdp_param[i])):
                    self.mdp_param[i][j]['mean']+=0.0005 
                    
        else:
            for i in range(len(self.mdp_param)):
                for j in range(len(self.mdp_param[i])):
                    self.mdp_param[i][j]['stdev']-=0.0003
        self._state_transitions = [_transition_probability(state,params, self._fail_pram)
                                     for state,params in enumerate(self.mdp_param)]
        self._state_transitions.append(lambda x: [0.0, 0.0,0.0, 1.0])
        #return self._current_wafer < self._number_wafers
    
    
            


