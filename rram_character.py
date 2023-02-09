import numpy as np
import argparse
import logging
import random
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import arc2_simulator as arc2

SAMPLE_VOLTAGE_IN_MV = 5
DEFAULT_NUMBER_DEVICES = 1000
DEFAULT_ALGORITHM = "random"


class Arc2Tester(ABC):
    """This is the deliverable"""
    def run(self, hardware: arc2.Arc2HardwareSimulator) -> list:
        """Run test on hardware"""
        _report = []
        while True:
            # get current state of device from the hardware
            _current_state = 0 
            # select the state we want this to be, random but could be read from file
            # and don't choose the current state
            _possible_target_states = set(range(arc2.NUM_NON_FAIL_STATES))
            _possible_target_states.remove(_current_state)
            _target_state = 2#random.choice(list(_possible_target_states))
            # determine the voltage to apply
            _voltage, _pulse_duration = self.get_action(_current_state,_target_state)
            # apply to hardware a number of times representing pulse length
            for _ in range(_pulse_duration):
                hardware.apply_voltage(_voltage)
            _new_state = hardware.get_current_device_state()
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
                    'success': _target_state == _new_state
                }
            )
            # check to see if we have finished the wafer
            if not hardware.move_to_next_device():
                break
        return _report

    @abstractmethod
    def get_action(self, current_state, target_state) -> tuple:
        raise NotImplementedError("Must override get_action")


class RandomVoltageArc2Tester(Arc2Tester):
    """First attempt - let's just pick a voltage randomly"""
    def __init__(self):
        super().__init__()

    def get_action(self, current_state, target_state) -> tuple:
        action= random.randrange(arc2.MIN_VOLTAGE*10,arc2.MAX_VOLTAGE*10,5)/10 , 1
        if current_state == 0:
            action= random.randrange(0,arc2.MAX_VOLTAGE*10,5)/10 , 1
        elif current_state == 2:
            action= -random.randrange(0,arc2.MAX_VOLTAGE*10,5)/10 , 1
        return action
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

    def get_action(self, current_state, target_state) -> tuple:
        return self._lookup_voltage_cheat[current_state][target_state], 1


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
    """Simulate a test module applying voltages to a number of devices in a wafer"""
    _arc2_hardware = arc2.Arc2HardwareSimulator(args.number_devices)
    if args.algorithm_to_use == "random":
        _arc_tester = RandomVoltageArc2Tester()
    elif args.algorithm_to_use == "expuser":
        _arc_tester = ExperiencedUserTester()
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
    print("Successful electroform: ", _successful_electroform)
    print("Failed devices: ", _failed_devices)


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

