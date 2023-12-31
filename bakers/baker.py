import numpy as np
import itertools
import random
import re
import math


def check_flags(tuning_functions, default_stim):
    unique_values = set()

    for dictionary in tuning_functions:
        dictionary.keys()
        values = dictionary['tuning_type']
        current_var = dictionary['default']
        current_var = [sublist[0] for sublist in current_var]
        values = set(values + current_var)
        unique_values.update(values)

    # make a list of all tuned parameters
    unique_values_list = list(unique_values)
    # for all tuning_functions, add needed default values to tuning_functions['default']
    for index in range(len(tuning_functions)):
        print(f"index = {index}")
        current_dic = tuning_functions[index]
        current_default = current_dic['default']

        # list of all defined default variables in the current dic
        current_var = current_dic['default']
        current_var = [sublist[0] for sublist in current_var]
        # add to this the list of tuned variables
        current_var = set(current_var + current_dic['tuning_type'])
        missing_list = list(set(unique_values_list) - set(current_var))

        #missing_list = [item.replace('-', '') for item in missing_list]

        # missing_values = [getattr(default_stim, param) for param in missing_list]
        current_default.extend([[param, getattr(default_stim, param.replace('-', ''))[0]] for param in missing_list])
        tuning_functions[index]['default'] = current_default

    return tuning_functions


def combine_config_strings(config_list, flag_list):
    full_config = '"'

    for flag_index, flag in enumerate(flag_list):
        print(f'current_flag = {flag}')
        var_config = ''
        for tc_index, config in enumerate(config_list):

            if var_config == '':
                flag_pattern = r'(' + flag + r'(\s?-?\d+\.?\d*,?)*)'

                current_config = re.findall(flag_pattern, config)[0][0]
                if current_config:
                    var_config += current_config

            else:
                current_flag = flag
                pattern = current_flag + r'(.*?)(?=-\D|$)'
                current_config = re.findall(pattern, config)
                current_config = ',' + current_config[0]
                current_config = current_config.replace(" ","")
                if current_config:
                    var_config += current_config
            # print(f"tc_index = {tc_index}")
            # numbers = re.findall(r'\d+\.\d+|\d+', current_config)
            # count = len(numbers)
            # print(f" {flag} has {count} numbers and {count / var_length[flag_index]} elements")
            # breakpoint()

        print(f'var_config =  {var_config}')
        # numbers = re.findall(r'\d+\.\d+|\d+', var_config)
        # count = len(numbers)
        # print(f" Final Count: {flag} has {count} numbers and {count/var_length[flag_index]} elements")
        # breakpoint()

        if var_config != '':
            var_config = ' ' + var_config
            full_config += var_config
    # full_config += ' "'
    return full_config


def randomize_trials(input_string, condition_string, var_list, num_trials):

    # Define the regex pattern
    # pattern = r'(-{1,2}\D+)(-?\d+\.?\d*,?)*'
    pattern = r'(-{1,2}\D+ .*?)(?=-\D|$)'
    # Find all matches using the pattern
    matches = re.findall(pattern, input_string)
    # Create a dictionary to store flag-value pairs
    flag_values = {}

    # Group the matches by flag
    flag_pattern = r'(-{1,2}[a-zA-Z]+)\s(.+)'

    for match in matches:

        flag_matches = re.findall(flag_pattern, match)
        print(f" match = {match}")
        print(f'flag_matches = {flag_matches}')
        flag = flag_matches[0][0]
        flag = flag.rstrip()
        values = flag_matches[0][1].split(',')
        # find the current flag in var_list and use the index to get the current_var_length
        flag_index = var_list['flags'].index(flag)
        current_var_length = var_list['var_length'][flag_index]
        values = [values[i:i + current_var_length] for i in range(0, len(values), current_var_length)]

        if flag in flag_values:
            flag_values[flag].append(values)
        else:
            flag_values[flag] = values
    # Randomize the order of values within each flag

    indices = np.random.permutation(num_trials)
    condition_string = [condition_string[i] for i in indices]

    for flag in flag_values:
        print(f"current flag  = {flag}")
        print(f"flag_values =  {flag_values[flag]}")
        temp = [i for i in indices]
        flag_values[flag] = [flag_values[flag][i] for i in indices]
        flag_values[flag] = flatten_list(flag_values[flag])

    # Create a new string with randomized flag-value pairs
    new_string = '" '
    for flag in flag_values:
        values_list = flag_values[flag]
        values_str = ', '.join(item for item in values_list)

        values_str = values_str.replace(" ", "")
        new_string += f"{flag} {values_str} "

    new_string += '"'

    # Print the randomized string
    return new_string, condition_string


def flatten_list(lst):
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened


def sweep_edge(leading_position, leading_phase, width, direction, sf, orientation):
    # for each combo of parameters (leading_position, width and direction) calculate the center
    # location (x and y) for a 2d grating
    center_position = [0]*len(width)
    center_phase = [0] * len(width)




    for trial_index in range(len(width)):
        # calculate x displacement based on direction and width
        direction_rad = math.radians(direction[trial_index])

        leading_phase[trial_index] = math.radians(leading_phase[trial_index])
        ori_rad = math.radians(orientation[trial_index])
        # Calculate the sine of the angle
        displace_x = sf * np.cos(direction_rad) * width[trial_index][0]/2
        displace_y = sf * np.sin(direction_rad) * width[trial_index][1]/2
        center_position[trial_index] = [leading_position[trial_index][0] - displace_x,
                                        leading_position[trial_index][1] - displace_y]
        # calculate center phase in order to keep the phase of the leading edge constant
        # across changes in width
        displace_phase_x = 2*np.pi * sf * np.cos(direction_rad) * width[trial_index][0]/2
        displace_phase_x = np.sin(ori_rad) * (displace_phase_x % (2*np.pi))
        displace_phase_y = 2*np.pi * sf * np.sin(direction_rad) * width[trial_index][1]/2
        displace_phase_y = np.cos(ori_rad) * (displace_phase_y % (2 * np.pi))
        center_phase[trial_index] = leading_phase[trial_index] + displace_phase_x - displace_phase_y
        # breakpoint()
    return center_position, center_phase


class Bakers:
    def __init__(self, **kwargs):
        self.default_stimulus = Stimulus(**kwargs)


class Stimulus:
    def __init__(self, **kwargs):
        self.C = kwargs.get('C', [[100.0], 1])
        self.A = kwargs.get('A', [[5.0], 1])
        self.S = kwargs.get('S', [[1.0], 1])
        self.T = kwargs.get('T', [[5.0], 1])
        self.O = kwargs.get('0', [[150], 1])
        self.P = kwargs.get('P', [[0.0], 1])
        self.Z = kwargs.get('Z', [[0, 0], 2])
        self.sweep = kwargs.get('sweep', [[1.0, 0.0, 0], 3])
        self.wh = kwargs.get('wh', [[5.0, 5.0], 2])


    def print(self):
        for attr, value in self.__dict__.items():
            print(f"{attr} = {value}")


class TuningFunction:
    def __init__(self, tuning_params):
        self.iv = [None]*len(tuning_params['values'])
        self.tuning_type = tuning_params['tuning_type']
        self.config_string = ''
        self.tuning_params = tuning_params['param_type']
        self.default_params = tuning_params['default']

        if tuning_params['param_type'] == 'param':
            for tuning_index in range(len(self.tuning_type)):
                min_value = tuning_params['values'][tuning_index][0]
                max_value = tuning_params['values'][tuning_index][1]
                step = tuning_params['values'][tuning_index][2]
                step = (max_value - min_value) / step
                self.iv[tuning_index] = np.arange(min_value, max_value+step, step).tolist()
                self.iv[tuning_index] = [x for x in self.iv[tuning_index]]
            self.combo_list = list(itertools.product(*self.iv))


            random.shuffle(self.combo_list)

        elif tuning_params['param_type'] == 'list':
            for tuning_index in range(len(self.tuning_type)):
                self.iv[tuning_index] = tuning_params['values'][tuning_index]
            # breakpoint()
            contains_none = any(item is None for item in self.iv)
            self.combo_list = list(itertools.product(*self.iv))
            random.shuffle(self.combo_list)

        elif tuning_params['param_type'] == 'combo':
            for tuning_index  in range(len(tuning_params['values'])):
                #self.iv[tuning_index] = [tuning_params['values'][trial_index][tuning_index] for trial_index in range(len(tuning_params['values']))]
                self.iv[tuning_index] = tuning_params['values'][tuning_index]

            self.combo_list = list(zip(*self.iv))



        else:
            raise ValueError("Invalid condition found")
        # print(len(self.combo_list ))
        # breakpoint()
        self.num_trials = len(self.combo_list)
        self.condition = tuning_params['condition'] * len(self.combo_list)
        self.generate_config_string()
        # breakpoint()

    def generate_config_string(self):
        n = self.num_trials
        for element_id in range(len(self.combo_list[0])):
            new_list = [element[element_id] for element in self.combo_list]
            if isinstance(new_list[0], list):
                new_list = flatten_list(new_list)
            result_string = ', '.join("{:.2f}".format(item) for item in new_list)
            result_string = result_string.replace(" ", "")
            self.config_string += f"{self.tuning_type[element_id]} {result_string} "
        if self.default_params is not None:

            for iv_index in range(len(self.default_params)):
                current_flag = self.default_params[iv_index][0]
                current_values = self.default_params[iv_index][1]
                current_values = [current_values] * n
                new_string = ", ".join(str(element) for element in current_values)
                new_string = new_string.replace(" ", "")
                self.config_string += f"{current_flag} {new_string} "
                # if type(current_values[0]) == list:
                #     self.var_length.append(len(current_values[0]))
                # else:
                #     self.var_length.append(1)
        self.config_string = self.config_string.replace("[", "").replace("]", "")
        self.config_string = self.config_string.replace("(", "").replace(")", "")



