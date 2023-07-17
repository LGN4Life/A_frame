import numpy as np
import itertools
import random
import re


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
        current_dic = tuning_functions[index]
        current_default = current_dic['default']
        # list of all defined default variables in the current dic
        current_var = current_dic['default']
        current_var = [sublist[0] for sublist in current_var]
        # add to this the list of tuned variables
        current_var = set(current_var + current_dic['tuning_type'])
        missing_list = list(set(unique_values_list) - set(current_var))
        missing_values = [getattr(default_stim, param) for param in missing_list]
        current_default.extend([[param, getattr(default_stim, param)] for param in missing_list])
        tuning_functions[index]['default'] = current_default
    return tuning_functions


def combine_config_strings(config_list, flag_list):
    full_config = '" '

    for flag_index, flag in enumerate(flag_list):
        print(f'current_flag = {flag}')
        var_config = ''
        for tc_index, config in enumerate(config_list):

            if var_config == '':
                flag_pattern = r'(-{1}' + flag + r' [^-]*)'
                current_config = re.findall(flag_pattern, config)
                if current_config:
                    var_config += current_config[0]
            else:
                flag_pattern = r'(-{1}' + flag + r')( [^-]*)'
                current_config = re.findall(flag_pattern, config)
                if current_config:
                    var_config += current_config[0][1]
        full_config += var_config
    full_config += '"'
    return full_config


class Bakers:
    def __init__(self, **kwargs):
        self.default_stimulus = Stimulus(**kwargs)


class Stimulus:
    def __init__(self, **kwargs):
        self.C = kwargs.get('C', 100)
        self.A = kwargs.get('A', 5)
        self.S = kwargs.get('S', 1)
        self.T = kwargs.get('T', 5)
        self.P = kwargs.get('P', 0)
        self.Z = kwargs.get('Z', [0, 0])
        self.sweep = kwargs.get('sweep', [1, 0, 5])
        self.wh = kwargs.get('wh', [5, 5])

    def print(self):
        for attr, value in self.__dict__.items():
            print(f"{attr} = {value}")


class TuningFunction:
    def __init__(self, tuning_params):
        self.iv = list(range(len(tuning_params['tuning_type'])))
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
            self.combo_list = list(itertools.product(*self.iv))
            random.shuffle(self.combo_list)
        else:
            raise ValueError("Invalid condition found")
        self.generate_config_string()

    def generate_config_string(self):

        for element_id in range(len(self.combo_list[0])):
            new_list = [element[element_id] for element in self.combo_list]
            result_string = ', '.join(str(item) for item in new_list)
            result_string = result_string.replace(" ", "")
            n = len(new_list)
            self.config_string += f"-{self.tuning_type[element_id]} {result_string} "


        if self.default_params is not None:

            for iv_index in range(len(self.default_params)):

                current_flag = self.default_params[iv_index][0]
                current_values = self.default_params[iv_index][1]
                current_values = [current_values] * n
                new_string = ", ".join(str(element) for element in current_values)
                self.config_string += f"-{current_flag} {new_string} "

        self.config_string = self.config_string.replace("[", "").replace("]", "")
        self.config_string = self.config_string.replace("(", "").replace(")", "")



