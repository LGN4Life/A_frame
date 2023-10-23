
import smr

dir_name = 'D:\\AwakeData\\Deep Array\\230413\\Record Node 101\\experiment1\\recording1\\smr triggers\\r4\\'
file_name = 'oe_tmj_230413_ori_003.csv'
file_name = dir_name + file_name
print(file_name)
smr_triggers = smr.load_smr_triggers(file_name)

print(smr_triggers.head())