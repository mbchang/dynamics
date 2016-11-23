import json
import pprint

jsonfile = 'mixed_n6_t60_ex50000_z_o_dras3_rda_chksize100_399.json'
x = json.load(open(jsonfile,'r'))

data = x['trajectories'][0]  # take first chunk

# len(data) = num_obj
first_step_for_each_object = [x[0] for x in data]
pprint.pprint(first_step_for_each_object)