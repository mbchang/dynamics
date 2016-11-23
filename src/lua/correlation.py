import numpy as np 
import json
import pprint

# parses json file
def read_json(jsonfile):
    data = json.load(open(jsonfile,'r'))
    trajectories =  data['trajectories']  # (bsize, num_obj, num_step)

    for ex in range(len(trajectories)):
        for obj in range(len(trajectories[ex])):
            trajectory = trajectories[ex][obj]
            positions = [x['position'] for x in trajectory]
            velocities = [x['velocity'] for x in trajectory]

            # convert to numpy array
            positions = [sorted(x.items(), key=lambda x: x[0]) for x in positions]
            print positions



if __name__=='__main__':
    read_json('/Users/MichaelChang/Dropbox (MIT Solar Car Team)/MacHD/Documents/Important/MIT/Research/SuperUROP/Code/dynamics/opmjlogs/tower_n4_t120_ex25000_rd_stable__tower_n4_t120_ex25000_rd_stable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps0_modelbffobj_lambda1000/tower_n4_t120_ex25000_rd_stablepredictions/gt_batch94.json')