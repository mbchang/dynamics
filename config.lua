local args = {

        -- datasaver
        position_normalize_constant=800,
        velocity_normalize_constant=50,
        angle_normalize_constant=2*math.pi,
        relative=true,
        masses={0.33, 1.0, 3.0, 1e30},
        rsi={px=1, py=2, vx=3, vy=4, a=5, av=6, m=7, oid=8},  -- raw state indicies
        si={px=1, py=2, vx=3, vy=4, a=5, av=6, m={7,10}, oid=11},  -- state indices
        permute_context=False,
        shuffle=False,
        maxwinsize=60,
        max_iters_per_json=37  -- TODO

        -- world params



        -- all the paths

    }




return args
