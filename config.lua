local args = {

        -- datasaver
        position_normalize_constant=800,
        velocity_normalize_cnstant=50,
        relative=true,
        masses={0.33, 1.0, 3.0, 1e30},
        rsi={px=1, py=2, vx=3, vy=4, m=5, oid=6},  -- raw state indicies
        si={px=1, py=2, vx=3, vy=4, m={5,8}, oid=9},  -- state indices
        permute_context=False,
        batch_size=4,
        shuffle=False,
        maxwinsize=60

    }

return args
