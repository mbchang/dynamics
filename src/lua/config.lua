--px: position x
--py: position y
--vx: velocity x
--vy: velocity y
--a: angle
--av: angular velocity
--m: mass {4.0, 1.0, 16.0, 1e30} (4)
--oid: object id {ball=1, obstacle=2, block=3} (3)
--os: object sizes {0.5, 1, 2} (3)
--g: gravity {on, off} (2)
--f: friction {on, off} (2)
--p: pairwise {on, off} (2)

-- total: 22

local args = {

        -- datasaver
        velocity_normalize_constant=60,
        angle_normalize_constant=math.pi, -- it should be 2*math.pi if you are in the range [0, 2pi]
        relative=true,
        masses={1.0, 5.0, 25.0, 1e30},  -- for now only the first two are used
        rsi={px=1, py=2, vx=3, vy=4, a=5, av=6, m=7, oid=8, os=9, g=10, f=11, p=12},  -- raw state indicies
        si={px=1, py=2, vx=3, vy=4, a=5, av=6, m={7,10}, oid={11,13}, os={14,16}, g={17,18}, f={19,20}, p={21,22}},  -- state indices
        permute_context=false,
        shuffle=true,
        maxwinsize=60,
        maxwinsize_long=120,
        max_iters_per_json=100,  -- TODO
        subdivide=true,
        object_base_size={ball=60, obstacle=80, block=60},  -- radius, length, block (note that this is block long length, whereas in js it is the short length!!)
        object_base_size_ids={60, 80, 60},
        object_base_size_ids_upper={60,80*math.sqrt(2)/2,math.sqrt(math.pow(60,2)+math.pow(60/3,2))},

        -- object_sizes={2/3, 1, 3/2}, -- multiplies object_base_size
        object_sizes={1/2, 1, 2}, -- multiplies object_base_size
        drastic_object_sizes={1/2, 1, 2}, -- multiplies object_base_size
        -- drastic_object_sizes={1/3, 1, 3}, -- multiplies object_base_size

        oids = {ball=1, obstacle=2, block=3},  -- {1=ball, 2=obstacle, 3=block},
        roids = {'ball', 'obstacle', 'block'},  -- reverse oids
        oid_ids = {1,2,3},  -- the values of oids
        boolean = {0,1},

        -- world params
        cx=400, -- 2*cx is width of world
        cy=300 -- 2*cy is height of world

        -- all the paths
        
    }

args.position_normalize_constant = math.max(args.cx,args.cy)*2
args.ossi = args.si.m[1]  -- object_state_start_index: CHANGE THIS WHEN YOU ADD STUFF TO RAW STATE INDICES!

-- args.object_base_size_ids = {}
-- for _,i in pairs(oid_ids) do 

-- end
return args
