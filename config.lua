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
        object_sizes={2/3, 1, 3/2}, -- multiplies object_base_size
        drastic_object_sizes={1/2, 1, 2}, -- multiplies object_base_size
        oids = {ball=1, obstacle=2, block=3},  -- {1=ball, 2=obstacle, 3=block},
        roids = {'ball', 'obstacle', 'block'},  -- reverse oids
        oid_ids = {1,2,3},
        boolean = {0,1},

        -- world params
        cx=400, -- 2*cx is width of world
        cy=300 -- 2*cy is height of world

        -- all the paths
        
    }

args.position_normalize_constant = math.max(args.cx,args.cy)*2
args.ossi = args.si.m[1]  -- object_state_start_index: CHANGE THIS WHEN YOU ADD STUFF TO RAW STATE INDICES!

return args


        -- demo.offset = 5;  // world offset
        -- demo.config = {}
        -- demo.config.cx = 400;
        -- demo.config.cy = 300;
        -- demo.config.masses = [1, 5, 25]
        -- demo.config.mass_colors = {'1':'#C7F464', '5':'#FF6B6B', '25':'#4ECDC4'}
        -- demo.config.sizes = [0.5, 1, 2]  // multiples
        -- demo.config.object_base_size = {'ball': 60, 'obstacle': 120, 'block': 40 }  // radius of ball, side of square obstacle, long side of block
        -- demo.config.objtypes = ['ball', 'obstacle', 'block']  // squares are obstacles
        -- demo.config.g = 0 // default? [0,1] Or should we make this a list? The index of the one hot. 0 is no, 1 is yes
        -- demo.config.f = 0 // default? [0,1]
        -- demo.config.p = 0 // default? [0,1,2]
        -- demo.config.max_velocity = 60

        -- demo.cx = demo.config.cx;
        -- demo.cy = demo.config.cy;
        -- demo.width = 2*demo.cx
        -- demo.height = 2*demo.cy