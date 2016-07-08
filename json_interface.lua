require 'json'
local args = require 'config'
local tablex = require 'pl.tablex'


-- from matter-js dump
function load_data_json(jsonfile)
    print('Loading json file: '..jsonfile)
    local data = json.load(jsonfile)  -- 1 indexed (num_balls, timesteps, data)
    -- data: {velocity{x,y}, mass, position{x,y}}
    local num_examples = #data
    local num_obj = #data[1]
    local T = #data[1][1]
    assert(num_examples <= args.max_iters_per_json)

    -- TODO: adapt to include other information
    for e=1,num_examples do
        for i=1,num_obj do
            for t=1,T do
                -- mutate the data itself
                local state = tablex.deepcopy(data[e][i][t])
                data[e][i][t] = {}
                data[e][i][t][args.rsi.px] = state.position.x
                data[e][i][t][args.rsi.py] = state.position.y
                data[e][i][t][args.rsi.vx] = state.velocity.x
                data[e][i][t][args.rsi.vy] = state.velocity.y
                data[e][i][t][args.rsi.a] = state.angle
                data[e][i][t][args.rsi.av] = state.angularVelocity
                data[e][i][t][args.rsi.m] = state.mass
                data[e][i][t][args.rsi.oid] = 1
            end
        end
    end

    data = torch.Tensor(data)
    return data
end

function data2table(data)
    -- data: (bsize, num_obj, num_steps, dim)
    local num_examples = data:size(1)
    local num_obj = data:size(2)
    local T = data:size(3)

    local trajectories = {}
    for e=1,num_examples do
        trajectories[e] = {}
        for i=1,num_obj do
            trajectories[e][i] = {}
            for t=1,T do
                local state = data[e][i][t]
                trajectories[e][i][t] = {}
                trajectories[e][i][t].position = {x=state[args.rsi.px],
                                                  y=state[args.rsi.py]}
                trajectories[e][i][t].velocity = {x=state[args.rsi.vx],
                                                  y=state[args.rsi.vy]}
                trajectories[e][i][t].angle = state[args.rsi.a]
                trajectories[e][i][t].anglularVelocity = state[args.rsi.av]
                trajectories[e][i][t].mass = state[args.rsi.m]
            end
        end
    end
    return trajectories
end

-- to matter-js dump
function dump_data_json(data, jsonfile)
    -- data: (bsize, num_obj, num_steps, dim)
    local trajectories = data2table(data)
    json.save(jsonfile, trajectories)
    return trajectories
end


-- local d = load_data_json('/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/physics_worlds/balls.json')
-- dump_data_json(d, '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/physics_worlds/balls_dump.json')
-- local d2 = load_data_json('/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/physics_worlds/balls.json')
--
-- print((d-d2):sum())
