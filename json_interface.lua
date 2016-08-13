require 'json'
local args = require 'config'
local tablex = require 'pl.tablex'
local stringx = require 'pl.stringx'

-- balls_n8_t60_s50000_mjs2
function get_global_params(jsonfile)
    local g=0  -- false by default
    local f=0  -- false by default
    local p=0  -- false by default
    if stringx.count(jsonfile, '_g') == 1 or not(string.find(jsonfile, 'tower') == nil) then  -- if tower is in jsonfile, then also turn gravity on!
        g = 1
    end
    if stringx.count(jsonfile, '_f') == 1 then
        f = 1
    end
    if stringx.count(jsonfile, '_p') == 1 then
        p = 1
    end
    return g, f, p
end


-- from matter-js dump
function load_data_json(jsonfile)
    print('Loading json file: '..jsonfile)
    local data = json.load(jsonfile)  -- 1 indexed (num_balls, timesteps, data)
    local trajectories = data.trajectories
    -- trajectories: {velocity{x,y}, mass, position{x,y}}
    local num_examples = #trajectories
    local num_obj = #trajectories[1]
    local T = #trajectories[1][1]
    assert(num_examples <= args.max_iters_per_json)

    local g,f,p = get_global_params(jsonfile)

    -- TODO: adapt to include other information
    for e=1,num_examples do
        for i=1,num_obj do
            for t=1,T do
                -- mutate the trajectories itself
                local state = tablex.deepcopy(trajectories[e][i][t])
                trajectories[e][i][t] = {}
                trajectories[e][i][t][args.rsi.px] = state.position.x
                trajectories[e][i][t][args.rsi.py] = state.position.y
                trajectories[e][i][t][args.rsi.vx] = state.velocity.x
                trajectories[e][i][t][args.rsi.vy] = state.velocity.y
                trajectories[e][i][t][args.rsi.a] = state.angle
                trajectories[e][i][t][args.rsi.av] = state.angularVelocity
                trajectories[e][i][t][args.rsi.m] = state.mass
                trajectories[e][i][t][args.rsi.oid] = args.oids[state.objtype]
                trajectories[e][i][t][args.rsi.os] = state.sizemul
                trajectories[e][i][t][args.rsi.g] = g
                trajectories[e][i][t][args.rsi.f] = f
                trajectories[e][i][t][args.rsi.p] = p
            end
        end
    end

    trajectories = torch.Tensor(trajectories)
    return trajectories
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
                trajectories[e][i][t].angularVelocity = state[args.rsi.av]
                trajectories[e][i][t].mass = state[args.rsi.m]
                trajectories[e][i][t].objtype = args.roids[state[args.rsi.oid]]
                trajectories[e][i][t].sizemul = state[args.rsi.os]
            end
        end
    end
    return trajectories
end
    
-- to matter-js dump
function dump_data_json(data, jsonfile)
    -- data: (bsize, num_obj, num_steps, dim)
    local trajectories = data2table(data)
    json.save(jsonfile, {trajectories=trajectories})
    return trajectories
end


-- local d = load_data_json('../data/tower_n8_t75_ex1_m_rd/jsons/tower_n8_t75_ex1_m_rd_chksize1_0.json')
-- dump_data_json(d, '../data/tower_n8_t75_ex1_m_rd/jsons/tower_n8_t75_ex1_m_rd_chksize1_0_dump.json')
-- local d2 = load_data_json('../data/tower_n8_t75_ex1_m_rd/jsons/tower_n8_t75_ex1_m_rd_chksize1_0_dump.json')
