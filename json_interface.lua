require 'json'

-- from matter-js dump
function load_data_json(jsonfile)
    print(jsonfile)
    local data = json.load(jsonfile)  -- 1 indexed (num_balls, timesteps, data)
    -- data: {velocity{x,y}, mass, position{x,y}}
    local num_examples = #data
    local num_obj = #data[1]
    local T = #data[1][1]

    -- TODO: adapt to include other information
    for e=1,num_examples do
        for i=1,num_obj do
            for t=1,T do
                -- mutate the data itself
                local state = data[e][i][t]
                data[e][i][t] = {state.position.x, state.position.y,
                                state.velocity.x, state.velocity.y,
                                state.mass, 1}  -- TODO: 1 is the object id
            end
        end
    end
    data = torch.Tensor(data)
    return data
end


-- load_data_json('/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/physics_worlds/balls.json')
