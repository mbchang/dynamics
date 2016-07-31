require 'nn'
require 'rnn'
require 'torch'
require 'nngraph'
require 'Base'
require 'IdentityCriterion'
require 'data_utils'
require 'modules'

nngraph.setDebug(true)

-- with a bidirectional lstm, no need to put a mask
-- however, you can have variable sequence length now!
function init_network(params)
    -- encoder produces: (bsize, rnn_inp_dim)
    -- decoder expects (bsize, 2*rnn_hid_dim)

    local layer, sequencer_type, dcoef
    if params.model == 'lstmobj' then
        layer = nn.LSTM(params.rnn_dim,params.rnn_dim)
        sequencer_type = nn.BiSequencer
        dcoef = 2
    elseif params.model == 'gruobj' then
        layer = nn.GRU(params.rnn_dim,params.rnn_dim)
        sequencer_type = nn.BiSequencer
        dcoef = 2
    elseif params.model == 'ffobj' then
        layer = nn.Linear(params.rnn_dim, params.rnn_dim)
        sequencer_type = nn.Sequencer
        dcoef = 1
    elseif params.model == 'bffobj' then
        layer = nn.Linear(params.rnn_dim, params.rnn_dim)
        sequencer_type = nn.Sequencer
        dcoef = 1
    else
        error('unknown model')
    end

    local encoder = init_object_encoder(params.input_dim, params.rnn_dim)
    local decoder = init_object_decoder_with_identity(dcoef*params.rnn_dim, 
                                                        params.layers,
                                                        params.num_past, 
                                                        params.num_future,
                                                        params.object_dim)

    local step = nn.Sequential()
    step:add(encoder)
    for i = 1,params.layers do
        step:add(layer:clone())  -- same param initial, but weights not shared
        step:add(nn.ReLU())
        if mp.batch_norm then 
            step:add(nn.BatchNormalization(params.rnn_dim))
        end
    end

    local sequencer = sequencer_type(step)
    sequencer:remember('neither')

    local net = nn.Sequential()
    local branches = nn.ParallelTable()
    local pairwise = nn.Sequential()
    pairwise:add(sequencer)
    -- input table of (bsize, 2*d_hid) of length seq_length
    -- output: tensor (bsize, 2*d_hid)
    pairwise:add(nn.CAddTable())
    branches:add(pairwise)
    branches:add(nn.Identity())  -- for focus object identity
    net:add(branches)
    net:add(decoder)
    return net
end


--------------------------------------------------------------------------------
--############################################################################--
--------------------------------------------------------------------------------

-- Now create the model class
local model = {}
model.__index = model

function model.create(mp_, preload, model_path)
    local self = {}
    setmetatable(self, model)
    self.mp = mp_

    assert(self.mp.input_dim == self.mp.object_dim * self.mp.num_past)
    assert(self.mp.out_dim == self.mp.object_dim * self.mp.num_future)
    if preload then
        print('Loading saved model.')
        local checkpoint = torch.load(model_path)
        self.network = checkpoint.model.network:clone()
        self.criterion = checkpoint.model.criterion:clone()
        self.identitycriterion = checkpoint.model.identitycriterion:clone()
        if self.mp.cuda then
            self.network:float()
            self.criterion:float()
            self.identitycriterion:float()
        end
    else
        self.criterion = nn.MSECriterion(false)  -- not size averaging!
        self.identitycriterion = nn.IdentityCriterion()
        self.network = init_network(self.mp)
        if self.mp.cuda then
            self.network:cuda()
            self.criterion:cuda()
            self.identitycriterion:cuda()
        end
    end
    -- print(self.network)
    -- assert(false)

    self.theta = {}
    self.theta.params, self.theta.grad_params = self.network:getParameters()

    collectgarbage()
    return self
end

function model:unpack_batch(batch, sim)
    local this, context, y, context_future, mask = unpack(batch)
    local x = {this=this,context=context}

    -- unpack inputs
    local this_past     = convert_type(x.this:clone(), mp.cuda)
    local context       = convert_type(x.context:clone(), mp.cuda)
    local this_future   = convert_type(y:clone(), mp.cuda)

    -- reshape
    this_past:resize(this_past:size(1), this_past:size(2)*this_past:size(3))
    context:resize(context:size(1), context:size(2), context:size(3)*context:size(4))
    this_future:resize(this_future:size(1),this_future:size(2)*this_future:size(3))

    assert(this_past:size(1) == mp.batch_size and
            this_past:size(2) == mp.input_dim,
            'Your batch size or input dim is wrong')  -- TODO RESIZE THIS
    assert(context:size(1) == mp.batch_size and
            context:size(2)==torch.find(mask,1)[1]
            and context:size(3) == mp.input_dim)

    assert(this_future:size(1) == mp.batch_size and
            this_future:size(2) == mp.out_dim)  -- TODO RESIZE THIS

    -- here you have to create a table of tables
    -- this: (bsize, input_dim)
    -- context: (bsize, mp.seq_length, dim)
    local input = {}
    for t=1,torch.find(mask,1)[1] do  -- not actually mp.seq_length!
        table.insert(input, {this_past,torch.squeeze(context[{{},{t}}])})
    end

    ------------------------------------------------------------------
    -- here do the local neighborhood thing
    -- TODO! change nbrhd to flag
    if self.mp.nbrhd then  
        print('hi')
        self.neighbor_masks = self:select_neighbors(input)  -- this gets updated every batch!
    else
        self.neighbor_masks = {}  -- don't mask out neighbors
        for i=1,#input do
            table.insert(self.neighbor_masks, torch.ones(mp.batch_size))
        end
    end
    input = self:apply_neighbor_mask(input, self.neighbor_masks)
    ------------------------------------------------------------------

    return {input, this_past}, this_future
end

-- in: model input: table of length num_context-1 of {(bsize, num_past*obj_dim),(bsize, num_past*obj_dim)}
-- out: {{indices of neighbors}, {indices of non-neighbors}}
-- maybe I can output a mask? then I can rename this function to neighborhood_mask
function model:select_neighbors(input)
    local threshold = config_args.neighborhood*config_args.circle_radius
    local neighbor_masks = {}
    for i, pair in pairs(input) do
        -- reshape
        local this = pair[1]:clone():resize(mp.batch_size, mp.num_past, mp.object_dim)
        local context = pair[2]:clone():resize(mp.batch_size, mp.num_past, mp.object_dim)

        -- compute where they will be in the next timestep
        local this_pos_next = self:update_position_one(this)
        local context_pos_next = self:update_position_one(context)

        -- compute euclidean distance between this_pos_next and context_pos_next
        local euc_dist_next = torch.squeeze(self:euc_dist(this_pos_next, context_pos_next)) -- (bsize)
        euc_dist_next = euc_dist_next * config_args.position_normalize_constant  -- turn into absolute coordinates

        -- find the indices in the batch for neighbors and non-neighbors
        local neighbor_mask = euc_dist_next:le(threshold):float()  -- 1 if neighbor 0 otherwise
        table.insert(neighbor_masks, neighbor_mask)
    end

    return neighbor_masks
end

-- we mask out this as well, because it is as if that interaction didn't happen
function model:apply_neighbor_mask(input, neighbor_mask)
    assert(#neighbor_mask == #input)
    for i, x in pairs(input) do 
        if type(x) == 'table' then
            x[1] = torch.cmul(x[1],neighbor_mask[i]:view(mp.batch_size,1):expandAs(x[1]))
            x[2] = torch.cmul(x[2], neighbor_mask[i]:view(mp.batch_size,1):expandAs(x[2]))
        else
            x = torch.cmul(x, neighbor_mask[i]:view(mp.batch_size,1):expandAs(x):float())
            input[i] = x -- it doesn't actually automatically mutate
        end
    end
    return input
end

-- Input to fp
-- {
--   1 : DoubleTensor - size: 4x2x9
--   2 : DoubleTensor - size: 4x2x2x9
--   3 : DoubleTensor - size: 4x48x9
--   4 : DoubleTensor - size: 4x2x48x9
--   5 : DoubleTensor - size: 10
-- }
function model:fp(params_, batch, sim)
    if params_ ~= self.theta.params then self.theta.params:copy(params_) end
    self.theta.grad_params:zero()  -- reset gradient

    local input, this_future = self:unpack_batch(batch, sim)

    local prediction = self.network:forward(input)

    local p_pos, p_vel, p_ang, p_ang_vel, p_obj_prop =
                        unpack(split_output(self.mp):forward(prediction))
    local gt_pos, gt_vel, gt_ang, gt_ang_vel, gt_obj_prop =
                        unpack(split_output(self.mp):forward(this_future))

    local loss_vel = self.criterion:forward(p_vel, gt_vel)
    local loss_ang_vel = self.criterion:forward(p_ang_vel, gt_ang_vel)
    local loss = loss_vel + loss_ang_vel
    loss = loss/(p_vel:nElement()+p_ang_vel:nElement()) -- manually do size average

    collectgarbage()
    return loss, prediction
end


-- local p_pos, p_vel, p_obj_prop=split_output(params):forward(prediction)
-- local gt_pos, gt_vel, gt_obj_prop=split_output(params):forward(this_future)
-- a lot of instantiations of split_output
function model:bp(batch, prediction, sim)
    self.theta.grad_params:zero() -- the d_parameters
    local input, this_future = self:unpack_batch(batch, sim)

    local splitter = split_output(self.mp)

    local p_pos, p_vel, p_ang, p_ang_vel, p_obj_prop = unpack(splitter:forward(prediction))
    local gt_pos, gt_vel, gt_ang, gt_ang_vel, gt_obj_prop =
                        unpack(split_output(self.mp):forward(this_future))

    -- NOTE! is there a better loss function for angle?
    self.identitycriterion:forward(p_pos, gt_pos)
    local d_pos = self.identitycriterion:backward(p_pos, gt_pos):clone()

    self.criterion:forward(p_vel, gt_vel)
    local d_vel = self.criterion:backward(p_vel, gt_vel):clone()
    d_vel = d_vel/d_vel:nElement()  -- manually do sizeAverage

    self.identitycriterion:forward(p_ang, gt_ang)
    local d_ang = self.identitycriterion:backward(p_ang, gt_ang):clone()

    self.criterion:forward(p_ang_vel, gt_ang_vel)
    local d_ang_vel = self.criterion:backward(p_ang_vel, gt_ang_vel):clone()
    d_ang_vel = d_ang_vel/d_ang_vel:nElement()  -- manually do sizeAverage

    self.identitycriterion:forward(p_obj_prop, gt_obj_prop)
    local d_obj_prop = self.identitycriterion:backward(p_obj_prop, gt_obj_prop):clone()

    local d_pred = splitter:backward({prediction}, {d_pos, d_vel, d_ang, d_ang_vel, d_obj_prop})
    -- self.network:backward(input,d_pred)  -- updates grad_params

    --self.network.modules[1]: ParallelTable
    --self.network.modules[1].modules[1]: pairwise
    --self.network.modules[1].modules[2]: identity

    ------------------------------------------------------------------
    -- here do the local neighborhood thing: 
    -- replace self.network:backward with decoder:backward, then zero out the
    -- gradients for the far away ones and then continue to backprop. 
    -- note that first you should test that doing backward in two steps works
    -- before you think about doing the zeroing out

    local decoder_in = self.network.modules[1].output  -- table {pairwise_out, this_past}
    local d_decoder = self.network.modules[2]:backward(decoder_in, d_pred)
    local caddtable_in = self.network.modules[1].modules[1].modules[1].output
    local d_caddtable = self.network.modules[1].modules[1].modules[2]:backward(caddtable_in, d_decoder[1])
    d_caddtable = self:apply_neighbor_mask(d_caddtable, self.neighbor_masks)
    local d_pairwise = self.network.modules[1].modules[1].modules[1]:backward(input[1], d_caddtable)
    local d_identity = self.network.modules[1].modules[2]:backward(input[2], d_decoder[2])
    -- d_input = {d_pairwise, d_identity}

    ------------------------------------------------------------------


    collectgarbage()
    return self.theta.grad_params
end

function model:update_position(this, pred)
    -- this: (mp.batch_size, mp.num_past, mp.object_dim)
    -- prediction: (mp.batch_size, mp.num_future, mp.object_dim)
    -- pred is with respect to this[{{},{-1}}]
    ----------------------------------------------------------------------------
    local px = config_args.si.px
    local py = config_args.si.py
    local vx = config_args.si.vx
    local vy = config_args.si.vy
    local pnc = config_args.position_normalize_constant
    local vnc = config_args.velocity_normalize_constant

    local this, pred = this:clone(), pred:clone()
    local lastpos = (this[{{},{-1},{px,py}}]:clone()*pnc)
    local lastvel = (this[{{},{-1},{vx,vy}}]:clone()*vnc)  -- TODO: this is without subsampling make sure that this is correct!
    local currpos = (pred[{{},{},{px,py}}]:clone()*pnc)
    local currvel = (pred[{{},{},{vx,vy}}]:clone()*vnc)

    -- this is length n+1
    local pos = torch.cat({lastpos, currpos},2)
    local vel = torch.cat({lastvel, currvel},2)

    -- iteratively update pos through num_future 
    for i = 1,pos:size(2)-1 do
        pos[{{},{i+1},{}}] = pos[{{},{i},{}}] + vel[{{},{i},{}}]  -- last dim=2
    end

    -- normalize again
    pos = pos/pnc
    assert(pos[{{},{1},{}}]:size(1) == pred:size(1))

    pred[{{},{},{px,py}}] = pos[{{},{2,-1},{}}]  -- reassign back to pred
    return pred
end


function model:update_angle(this, pred)
    local a = config_args.si.a
    local av = config_args.si.av
    local anc = config_args.angle_normalize_constant

    local this, pred = this:clone(), pred:clone()

    -- TODO: use config!
    local last_angle = this[{{},{-1},{a}}]:clone()*anc
    local last_angular_velocity = this[{{},{-1},{av}}]:clone()*anc
    local curr_angle = pred[{{},{},{a}}]:clone()*anc
    local curr_angular_velocity = pred[{{},{},{av}}]:clone()*anc

    -- this is length n+1
    local ang = torch.cat({last_angle, curr_angle},2)
    local ang_vel = torch.cat({last_angular_velocity, curr_angular_velocity},2)

    -- iteratively update ang through time. 
    for i = 1,ang:size(2)-1 do
        ang[{{},{i+1},{}}] = ang[{{},{i},{}}] + ang_vel[{{},{i},{}}]  -- last dim=2
    end

    -- normalize again
    ang = ang/config_args.angle_normalize_constant
    assert(ang[{{},{1},{}}]:size(1) == pred:size(1))

    pred[{{},{},{a}}] = ang[{{},{2,-1},{}}]  -- reassign back to pred
    return pred
end

-- return a table of euc dist between this and each of context
-- size is the number of items in context
-- is this for the last timestep of this?
-- TODO: later we can plot for all timesteps
function model:get_euc_dist(this, context, t)
    local num_context = context:size(2)
    local t = t or -1  -- default use last timestep
    local px = config_args.si.px
    local py = config_args.si.py

    local this_pos = this[{{},{t},{px, py}}]  -- TODO: use config args
    local context_pos = context[{{},{},{t},{px, py}}]
    local euc_dists = self:euc_dist(this_pos:repeatTensor(1,num_context,1), context_pos)
    euc_dists = torch.split(euc_dists, 1,2)  --convert to table of (bsize, 1, 1)
    for i=1,#euc_dists do
        euc_dists[i] = torch.squeeze(euc_dists[i])
    end
    return euc_dists
end

-- b and a must be same size
function model:euc_dist(a,b)
    local diff = torch.squeeze(b - a, 3) -- (bsize, num_context, 2)
    local diffsq = torch.pow(diff,2)
    local euc_dists = torch.sqrt(diffsq[{{},{},{1}}]+diffsq[{{},{},{2}}])  -- (bsize, num_context, 1)
    return euc_dists
end

-- update position at time t to get position at t+1
-- default t is the last t
function model:update_position_one(state, t)
    local t = t or -1
    local px = config_args.si.px
    local py = config_args.si.py
    local vx = config_args.si.vx
    local vy = config_args.si.vy
    local pnc = config_args.position_normalize_constant
    local vnc = config_args.velocity_normalize_constant

    local pos_now, vel_now
    if state:dim() == 4 then
        pos_now = state[{{},{},{t},{px, py}}]
        vel_now = state[{{},{},{t},{vx, vy}}]
    else
        pos_now = state[{{},{t},{px, py}}]
        vel_now = state[{{},{t},{vx, vy}}]
    end

    local pos_next = (pos_now:clone()*pnc + vel_now:clone()*vnc)/pnc
    return pos_next, pos_now
end

-- similar to update_position
function model:get_velocity_direction(this, context, t)
    local num_context = context:size(2)

    local this_pos_next, this_pos_now = self:update_position_one(this)
    local context_pos_next, context_pos_now = self:update_position_one(context)

    -- find difference in distances from this_pos_now to context_pos_now
    -- and from his_pos_now to context_pos_next. This will be +/- number
    local euc_dist_now = self:euc_dist(this_pos_now:repeatTensor(1,num_context,1), context_pos_now)
    local euc_dist_next = self:euc_dist(this_pos_now:repeatTensor(1,num_context,1), context_pos_next)
    local euc_dist_diff = euc_dist_next - euc_dist_now  -- (bsize, num_context, 1)  negative if context moving toward this
    euc_dist_diffs = torch.split(euc_dist_diff, 1,2)  --convert to table of (bsize, 1, 1)
    for i=1,#euc_dist_diffs do
        euc_dist_diffs[i] = torch.squeeze(euc_dist_diffs[i])
    end
    -- assert(false)
    return euc_dist_diffs
end

-- simulate batch forward one timestep
function model:sim(batch)
    
    -- get data
    local this_orig, context_orig, y_orig, context_future_orig, mask = unpack(batch)  -- NOTE CHANGE BATCH HERE

    -- crop to number of timestesp
    y_orig = y_orig[{{},{1, numsteps}}]
    context_future_orig = context_future_orig[{{},{},{1, numsteps}}]

    local num_particles = torch.find(mask,1)[1] + 1

    -- arbitrary notion of ordering here
    -- past: (bsize, num_particles, mp.numpast*mp.objdim)
    -- future: (bsize, num_particles, (mp.winsize-mp.numpast), mp.objdim)
    local past = torch.cat({unsqueeze(this_orig:clone(),2), context_orig},2)
    local future = torch.cat({unsqueeze(y_orig:clone(),2), context_future_orig},2)

    assert(past:size(2) == num_particles and future:size(2) == num_particles)

    local pred_sim = model_utils.transfer_data(
                        torch.zeros(mp.batch_size, num_particles,
                                    numsteps, mp.object_dim),
                        mp.cuda)

    -- loop through time
    for t = 1, numsteps do

        -- for each particle, update to the next timestep, given
        -- the past configuration of everybody
        -- total_particles = total_particles+num_particles

        for j = 1, num_particles do
            -- construct batch
            local this = torch.squeeze(past[{{},{j}}])

            local context
            if j == 1 then
                context = past[{{},{j+1,-1}}]
            elseif j == num_particles then
                context = past[{{},{1,-2}}]
            else
                context = torch.cat({past[{{},{1,j-1}}],
                                                    past[{{},{j+1,-1}}]},2)
            end

            local y = future[{{},{j},{t}}]
            y:resize(mp.batch_size, mp.num_future, mp.object_dim)

            local batch = {this, context, y, _, mask} -- TODO: this may be the problem!

            -- predict
            local loss, pred = model:fp(params_,batch,true)   -- NOTE CHANGE THIS!
            avg_loss = avg_loss + loss
            count = count + 1

            pred = pred:reshape(mp.batch_size, mp.num_future, mp.object_dim)
            this = this:reshape(mp.batch_size, mp.num_past, mp.object_dim)  -- unnecessary

            -- -- relative coords for next timestep
            if mp.relative then
                pred = data_process.relative_pair(this, pred, true)
            end

            -- restore object properties because we aren't learning them
            pred[{{},{},{config_args.ossi,-1}}] = this[{{},{-1},{config_args.ossi,-1}}]  -- NOTE! THIS DOESN'T TAKE ANGLE INTO ACCOUNT!
            
            -- update position
            pred = update_position(this, pred)

            -- update angle
            pred = update_angle(this, pred)
            -- pred = unsqueezer:forward(pred)
            pred = unsqueeze(pred, 2)

            -- write into pred_sim
            pred_sim[{{},{j},{t},{}}] = pred
        end

        -- update past for next timestep
        if mp.num_past > 1 then
            past = torch.cat({past[{{},{},{2,-1},{}}],
                                pred_sim[{{},{},{t},{}}]}, 3)
        else
            assert(mp.num_past == 1)
            past = pred_sim[{{},{},{t},{}}]:clone()
        end

        -- local this_orig, context_orig, y_orig, context_future_orig, this_pred, context_future_pred, loss = model:sim(batch)

        
    end
    --- to be honest I don't think we need to break into past and context
    -- future, but actually that might be good for coloriing past and future, but
    -- actually I don't think so. For now let's just adapt it

    -- at this point, pred_sim should be all filled out
    -- break pred_sim into this and context_future
    -- recall: pred_sim: (batch_size,seq_length+1,numsteps,object_dim)
    -- recall that you had defined this_pred as the first obj in the future tensor
    local this_pred = torch.squeeze(pred_sim[{{},{1}}])
    if numsteps == 1 then this_pred = unsqueeze(this_pred,2) end

    local context_pred = pred_sim[{{},{2,-1}}]

    if mp.relative then
        y_orig = data_process.relative_pair(this_orig, y_orig, true)
    end

    -- local this_orig, context_orig, y_orig, context_future_orig, this_pred, context_future_pred, loss = model:sim(batch)

    -- if saveoutput and i <= mp.ns then
    --     save_ex_pred_json({this_orig, context_orig,
    --                         y_orig, context_future_orig,
    --                         this_pred, context_pred},
    --                         'batch'..test_loader.datasamplers[current_dataset].current_sampled_id..'.json',
    --                         current_dataset)
    -- end
end

return model
