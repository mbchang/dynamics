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
    else
        error('unknown model')
    end

    local encoder = init_object_encoder(params.input_dim, params.rnn_dim)
    local decoder = init_object_decoder(dcoef*params.rnn_dim, params.num_future,
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

    -- I think if I add: sequencer_type(sequencer), then it'd be able to go through time as well.
    --
    local net = nn.Sequential()
    net:add(sequencer)

    -- input table of (bsize, 2*d_hid) of length seq_length
    -- output: tensor (bsize, 2*d_hid)
    net:add(nn.CAddTable())  -- add across the "timesteps" to sum contributions
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

    self.theta = {}
    self.theta.params, self.theta.grad_params = self.network:getParameters()

    collectgarbage()
    return self
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

    local input, this_future = unpack_batch(batch, sim)

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
    local input, this_future = unpack_batch(batch, sim)

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
    self.network:backward(input,d_pred)  -- updates grad_params

    collectgarbage()
    return self.theta.grad_params
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
