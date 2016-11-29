require 'nn'
require 'rnn'
require 'torch'
require 'nngraph'
require 'IdentityCriterion'
require 'data_utils'
require 'modules'

nngraph.setDebug(true)

-- input: table of length num_obj with size (bsize, num_past*obj_dim)
-- output: table of length num_obj with size (bsize, num_future*obj_dim)
function init_network(params)
    local hid_dim = params.rnn_dim
    local obj_dim = params.object_dim
    local max_obj = params.seq_length
    local num_past = params.num_past
    local num_future = params.num_future
    local in_dim = num_past*obj_dim
    local out_dim = num_future*obj_dim
    local num_layers = params.layers

    assert(num_layers > 0)

    local net = nn.Sequential()
    local anLSTM
    for i = 1, num_layers do
        nn.FastLSTM.usenngraph = true
        nn.FastLSTM.bn = true
        if i == 1 then anLSTM = nn.FastLSTM(in_dim, hid_dim)
        else
            anLSTM = nn.FastLSTM(hid_dim, hid_dim)
            if mp.dropout > 0 then 
                net:add(nn.Sequencer(nn.Dropout(mp.dropout))) 
            end
        end
        net:add(nn.Sequencer(anLSTM))
        net:add(nn.Sequencer(nn.ReLU()))
    end
    net:add(nn.Sequencer(nn.Linear(hid_dim, out_dim)))

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
            self.network:cuda()
            self.criterion:cuda()
            self.identitycriterion:cuda()
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

function model:cuda()
    self.network = self.network:cuda()
    self.criterion:cuda()
    self.identitycriterion:cuda()
end

function model:float()
    self.network = self.network:float()
    self.criterion:float()
    self.identitycriterion:float()
end

function model:clearState()
    self.network:clearState()
end

function model:unpack_batch(batch, sim)
    local this, context, y, context_future, mask = unpack(batch)
    local x = {this=this,context=context}

    -- unpack inputs
    local this_past     = convert_type(x.this:clone(), mp.cuda)
    local context       = convert_type(x.context:clone(), mp.cuda)
    local this_future   = convert_type(y:clone(), mp.cuda)

    local bsize, num_past, obj_dim = this_past:size(1), this_past:size(2), this_past:size(3)
    local num_context = context:size(2)

    if mp.of then
        this_past = torch.cat({this_past:clone(), convert_type(torch.ones(mp.batch_size, mp.num_past, 1), mp.cuda)},3)  -- good
        this_future = torch.cat({this_future:clone(), convert_type(torch.ones(mp.batch_size, mp.num_future, 1), mp.cuda)},3)  -- good
        context = torch.cat({context:clone(), convert_type(torch.zeros(mp.batch_size, num_context, mp.num_past, 1), mp.cuda)},4)  -- good
    end

    -- reshape
    this_past = this_past:reshape(this_past:size(1), this_past:size(2)*this_past:size(3))
    context = context:reshape(context:size(1), context:size(2), context:size(3)*context:size(4))
    this_future = this_future:reshape(this_future:size(1),this_future:size(2)*this_future:size(3))

    assert(this_past:size(1) == mp.batch_size and
            this_past:size(2) == mp.input_dim,
            'Your batch size or input dim is wrong')
    assert(context:size(1) == mp.batch_size and
            context:size(2)==torch.find(mask,1)[1]
            and context:size(3) == mp.input_dim)
    assert(this_future:size(1) == mp.batch_size and
            this_future:size(2) == mp.out_dim)

    local shuffind
    if sim then
        shuffind = torch.range(1,num_context)
    else
        shuffind = torch.randperm(num_context)
    end

    -- here you have to create a table of tables
    -- this: (bsize, input_dim)
    -- context: (bsize, mp.seq_length, dim)
    local contexts = {}
    for t=1,torch.find(mask,1)[1] do  -- not actually mp.seq_length!
        local one_context = torch.squeeze(context[{{},{shuffind[t]}}])
        table.insert(contexts, one_context)  -- good
    end

    local focus_past = this_past:clone()
    local focus_future = this_future:clone()

    ------------------------------------------------------------------
    -- here do the local neighborhood thing
    if self.mp.nbrhd then  
        -- this gets updated every batch!
        self.neighbor_masks = self:select_neighbors(contexts, focus_past:clone())  
    else
        self.neighbor_masks = {}  -- don't mask out neighbors
        for i=1,#input do
            table.insert(self.neighbor_masks, convert_type(torch.ones(mp.batch_size), self.mp.cuda))  -- good
        end
    end

    contexts = self:apply_mask(contexts, self.neighbor_masks)

    local all_past = contexts
    table.insert(all_past, focus_past:clone())  -- last element always this_past

    local all_future = {}
    for i=1,num_context do
        table.insert(all_future, convert_type(torch.zeros(mp.batch_size, mp.num_future*mp.object_dim),self.mp.cuda))
    end
    table.insert(all_future, focus_future)

    -- the last element is always the focus object
    -- context are shuffled
    return all_past, all_future  -- at this point, we have added a dimension to all_past AND all_future
end

-- in: model input: table of length num_context-1 of {(bsize, num_past*obj_dim),(bsize, num_past*obj_dim)}
-- out: {{indices of neighbors}, {indices of non-neighbors}}
-- maybe I can output a mask? then I can rename this function to neighborhood_mask
function model:select_neighbors(contexts, this)
    local threshold
    local neighbor_masks = {}

    this = this:clone():reshape(mp.batch_size, mp.num_past, mp.object_dim)
    for i, c in pairs(contexts) do
        -- reshape
        local context = c:clone():reshape(mp.batch_size, mp.num_past, mp.object_dim)

        local oid_onehot, template_ball, template_block = get_oid_templates(this, config_args, self.mp.cuda)

        if (oid_onehot-template_ball):norm()==0 then
            threshold = self.mp.nbrhdsize*config_args.object_base_size.ball  -- this is not normalized!
        elseif oid_onehot:equal(template_block) then
            threshold = self.mp.nbrhdsize*config_args.object_base_size.block
        else
            print(oid_onehot)
            print(template_ball)
            print((oid_onehot-template_ball):norm())
            assert(false, 'Unknown object id')
        end

        -- compute where they will be in the next timestep
        local this_pos_next, this_pos_now = self:update_position_one(this)
        local context_pos_next, context_pos_now = self:update_position_one(context)

        this_pos_next = this_pos_now:clone()
        context_pos_next = context_pos_now:clone()

        -- compute euclidean distance between this_pos_next and context_pos_next
        local euc_dist_next = torch.squeeze(self:euc_dist(this_pos_next, context_pos_next)) -- (bsize)
        euc_dist_next = euc_dist_next * config_args.position_normalize_constant  -- turn into absolute coordinates

        -- find the indices in the batch for neighbors and non-neighbors
        local neighbor_mask = convert_type(euc_dist_next:le(threshold), mp.cuda)  -- 1 if neighbor 0 otherwise
        table.insert(neighbor_masks, neighbor_mask:clone()) -- good
    end

    return neighbor_masks
end

-- we mask out this as well, because it is as if that interaction didn't happen
function model:apply_mask(input, batch_mask)
    assert(#batch_mask == #input)
    for i, x in pairs(input) do 
        if type(x) == 'table' then
            -- mutates within place
            x[1] = torch.cmul(x[1],batch_mask[i]:view(mp.batch_size,1):expandAs(x[1]))
            x[2] = torch.cmul(x[2], batch_mask[i]:view(mp.batch_size,1):expandAs(x[2]))
        else
            x = torch.cmul(x, convert_type(batch_mask[i]:view(mp.batch_size,1):expandAs(x), mp.cuda))
            input[i] = x -- it doesn't actually automatically mutate
        end
    end
    return input
end


function model:fp(params_, batch, sim)
    if params_ ~= self.theta.params then self.theta.params:copy(params_) end
    self.theta.grad_params:zero()  -- reset gradient

    local all_past, all_future = self:unpack_batch(batch, sim)
    local prediction = self.network:forward(all_past)

    local num_objects = #prediction

    local p_pos, p_vel, p_ang, p_ang_vel, p_obj_prop =
                        unpack(split_output(self.mp):forward(prediction[num_objects]))
    local gt_pos, gt_vel, gt_ang, gt_ang_vel, gt_obj_prop =
                        unpack(split_output(self.mp):forward(all_future[num_objects]))

    local loss_vel = self.criterion:forward(p_vel, gt_vel)
    local loss_ang_vel = self.criterion:forward(p_ang_vel, gt_ang_vel)
    local loss = loss_vel + loss_ang_vel

    loss = loss/(p_vel:nElement()+p_ang_vel:nElement()) -- manually do size average

    if mp.cuda then cutorch.synchronize() end
    collectgarbage()
    return loss, prediction[num_objects], loss_vel/p_vel:nElement(), loss_ang_vel/p_ang_vel:nElement()
end


function model:fp_batch(params_, batch, sim)
    if params_ ~= self.theta.params then self.theta.params:copy(params_) end
    self.theta.grad_params:zero()  -- reset gradient

    local all_past, all_future = self:unpack_batch(batch, sim)

    local prediction = self.network:forward(all_past)

    local num_objects = #prediction

    local p_pos, p_vel, p_ang, p_ang_vel, p_obj_prop =
                        unpack(split_output(self.mp):forward(prediction[num_objects]))
    local gt_pos, gt_vel, gt_ang, gt_ang_vel, gt_obj_prop =
                        unpack(split_output(self.mp):forward(all_future[num_objects]))
    -- p_vel: (bsize, 1, p_veldim)
    -- p_ang_vel: (bsize, 1, p_ang_veldim)

    local loss_all = {}
    local loss_vel_all = {}
    local loss_ang_vel_all = {}
    for i=1,mp.batch_size do
        local loss_vel = self.criterion:forward(p_vel[{{i}}], gt_vel[{{i}}])
        local loss_ang_vel = self.criterion:forward(p_ang_vel[{{i}}], gt_ang_vel[{{i}}])
        local loss = loss_vel + loss_ang_vel
        loss = loss/(p_vel[{{i}}]:nElement()+p_ang_vel[{{i}}]:nElement()) -- manually do size average
        loss_vel = loss_vel/p_vel[{{i}}]:nElement()
        loss_ang_vel = loss_ang_vel/p_ang_vel[{{i}}]:nElement()
        table.insert(loss_all, loss)
        table.insert(loss_vel_all, loss_vel)
        table.insert(loss_ang_vel_all, loss_ang_vel)

    end

    collectgarbage()
    return torch.Tensor(loss_all), prediction[num_objects], torch.Tensor(loss_vel_all), torch.Tensor(loss_ang_vel_all)
end


-- local p_pos, p_vel, p_obj_prop=split_output(params):forward(prediction)
-- local gt_pos, gt_vel, gt_obj_prop=split_output(params):forward(this_future)
-- a lot of instantiations of split_output
function model:bp(batch, prediction, sim)
    self.theta.grad_params:zero() -- the d_parameters
    local all_past, all_future = self:unpack_batch(batch, sim)

    local d_pred = {}

    -- no gradient for every step except the last one
    for i = 1, #all_future-1 do
        table.insert(d_pred, convert_type(torch.zeros(mp.batch_size, mp.num_future*mp.object_dim), mp.cuda))
    end

    -- pass gradients through to the last step, the focus object
    local splitter = split_output(self.mp)

    -- local p_pos, p_vel, p_ang, p_ang_vel, p_obj_prop = unpack(splitter:forward(prediction[#prediction]))
    local p_pos, p_vel, p_ang, p_ang_vel, p_obj_prop = unpack(splitter:forward(prediction))
    local gt_pos, gt_vel, gt_ang, gt_ang_vel, gt_obj_prop =
                        unpack(split_output(self.mp):forward(all_future[#all_future]))

    -- TODO: get better cost function for angle
    self.identitycriterion:forward(p_pos, gt_pos)
    local d_pos = self.identitycriterion:backward(p_pos, gt_pos):clone()

    self.criterion:forward(p_vel, gt_vel)
    local d_vel = self.criterion:backward(p_vel, gt_vel):clone()

    d_vel:mul(mp.vlambda)
    d_vel = d_vel/d_vel:nElement()  -- manually do sizeAverage

    self.identitycriterion:forward(p_ang, gt_ang)
    local d_ang = self.identitycriterion:backward(p_ang, gt_ang):clone()

    self.criterion:forward(p_ang_vel, gt_ang_vel)
    local d_ang_vel = self.criterion:backward(p_ang_vel, gt_ang_vel):clone()
    d_ang_vel:mul(mp.lambda)
    d_ang_vel = d_ang_vel/d_ang_vel:nElement()  -- manually do sizeAverage

    self.identitycriterion:forward(p_obj_prop, gt_obj_prop)
    local d_obj_prop = self.identitycriterion:backward(p_obj_prop, gt_obj_prop):clone()

    local obj_d_pred = splitter:backward({prediction[i]}, {d_pos, d_vel, d_ang, d_ang_vel, d_obj_prop}):clone()
    table.insert(d_pred, obj_d_pred)

    self.network:backward(all_past,d_pred)  -- updates grad_params

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
    local lastvel = (this[{{},{-1},{vx,vy}}]:clone()*vnc)
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
    ang = ang/anc
    assert(ang[{{},{1},{}}]:size(1) == pred:size(1))

    pred[{{},{},{a}}] = ang[{{},{2,-1},{}}]  -- reassign back to pred
    return pred
end

-- return a table of euc dist between this and each of context
-- size is the number of items in context
-- is this for the last timestep of this?
-- TODO_lowpriority: later we can plot for all timesteps
function model:get_euc_dist(this, context, t)
    local num_context = context:size(2)
    local t = t or -1  -- default use last timestep
    local px = config_args.si.px
    local py = config_args.si.py

    local this_pos = this[{{},{t},{px, py}}]
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
    return compute_euc_dist(a,b)
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
    return euc_dist_diffs
end

return model
