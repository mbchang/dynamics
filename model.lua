require 'metaparams'
require 'nn'
require 'torch'
require 'nngraph'
require 'Base'
local model_utils = require 'model_utils'

if common_mp.cuda then
    require 'cutorch'
    require 'cunn'
end

if common_mp.cudnn then
    require 'cudnn'
end

nngraph.setDebug(true)

function init_baseline_encoder(mp, output_dim)
    --[[ Maps a 150 x 150 image to a vector of size output_dim 

    --]]
    if mp.cudnn then 
        local encoder_in      = nn.Identity()()

        -- Conv Layer 1
        local enc_l1conv      = cudnn.SpatialConvolution(mp['color_channels'],mp['feature_maps'],mp['decoder_filter_size'],mp['decoder_filter_size'])(encoder_in)
        local enc_l1pool      = cudnn.SpatialMaxPooling(mp['pool_size'],mp['pool_size'],mp['pool_size'],mp['pool_size'])(enc_l1conv)
        local enc_l1relu      = cudnn.ReLU(true)(enc_l1pool)

        -- Conv Layer 2
        local enc_l2conv      = cudnn.SpatialConvolution(mp['feature_maps'],mp['feature_maps']/2,mp['encoder_filter_size'],mp['encoder_filter_size'])(enc_l1relu)
        local enc_l2pool      = cudnn.SpatialMaxPooling(mp['pool_size'],mp['pool_size'],mp['pool_size'],mp['pool_size'])(enc_l2conv)
        local enc_l2relu      = cudnn.ReLU(true)(enc_l2pool)

        -- Conv Layer 3
        local enc_l3conv      = cudnn.SpatialConvolution(mp['feature_maps']/2,mp['feature_maps']/4,mp['encoder_filter_size'],mp['encoder_filter_size'])(enc_l2relu)
        local enc_l3pool      = cudnn.SpatialMaxPooling(mp['pool_size'],mp['pool_size'],mp['pool_size'],mp['pool_size'])(enc_l3conv)
        local enc_l3relu      = cudnn.ReLU(true)(enc_l3pool)

        -- FC Layer 4
        local enc_l4reshape   = nn.Reshape((mp['feature_maps']/4)*mp['cnn_last_dim']*mp['cnn_last_dim'])(enc_l3relu)
        local enc_l4linear    = nn.Linear((mp['feature_maps']/4)*mp['cnn_last_dim']*mp['cnn_last_dim'], output_dim)(enc_l4reshape)
        local enc_l4relu      = cudnn.ReLU(true)(enc_l4linear)

        local encoder_out     = enc_l4relu

        collectgarbage()
        return nn.gModule({encoder_in}, {encoder_out})  -- input and output must be in table!
    else 
        local encoder_in      = nn.Identity()()

        -- Conv Layer 1
        local enc_l1conv      = nn.SpatialConvolution(mp['color_channels'],mp['feature_maps'],mp['decoder_filter_size'],mp['decoder_filter_size'])(encoder_in)
        local enc_l1pool      = nn.SpatialMaxPooling(mp['pool_size'],mp['pool_size'],mp['pool_size'],mp['pool_size'])(enc_l1conv)
        local enc_l1relu      = nn.ReLU(true)(enc_l1pool)

        -- Conv Layer 2
        local enc_l2conv      = nn.SpatialConvolution(mp['feature_maps'],mp['feature_maps']/2,mp['encoder_filter_size'],mp['encoder_filter_size'])(enc_l1relu)
        local enc_l2pool      = nn.SpatialMaxPooling(mp['pool_size'],mp['pool_size'],mp['pool_size'],mp['pool_size'])(enc_l2conv)
        local enc_l2relu      = nn.ReLU(true)(enc_l2pool)

        -- Conv Layer 3
        local enc_l3conv      = nn.SpatialConvolution(mp['feature_maps']/2,mp['feature_maps']/4,mp['encoder_filter_size'],mp['encoder_filter_size'])(enc_l2relu)
        local enc_l3pool      = nn.SpatialMaxPooling(mp['pool_size'],mp['pool_size'],mp['pool_size'],mp['pool_size'])(enc_l3conv)
        local enc_l3relu      = nn.ReLU(true)(enc_l3pool)

        -- FC Layer 4
        local enc_l4reshape   = nn.Reshape((mp['feature_maps']/4)*mp['cnn_last_dim']*mp['cnn_last_dim'])(enc_l3relu)
        local enc_l4linear    = nn.Linear((mp['feature_maps']/4)*mp['cnn_last_dim']*mp['cnn_last_dim'], output_dim)(enc_l4reshape)
        local enc_l4relu      = nn.ReLU(true)(enc_l4linear)

        local encoder_out     = enc_l4relu

        collectgarbage()
        return nn.gModule({encoder_in}, {encoder_out})  -- input and output must be in table!
    end
end 


function init_baseline_decoder(mp, input_dim)
    --[[ Maps a vector of size 50 to an image of size 150 x 150
        input_dim is input dimension to decoder
    --]]
    if mp.cudnn then 
        local decoder_in      = nn.Identity()()

        -- FC Layer 1
        local dec_l1linear    = nn.Linear(input_dim, (mp['feature_maps']/4)*mp['cnn_last_dim']*mp['cnn_last_dim'])(decoder_in)
        local dec_l1relu      = cudnn.ReLU(true)(dec_l1linear)  -- necessary?
        local dec_l1reshape   = nn.Reshape((mp['feature_maps']/4),mp['cnn_last_dim'],mp['cnn_last_dim'])(dec_l1relu)

        -- Deconv Layer 2    
        local dec_l2upsamp    = nn.SpatialUpSamplingNearest(2)(dec_l1reshape)
        local dec_l2conv      = cudnn.SpatialConvolution(mp['feature_maps']/4,mp['feature_maps']/2, mp['decoder_filter_size'], mp['decoder_filter_size'])(dec_l2upsamp)
        local dec_l2relu      = cudnn.ReLU(true)(dec_l2conv)

        -- Deconv Layer 3
        local dec_l3upsamp    = nn.SpatialUpSamplingNearest(2)(dec_l2relu)
        local dec_l3conv      = cudnn.SpatialConvolution(mp['feature_maps']/2,mp['feature_maps']/2,mp['decoder_filter_size'],mp['decoder_filter_size'])(dec_l3upsamp)
        local dec_l3relu      = cudnn.ReLU(true)(dec_l3conv)

        -- Deconv Layer 4
        local dec_l4upsamp    = nn.SpatialUpSamplingNearest(2)(dec_l3relu)
        local dec_l4conv      = cudnn.SpatialConvolution(mp['feature_maps']/2,mp['feature_maps'],mp['decoder_filter_size'],mp['decoder_filter_size'])(dec_l4upsamp)
        local dec_l4relu      = cudnn.ReLU(true)(dec_l4conv)

        -- Deconv Layer 5
        local dec_l5upsamp    = nn.SpatialUpSamplingNearest(2)(dec_l4relu)
        local dec_l5conv      = cudnn.SpatialConvolution(mp['feature_maps'],mp['color_channels'],mp['decoder_filter_size'],mp['decoder_filter_size'])(dec_l5upsamp)  -- NOTE I am not doing anything to featuremaps!
        local dec_l5sig       = cudnn.Sigmoid()(dec_l5conv)

        local decoder_out     = dec_l5sig

        collectgarbage()
        return nn.gModule({decoder_in}, {decoder_out})
    else 
        local decoder_in      = nn.Identity()()

        -- FC Layer 1
        local dec_l1linear    = nn.Linear(input_dim, (mp['feature_maps']/4)*mp['cnn_last_dim']*mp['cnn_last_dim'])(decoder_in)
        local dec_l1relu      = nn.ReLU(true)(dec_l1linear)  -- necessary?
        local dec_l1reshape   = nn.Reshape((mp['feature_maps']/4),mp['cnn_last_dim'],mp['cnn_last_dim'])(dec_l1relu)

        -- Deconv Layer 2    
        local dec_l2upsamp    = nn.SpatialUpSamplingNearest(2)(dec_l1reshape)
        local dec_l2conv      = nn.SpatialConvolution(mp['feature_maps']/4,mp['feature_maps']/2, mp['decoder_filter_size'], mp['decoder_filter_size'])(dec_l2upsamp)
        local dec_l2relu      = nn.ReLU(true)(dec_l2conv)

        -- Deconv Layer 3
        local dec_l3upsamp    = nn.SpatialUpSamplingNearest(2)(dec_l2relu)
        local dec_l3conv      = nn.SpatialConvolution(mp['feature_maps']/2,mp['feature_maps']/2,mp['decoder_filter_size'],mp['decoder_filter_size'])(dec_l3upsamp)
        local dec_l3relu      = nn.ReLU(true)(dec_l3conv)

        -- Deconv Layer 4
        local dec_l4upsamp    = nn.SpatialUpSamplingNearest(2)(dec_l3relu)
        local dec_l4conv      = nn.SpatialConvolution(mp['feature_maps']/2,mp['feature_maps'],mp['decoder_filter_size'],mp['decoder_filter_size'])(dec_l4upsamp)
        local dec_l4relu      = nn.ReLU(true)(dec_l4conv)

        -- Deconv Layer 5
        local dec_l5upsamp    = nn.SpatialUpSamplingNearest(2)(dec_l4relu)
        local dec_l5conv      = nn.SpatialConvolution(mp['feature_maps'],mp['color_channels'],mp['decoder_filter_size'],mp['decoder_filter_size'])(dec_l5upsamp)  -- NOTE I am not doing anything to featuremaps!
        local dec_l5sig       = nn.Sigmoid()(dec_l5conv)

        local decoder_out     = dec_l5sig

        collectgarbage()
        return nn.gModule({decoder_in}, {decoder_out})
    end
end


function lstm(x, prev_c, prev_h, params)
    -- Calculate all four gates in one go
    local i2h = nn.Linear(params.rnn_dim, 4*params.rnn_dim)(x) -- this is the problem when I go two layers
    local h2h = nn.Linear(params.rnn_dim, 4*params.rnn_dim)(prev_h)
    local gates = nn.CAddTable()({i2h, h2h})

    -- Reshape to (bsize, n_gates, hid_size)
    -- Then slize the n_gates dimension, i.e dimension 2
    local reshaped_gates =  nn.Reshape(4,params.rnn_dim)(gates)
    local sliced_gates = nn.SplitTable(2)(reshaped_gates)

    -- Use select gate to fetch each gate and apply nonlinearity
    local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
    local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
    local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
    local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

    local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    return next_c, next_h
end


function init_object_encoder(input_dim, rnn_inp_dim)
    assert(rnn_inp_dim % 2 == 0)
    local thisp     = nn.Identity()() -- this particle of interest  (batch_size, input_dim)
    local contextp  = nn.Identity()() -- the context particle  (batch_size, partilce_dim)

    local thisp_out     = nn.Tanh()(nn.Linear(input_dim, rnn_inp_dim/2)(thisp))  -- (batch_size, rnn_inp_dim/2)
    local contextp_out  = nn.Tanh()(nn.Linear(input_dim, rnn_inp_dim/2)(contextp)) -- (batch_size, rnn_inp_dim/2)

    -- Concatenate
    local encoder_out = nn.JoinTable(2)({thisp_out, contextp_out})  -- (batch_size, rnn_inp_dim)

    return nn.gModule({thisp, contextp}, {encoder_out})
end


function init_object_decoder(rnn_hid_dim, out_dim)
    local rnn_out = nn.Identity()()  -- rnn_out had better be of dim (batch_size, rnn_hid_dim)
    local decoder_out = nn.Tanh()(nn.Linear(rnn_hid_dim, out_dim)(rnn_out))

    return nn.gModule({rnn_out}, {decoder_out}) 
end


-- do not need the mask
-- params: layers, input_dim, goo_dim, rnn_inp_dim, rnn_hid_dim, out_dim
function init_network(params)
    -- Initialize encoder and decoder
    local encoder = init_object_encoder(params.input_dim, params.rnn_dim)
    local decoder = init_object_decoder(params.rnn_dim, params.out_dim)

    -- Input to netowrk
    local thisp_past    = nn.Identity()() -- this particle of interest, past
    local contextp      = nn.Identity()() -- the context particle
    local thisp_future  = nn.Identity()() -- the particle of interet, future

    -- Input to LSTM
    local lstm_input = encoder({thisp_past, contextp})
    local prev_s = nn.Identity()() -- LSTM

    -- Go through each layer of LSTM
    local rnn_inp = {[0] = nn.Identity()(lstm_input)}  -- rnn_inp[i] holds the input at layer i+1
    local next_s = {}
    local split = {prev_s:split(2 * params.layers)}
    local next_c, next_h
    for layer_idx = 1, params.layers do
        local prev_c         = split[2 * layer_idx - 1]  -- odd entries
        local prev_h         = split[2 * layer_idx]  -- even entries
        local dropped        = rnn_inp[layer_idx - 1]
        next_c, next_h = lstm(dropped, prev_c, prev_h, params)  -- you can make this a gModule if you wnant
        table.insert(next_s, next_c)
        table.insert(next_s, next_h)
        rnn_inp[layer_idx] = next_h
    end

    local prediction = decoder({next_h})  -- next_h is the output of the last layer
    local err = nn.MSECriterion()({prediction, thisp_future})  -- should be MSECriterion()({prediction, thisp_future})
    return nn.gModule({thisp_past, contextp, prev_s, thisp_future}, {err, nn.Identity()(next_s), prediction})  -- last output should be prediction
end


function init_baseline_lstm(mp, input_size, rnn_size)
    if mp.cudnn then 
        --------------------- input structure ---------------------
        local input = nn.Identity()()  -- lstm input
        local prev_c = nn.Identity()()  -- c at time t-1
        local prev_h = nn.Identity()()  -- h at time t-1

        --------------------- preactivations ----------------------
        function new_input_sum()
            -- transforms input
            local i2h            = nn.Linear(input_size, rnn_size)(input)  -- multiply input by weight vector (How do you set the initial weights?)
            -- transforms previous timestep's output
            local h2h            = nn.Linear(rnn_size, rnn_size)(prev_h)  -- 
            local preactivations = nn.CAddTable()({i2h, h2h})  -- element wise addition, with a variable number of arguments
            return preactivations
        end

        --------------------------- gates --------------------------
        local in_gate          = cudnn.Sigmoid()(new_input_sum())
        local forget_gate      = cudnn.Sigmoid()(new_input_sum())
        local out_gate         = cudnn.Sigmoid()(new_input_sum())
        local in_transform     = cudnn.Tanh()(new_input_sum())

        --------------------- next cell state ---------------------

        local next_c           = nn.CAddTable()({
            nn.CMulTable()({forget_gate, prev_c}),  
            nn.CMulTable()({in_gate,     in_transform})
        })

        -------------------- next hidden state --------------------
        local next_h           = nn.CMulTable()({out_gate, cudnn.Tanh()(next_c)}) 

        -- packs the graph into a convenient module with standard API (:forward(), :backward())
        collectgarbage()
        return nn.gModule({input, prev_c, prev_h}, {next_c, next_h})
    else 
        --------------------- input structure ---------------------
        local input = nn.Identity()()  -- lstm input
        local prev_c = nn.Identity()()  -- c at time t-1
        local prev_h = nn.Identity()()  -- h at time t-1

        --------------------- preactivations ----------------------
        function new_input_sum()
            -- transforms input
            local i2h            = nn.Linear(input_size, rnn_size)(input)  -- multiply input by weight vector (How do you set the initial weights?)
            -- transforms previous timestep's output
            local h2h            = nn.Linear(rnn_size, rnn_size)(prev_h)  -- 
            local preactivations = nn.CAddTable()({i2h, h2h})  -- element wise addition, with a variable number of arguments
            return preactivations
        end

        --------------------------- gates --------------------------
        local in_gate          = nn.Sigmoid()(new_input_sum())
        local forget_gate      = nn.Sigmoid()(new_input_sum())
        local out_gate         = nn.Sigmoid()(new_input_sum())
        local in_transform     = nn.Tanh()(new_input_sum())

        --------------------- next cell state ---------------------

        local next_c           = nn.CAddTable()({
            nn.CMulTable()({forget_gate, prev_c}),  
            nn.CMulTable()({in_gate,     in_transform})
        })

        -------------------- next hidden state --------------------
        local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)}) 

        -- packs the graph into a convenient module with standard API (:forward(), :backward())
        collectgarbage()
        return nn.gModule({input, prev_c, prev_h}, {next_c, next_h})  
    end
end


function init_baseline_model()
    local encoder       = init_baseline_encoder(mp['LSTM_input_dim'])
    local lstm          = init_baseline_lstm(mp['LSTM_input_dim'], mp['LSTM_hidden_dim']) -- todo: how to specify output dimension?
    local decoder       = init_baseline_decoder(mp['LSTM_hidden_dim'])

    -- Construct containers for the inputs
    local input         = nn.Identity()()  -- lstm input
    local prev_c        = nn.Identity()()  -- c at time t-1
    local prev_h        = nn.Identity()()  -- h at time t-1

    -- Encoder
    local encoder_out   = encoder(input)  -- output of encoder is input to lstm

    -- LSTM
    local lstm_allout   = {lstm({encoder_out, prev_c, prev_h}):split(2)} -- can you use unpack here? maybe not when defining the graphs
    local next_c        = lstm_allout[1]
    local next_h        = lstm_allout[2]

    -- Decoder
    local decoder_out   = decoder(next_h)  -- output of lstm is input to decoder

    -- Module
    local module        = nn.gModule({input, prev_c, prev_h}, {decoder_out, next_c, next_h})
    -- module:cuda()
    collectgarbage()

    return module
end 

function init_baseline_cnn_model()
    local encoder       = init_baseline_encoder(mp['LSTM_input_dim'])
    local decoder       = init_baseline_decoder(mp['LSTM_input_dim'])

    -- Construct containers for the inputs
    local input         = nn.Identity()()  -- lstm input

    -- Encoder
    local encoder_out   = encoder(input)  -- output of encoder is input to lstm

    -- Decoder
    local decoder_out   = decoder(encoder_out)  -- output of lstm is input to decoder

    -- Module
    local module        = nn.gModule({input}, {decoder_out})
    -- module:cuda()
    collectgarbage()

    return module
end


-- Create Model
function test_model()
    local params = {
                    layers          = 2,
                    input_dim        = 8*10,
                    rnn_dim         = 100,
                    out_dim         = 8*10
                    }

    local batch_size = 6
    local seq_length = 3

    -- Network
    local network = init_network(params)
    local p, gp = network:getParameters()
    print(p:size())
    print(gp:size())
    -- assert(false)
    local rnns = g_cloneManyTimes(network, seq_length, not network.parameters)

    -- Data
    local this_past       = torch.uniform(torch.Tensor(batch_size, params.input_dim))
    local context         = torch.uniform(torch.Tensor(batch_size, seq_length, params.input_dim))
    local this_future     = torch.uniform(torch.Tensor(batch_size, params.input_dim))
    local mask            = torch.Tensor({1,1,1})

    for i=1,10000 do

    -- State
    local s = {}
    for j = 0, seq_length do
        s[j] = {}
        for d = 1, 2 * params.layers do
            s[j][d] = torch.zeros(batch_size, params.rnn_dim) 
        end
    end

    local ds = {}
    for d = 1, 2 * params.layers do
        ds[d] = torch.zeros(batch_size, params.rnn_dim)
    end

    -- Forward
    local loss = torch.zeros(seq_length)
    local predictions = {}
    for i = 1, seq_length do
        local sim1 = s[i-1]
        loss[i], s[i], predictions[i] = unpack(rnns[i]:forward({this_past, context[{{},i}], sim1, this_future}))  -- problem! (feeding thisp_future every time; is that okay because I just update the gradient based on desired timesstep?)
    end 
    print('total loss', loss:sum())
    -- print('accum_loss', accum_loss)

    -- Backward
    for i = seq_length, 1, -1 do
        local sim1 = s[i - 1]
        local derr
        if mask:clone()[i] == 1 then 
            derr = torch.ones(1)
        elseif mask:clone()[i] == 0 then
            derr = torch.zeros(1) 
        else
            error('invalid mask')
        end
        local dpred = torch.zeros(batch_size,params.out_dim)
        local dtp, dc, dsim1, dtf = unpack(rnns[i]:backward({this_past, context[{{},i}], sim1, this_future}, {derr, ds, dpred}))
        g_replace_table(ds, dsim1)
    end



    -- config = {
    --     learningRate = 0.0001,
    --     momentumDecay = 0.1,
    --     updateDecay = 0.01
    -- }
    -- p = rmsprop(gp, p, config, state)


    -- network:updateParameters(0.00001)
    -- gp:clamp(-self.mp.max_grad_norm, self.mp.max_grad_norm)
    -- collectgarbage()
    -- return loss, self.theta.grad_params


end
end


function test_encoder()
    local params = {
                    layers          = 2,
                    input_dim    = 7,
                    rnn_inp_dim     = 50,
                    rnn_hid_dim     = 50,  -- problem: basically inpdim and hiddim should be the same if you want to do multilayer
                    out_dim         = 7
                    }

    local batch_size = 8
    local seq_length = 3

    -- Data
    local thisp_past       = torch.random(torch.Tensor(batch_size, params.input_dim))
    local contextp         = torch.random(torch.Tensor(batch_size, params.input_dim))

    print(thisp_past:size())
    print(contextp:size())

    local encoder = init_object_encoder(params.input_dim, params.rnn_inp_dim)
    local encoder_out = encoder:forward({thisp_past, contextp})

    print(encoder_out:size())
end




-- -- nn.gModule({thisp_past,contextp, thisp_future}, {err, nn.Identity()(next_s), prediction}) 


--     local x          = torch.random(torch.Tensor(torch.LongStorage{mp.batch_size, mp.seq_length, mp.color_channels, mp.frame_height, mp.frame_width}))
--     local prev_c     = torch.ones(mp.batch_size, mp.LSTM_hidden_dim)
--     local prev_h     = torch.ones(mp.batch_size, mp.LSTM_hidden_dim)

--     -- Model
--     local encoder = init_baseline_encoder(mp, mp.LSTM_input_dim)
--     local lstm = nn.SteppableLSTM(mp.LSTM_input_dim, mp.LSTM_hidden_dim, mp.LSTM_hidden_dim, 1, false)  -- TODO: consider the hidden-output connection of LSTM!
--     local decoder = init_baseline_decoder(mp, mp.LSTM_input_dim, mp.LSTM_hidden_dim)
--     local criterion = nn.BCECriterion()

--     -- Initial conditions
--     local embeddings = {}
--     local lstm_c = {[0]=self.lstm_init_state.initstate_c} -- internal cell states of LSTM
--     local lstm_h = {[0]=self.lstm_init_state.initstate_h} -- output values of LSTM
--     local predictions = {}

--     -- Forward pass
--     local loss = 0
--     for t = 1,mp.seq_length do
--         embeddings[t] = encoder:forward(torch.squeeze(x[{{},{t}}]))
        
--         lstm_c[t], lstm_h[t] = unpack(self.clones.lstm[t]:forward{embeddings[t], lstm_c[t-1], lstm_h[t-1]})
--         predictions[t] = self.clones.decoder[t]:forward(lstm_h[t])
--         loss = loss + self.clones.criterion[t]:forward(predictions[t], torch.squeeze(y[{{},{t}}]))

--         -- DEBUGGING
--         --image.display(predictions[t])  -- here you can also save the image
--     end
--     collectgarbage()
--     return loss, embeddings, lstm_c, lstm_h, predictions
-- end 





-- test_model()
-- test_encoder()


