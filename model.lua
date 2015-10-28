require 'metaparams'
require 'nn'
require 'torch'
require 'nngraph'
-- require 'SteppableLSTM'

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
    -- local lstm_out      = lstm_allout[1] -- PROBLEM: lstm_out is nil? 
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


--[[

-- From "Learning to Execute"
function create_network()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local i                = {[0] = Embedding(symbolsManager.vocab_size,
                                            mp.rnn_size)(x)}
  local next_s           = {}
  local splitted         = {prev_s:split(2 * mp.layers)}
  for layer_idx = 1, mp.layers do
    local prev_c         = splitted[2 * layer_idx - 1]
    local prev_h         = splitted[2 * layer_idx]
    local next_c, next_h = lstm(i[layer_idx - 1], prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(mp.rnn_size, symbolsManager.vocab_size)
  local pred             = nn.LogSoftMax()(h2y(i[mp.layers]))
  local err              = MaskedLoss()({pred, y})
  local module           = nn.gModule({x, y, prev_s}, 
                                      {err, nn.Identity()(next_s)})
  module:getParameters():uniform(-mp.init_weight, mp.init_weight)
  return module:cuda()
end


-- --]]

-- Create Model
function test_model(mp)
    mp.batch_size = 3
    mp.seq_length = 2

    -- Data
    local x          = torch.random(torch.Tensor(torch.LongStorage{mp.batch_size, mp.seq_length, mp.color_channels, mp.frame_height, mp.frame_width}))
    local prev_c     = torch.ones(mp.batch_size, mp.LSTM_hidden_dim)
    local prev_h     = torch.ones(mp.batch_size, mp.LSTM_hidden_dim)

    -- Model
    local encoder = init_baseline_encoder(mp, mp.LSTM_input_dim)
    local lstm = nn.SteppableLSTM(mp.LSTM_input_dim, mp.LSTM_hidden_dim, mp.LSTM_hidden_dim, 1, false)  -- TODO: consider the hidden-output connection of LSTM!
    local decoder = init_baseline_decoder(mp, mp.LSTM_input_dim, mp.LSTM_hidden_dim)
    local criterion = nn.BCECriterion()

    -- Initial conditions
    local embeddings = {}
    local lstm_c = {[0]=self.lstm_init_state.initstate_c} -- internal cell states of LSTM
    local lstm_h = {[0]=self.lstm_init_state.initstate_h} -- output values of LSTM
    local predictions = {}

    -- Forward pass
    local loss = 0
    for t = 1,mp.seq_length do
        embeddings[t] = encoder:forward(torch.squeeze(x[{{},{t}}]))
        
        lstm_c[t], lstm_h[t] = unpack(self.clones.lstm[t]:forward{embeddings[t], lstm_c[t-1], lstm_h[t-1]})
        predictions[t] = self.clones.decoder[t]:forward(lstm_h[t])
        loss = loss + self.clones.criterion[t]:forward(predictions[t], torch.squeeze(y[{{},{t}}]))

        -- DEBUGGING
        --image.display(predictions[t])  -- here you can also save the image
    end
    collectgarbage()
    return loss, embeddings, lstm_c, lstm_h, predictions
end 






