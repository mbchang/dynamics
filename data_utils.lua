require 'hdf5'
require 'nn'
require 'nngraph'
require 'torchx'
local pltx = require 'pl.tablex'
local pls = require 'pl.stringx'

function get_keys(table)
    local keyset={}
    local n=0

    for k,v in pairs(table) do
        n=n+1
        keyset[n]=k
    end
    return keyset
end

function split_table(table, num_chunks)
    --[[
        input
            :type table: table
            :param table: table of elements

            :type num_chunks: int
            :param num_chunks: number of chunks you want to split the table into
        output
            :type: table of subtables
            :value: the number of subtables is num_chunks, each of size math.floor(#table/num_chunks)
    --]]
    local n = #table
    local chunk_size = math.floor(n/num_chunks)
    local splitted_table = {}
    local current_chunk = {}
    for i = 1, n do
        current_chunk[#current_chunk+1] = table[i]
        if i % chunk_size == 0 then
            splitted_table[#splitted_table+1] = current_chunk
            current_chunk = {}
        end
    end
    collectgarbage()
    return splitted_table
end

function find_all_sequences(folders_list, parent_folder_path, seq_length)
    local data_list = {}
    for f = 1, #folders_list do
        local data_path = parent_folder_path .. '/' .. folders_list[f]

        -- get number of images in this folder
        local num_images_f = io.popen('ls "' .. data_path .. '" | wc -l')
        local num_images = nil
        for x in num_images_f:lines() do num_images = x end
        local num_examples = math.floor(num_images/(seq_length))
        num_images = num_examples*seq_length

        -- cycle through images
        local p = io.popen('find "' .. data_path .. '" -type f -name "*.png"')  -- Note: this is not in order!
        local j = 0
        local ex_string = {}
        for img_name in p:lines() do
            j = j + 1
            ex_string[#ex_string+1] = data_path .. '/' .. j .. '.png'  -- force the images to be in order
            if j % seq_length == 0 then
                data_list[#data_list+1] = ex_string
                ex_string = {}
            end
        end
    end
    collectgarbage()
    return data_list
end

function save_to_hdf5(filename, data)
    -- filename: name of hdf5 file
    -- data: dict of {datapath: data}
    local myFile = hdf5.open(filename, 'w')
    for k,v in pairs(data) do
        myFile:write(k, v)  -- I can write many preds in here, indexed by the starting time?
    end
    myFile:close()
end


function concatenate_table(table)
    -- concatenates a table of torch tensors
    print(table)
    local num_tensors = #table
    print('num_tensors')
    print(num_tensors)
    local other_dims = table[1]:size()
    local dims = {num_tensors, unpack(other_dims:totable())}
    print('dims')
    print(dims)

    -- construct container
    local container = torch.zeros(unpack(dims))
    for i=1,num_tensors do
        container[{{i}}] = table[i]
    end
    return container
end

function convert_type(x, should_cuda)
    if should_cuda then
        return x:cuda()
    else
        return x:float()
    end
end


-- tensor (batchsize, winsize*obj_dim)
-- reshapesize (batchsize, winsize, obj_dim)
-- cropdim (dim, amount_to_take) == (dim, mp.num_future)
function crop_future(tensor, reshapesize, cropdim)
    print('crop_future')
    print(tensor:size())
    print(reshapesize)
    print(cropdim)

    local crop = tensor:clone()
    crop = crop:reshape(unpack(reshapesize))
    --hacky
    if crop:dim() == 3 then
        assert(cropdim[1]==2)
        crop = crop[{{},{1,cropdim[2]},{}}]  -- (num_samples x num_future x 8) -- TODO the -1 should be a function of 1+num_future
        crop = crop:reshape(reshapesize[1], cropdim[2] * mp.object_dim)
    else
        assert(crop:dim()==4 and cropdim[1] == 3)
        crop = crop[{{},{},{1,cropdim[2]},{}}]
        crop = crop:reshape(reshapesize[1], mp.seq_length,
                            cropdim[2] * mp.object_dim)   -- TODO RESIZE THIS (use reshape size here)
    end
    return crop
end

-- dim will be where the one is, and the dimensions after will be shifted right
function broadcast(tensor, dim)
    local ndim = tensor:dim()

    if dim == 1 then
        return tensor:reshape(1,unpack(torch.totable(tensor:size())))
    elseif dim == ndim + 1 then
        local dims = {unpack(torch.totable(tensor:size())),1}
        return tensor:reshape(unpack(dims))
    elseif dim > 1 and dim <= ndim then
        local before = torch.Tensor(torch.totable(tensor:size()))[{{1,dim-1}}]
        local after = torch.Tensor(torch.totable(tensor:size()))[{{dim,-1}}]
        print(before)
        print(after)
        print(unpack(torch.totable(before)))
        local a = {unpack(torch.totable(before)),1,unpack(torch.totable(after))}
        local b = {unpack(torch.totable(before)),1}
        print(a)
        print(b)
        return tensor:reshape(unpack(torch.totable(before)), 1,
                                unpack(torch.totable(after)))
    else
        error('invalid dim')
    end
end


function extract_flag(flags_list, delim)
    local extract = pltx.filter(flags_list, function(x) return pls.startswith(x, delim) end)
    assert(#extract == 1)
    return string.sub(extract[1], #delim+1)
end

function unpack_batch(batch, sim)
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

    return input, this_future
end

-- each inner table contains the same number of tensors, for which all
-- the dimensions (except the first) are the same
function join_table_of_tables(table_of_tables)
    if #table_of_tables == 0 then return table_of_tables end
    local all
    for _, inner in pairs(table_of_tables) do
        if all == nil then
            all = pltx.deepcopy(inner)
        else
            for k, tensor in pairs(inner) do
                all[k] = torch.cat({all[k], tensor:clone()}, 1)
            end
        end
    end
    return all
end


function preprocess_input(mask)
    -- in: {(bsize, input_dim), (bsize, mp.seq_length, input_dim)}
    -- out: table of length torch.find(mask,1)[1] of pairs {(bsize, input_dim), (bsize, input_dim)}

    local this_past = nn.Identity()()
    local context = nn.Identity()()

    -- this: (bsize, input_dim)
    -- context: (bsize, mp.seq_length, dim)
    local input = {}
    for t = 1, torch.find(mask,1)[1] do
        table.insert(input, nn.Identity()
                        ({this_past, nn.Squeeze()(nn.Select(2,t)(context))}))
    end
    input = nn.Identity()(input)
    return nn.gModule({this_past, context}, {input})
end


function checkpointtofloat(checkpoint)
    -- just mutates checkpoint though
    checkpoint.model.network:clearState()
    checkpoint.model.network:float()
    checkpoint.model.criterion:float()
    checkpoint.model.identitycriterion:float()
    checkpoint.model.theta.params = checkpoint.model.theta.params:float()
    checkpoint.model.theta.grad_params=checkpoint.model.theta.grad_params:float()
    return checkpoint
end

function checkpointtocuda(checkpoint)
    -- just mutates checkpoint though
    checkpoint.model.network:clearState()
    checkpoint.model.network:cuda()
    checkpoint.model.criterion:cuda()
    checkpoint.model.identitycriterion:cuda()
    checkpoint.model.theta.params = checkpoint.model.theta.params:cuda()
    checkpoint.model.theta.grad_params=checkpoint.model.theta.grad_params:cuda()
    return checkpoint
end

function unsqueeze(tensor, dim)
    local ndims = tensor:dim()
    assert(dim >= 1 and dim <= ndims+1 and dim % 1 ==0,
            'can only unsqueeze up to one extra dimension')
    local old_size = torch.totable(tensor:size())
    local j = 1
    local new_size = {}
    for i=1,ndims+1 do
        if i == dim then
            table.insert(new_size, 1)
        else
            table.insert(new_size, old_size[j])
            j = j + 1
        end
    end
    tensor = tensor:clone():reshape(unpack(new_size))
    return tensor
end

function mj_interface(batch)
    -- {
    --   1 : FloatTensor - size: 50x2x9
    --   2 : FloatTensor - size: 50x10x2x9
    --   3 : FloatTensor - size: 50x2x9
    --   4 : FloatTensor - size: 10
    --   5 : "worldm5_np=2_ng=0_slow"
    --   6 : 1
    --   7 : 50
    --   8 : FloatTensor - size: 50x10x2x9
    -- }

    local focus_past = batch[1]
    local context_past = batch[2]
    local focus_future = batch[3]
    local mask = batch[4]
    local config_name = batch[5]
    local start = batch[6]
    local finish = batch[7]
    local context_future = batch[8]

    return {focus_past, context_past, focus_future, context_future, mask}
end

-- mask = torch.Tensor({1,0,0,0,0,0,0,0,0,0})
-- p = preprocess_input(mask)
--
-- bsize = 3
-- tp = torch.rand(bsize,10)
-- c = torch.rand(bsize,10,10)
-- x = p:forward({tp, c})
--
-- print(x)
--
-- p:backward({tp,c},x)
-- print(p.gradInput[1])
