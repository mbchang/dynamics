require 'hdf5'

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
        -- print('writing'..k)
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
        return x:double()
    end
end


-- tensor (batchsize, winsize*obj_dim)
-- reshapesize (batchsize, winsize, obj_dim)
-- cropdim (dim, amount_to_take) == (dim, mp.num_future)
function crop_future(tensor, reshapesize, cropdim)
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
        crop = crop:reshape(reshapesize[1], mp.seq_length, cropdim[2] * mp.object_dim)   -- TODO RESIZE THIS (use reshape size here)
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
        return tensor:reshape(unpack(torch.totable(before)),
                                1,
                                unpack(torch.totable(after)))
    else
        error('invalid dim')
    end
end

function unpack_batch(batch, sim)
    local this, context, y, mask = unpack(batch)
    local x = {this=this,context=context}

    if not sim then
        y = crop_future(y, {y:size(1), mp.winsize-mp.num_past, mp.object_dim},
                            {2,mp.num_future})
    end

    -- unpack inputs
    local this_past     = convert_type(x.this:clone(), mp.cuda)
    local context       = convert_type(x.context:clone(), mp.cuda)
    local this_future   = convert_type(y:clone(), mp.cuda)

    assert(this_past:size(1) == mp.batch_size and
            this_past:size(2) == mp.input_dim)  -- TODO RESIZE THIS
    assert(context:size(1) == mp.batch_size and
            context:size(2)==mp.seq_length
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

-- local t = torch.rand(2,3,4)
-- print(broadcast(t,1):size())
-- print(broadcast(t,2):size())
-- -- print(broadcast(t,3):size())
-- print(broadcast(t,4):size())
