require 'torch'
require 'paths'
require 'hdf5'
require 'data_utils'

--[[ Loads the dataset as a table of configurations

    Input
        .h5 file data
        The data for each configuration is spread out across 3 keys in the .h5 file
        Let <config> be the configuration name
            "<config>particles": (num_samples, num_particles, windowsize, [px, py, vx, vy, mass])
            "<config>goos": (num_samples, num_goos, [left, top, right, bottom, gooStrength])
            "<config>mask": binary vector of length 5, trailing zeros are padding when #particles < 6

    Output
    {
        configuration:
            {
              particles : DoubleTensor - size: (num_samples x num_particles x windowsize x 5)
              goos : DoubleTensor - size: (num_samples x num_goos x 5)
              mask : DoubleTensor - size: 5 
            }
    }

    The last dimension of particles is 5 because: [px, py, vx, vy, mass]
    The last dimension of goos is 5 because: [left, top, right, bottom, gooStrength]
    The mask is dimension 5 because: our data has at most 6 particles -- ]]
function load_data(dataset_name, dataset_folder)
    local dataset_file = hdf5.open(dataset_folder .. '/' .. dataset_name, 'r')

    -- Get all keys: note they might not be in order though!
    local examples = {}
    local subkeys = {'particles', 'goos', 'mask'}  -- hardcoded
    for k,v in pairs(dataset_file:all()) do

        -- find the subkey of interest
        local this_subkey
        local example_key
        local counter = 0
        for sk, sv in pairs(subkeys) do
            if k:find(sv) then
                counter = counter + 1
                this_subkey = sv
                example_key = k:sub(0,k:find(sv)-1)
            end
        end
        assert(counter == 1)
        assert(this_subkey and example_key)

        if examples[example_key] then
            examples[example_key][this_subkey] = v
        else
            examples[example_key] = {}
            examples[example_key][this_subkey] = v
        end
    end
    -- print(examples)
    print(get_keys(examples))
    return examples
end



-- remove trailing and leading whitespace from string.
-- http://en.wikipedia.org/wiki/Trim_(8programming)
function trim(s)
  -- from PiL2 20.4
  return (s:gsub("^%s*(.-)%s*$", "%1"))
end

-- opens a file and extracts information
function extract_info(filename)
    local raw = io.open(filename)

    -- data[1] is the physics
    -- data[2] is the initial positions
    -- data[3] is the initial velocities
    -- data[4] is the observed path
    local data = {}
    for line in raw:lines() do data[#data+1] = line end


    print(string.gsub(data[1], '\"', ''))
end

function fix_input_syntax(line)
    -- remove multiple contiguous whitespaces
    line = trim(string.gsub(line, '$s+', " "))

    line = string.gsub(line, " ", ",")  -- replace space with commas

    -- find index of first "'" and index of last ',' to get list representation
    begin = string.find(line, "'")
    finish = #line - line:reverse():find(",")
    line = string.sub(line, begin, finish)

    -- remove all "'"
    line = string.gsub(line, "'", "")

    line = string.gsub(line, '([a-z])(%))', "%1'%2")  -- put a quotation after a word before parentheses
    line = string.gsub(line, '([a-z])(,)', "%1'%2")  -- put a quotation after a word before space
    line = string.gsub(line, '(%()([a-z])', "%1'%2")  -- put a quotation before a word after parentheses
    line = string.gsub(line, '(,)([a-z])', "%1'%2")  -- put a quotation before a word after space

    line = string.gsub(line, "%(", "{")  -- replace ( with {
    line = string.gsub(line, "%)", "}")  -- repace ) with }
    print(line)
    return line
end

function test_fix_input_syntax()
    local line = "(define saved-world (list none '((1 1 attract) (1 -1 repel) (-1 1 repel)) '(((mass 3.0) (elastic 0.9) (size 4e1) (color blue) (field-color black) (field-strength 0)) ((mass 1.0) (elastic 0.9) (size 4e1) (color red) (field-color black) (field-strength 0)) ((mass 0.33) (elastic 0.9) (size 4e1) (color yellow)  (field-color black) (field-strength 0)) ((mass 1.0) (elastic 0.9) (size 4e1) (color red) (field-color black) (field-strength 0))) '(((74 206) (206 354) 0 darkmagenta) ((381 339) (493 509) -20 yellowgreen)) ))"
    print(fix_input_syntax(line))
end 


load_data('trainset.h5', 'hey')
-- load_data('valset.h5', '/om/user/mbchang/physics-data/dataset_files')
-- load_data('testset.h5', '/om/user/mbchang/physics-data/dataset_files')
