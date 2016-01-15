local T = require 'pl.tablex'

-- From https://gist.github.com/cwarden/1207556
function catch(what)
   return what[1]
end

-- From https://gist.github.com/cwarden/1207556
function try(what)
   status, result = pcall(what[1])
   if not status then
      what[2](result)
   end
   return result
end

function subrange(t, first, last)
  local sub = {}
  for i=first,last do
    sub[#sub + 1] = t[i]
  end
  return sub
end

-- merge t2 into t1
function merge_tables(t1, t2)
    -- Merges t2 and t1, overwriting t1 keys by t2 keys when applicable
    merged_table = T.deepcopy(t1)
    for k,v in pairs(t2) do
        -- if merged_table[k] then
        --     error('t1 and t2 both contain the key: ' .. k)
        -- end
        merged_table[k]  = v
    end
    return merged_table
end

-- merge t2 into t1
-- TODO do set functions
function merge_tables_by_value(t1, t2)
    -- Merges t2 and t1, overwriting t1 keys by t2 keys when applicable
    for k,v in pairs(t1) do assert(type(k) == 'number') end
    merged_table = T.deepcopy(t1)
    for _,v in pairs(t2) do
        if isin(v, merged_table) then
            error('t1 and t2 both contain the value: ' .. v)
        end
        merged_table[#merged_table+1] = v  -- just append
    end
    return merged_table
end

function is_subset(small_table, big_table)
    for _, el in pairs(small_table) do
        if not isin(el, big_table) then
            return false
        end
    end
    return true
end

function isin(element, table)
    for _,v in pairs(table) do
        if v == element then
            return true
        end
    end
    return false
end

function is_empty(table)
    if next(table) == nil then return true end
end

function all_args_exist(args_table)
    local exist = true
    for _,a in pairs(args_table) do
        if a == nil then
            exist = false
        end
    end
    return exist
end

function is_substring(substring, string)
    return not (string:find(substring) == nil)
end

-- print(merge_tables_by_value({['a']=1}, {['b'] = 2, ['c'] = 5}))
