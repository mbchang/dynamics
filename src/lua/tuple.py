for line in open('scratch2.txt','r').readlines():
    if 'lstmcat' in line:
        x = 'LSTMCAT'
    elif 'modelind' in line:
        x = 'Independent'
    elif 'modelbl' in line or 'modellstm' in line:
        if 'modelbl' in line:
            x = 'Bidirectional LSTM'
        else:
            x = 'LSTM'

        # # find layers
        # layers_begin = line.find('layers')+len('layers')
        # layers_end = layers_begin + line[layers_begin:].find('_')
        # layers = int(line[layers_begin:layers_end])

        # # find learning rate
        # lr_begin = line.find('_lr')+len('_lr')
        # if not line[lr_begin:lr_begin+1].isdigit():
        #     lr_begin = lr_begin + line[lr_begin:].find('_lr')+len('_lr')
        # lr_end = lr_begin + line[lr_begin:].find('_')
        # lr = float(line[lr_begin:lr_end])

        # # find dim
        # dim_begin = line.find('dim')+len('dim')
        # dim_end = dim_begin + line[dim_begin:].find('_')
        # dim = int(line[dim_begin:dim_end])

        # # x += ' Layers ' + str(layers) + ' Dim ' + str(dim) + ' LR ' + str(lr)
        # x += ' Dim ' + str(dim) + ' LR ' + str(lr)

    elif 'np' in line:
        x = 'No Pairwise'
    elif 'bffobj' in line:
        if 'nbrhd' in line:
            if 'nlan' in line:
                x = 'NPE'
            else:
                x = 'NPE'
        else:
            x = 'NPE No Neighborhood'
    else:
        assert False, 'Unknown name'

    if '_of_' in line: x += ' OF'
    if '_duo_' in line: x += ' DUO'

    # find layers
    layers_begin = line.find('layers')+len('layers')
    layers_end = layers_begin + line[layers_begin:].find('_')
    layers = int(line[layers_begin:layers_end])

    # find learning rate
    lr_begin = line.find('_lr')+len('_lr')
    if not line[lr_begin:lr_begin+1].isdigit():
        lr_begin = lr_begin + line[lr_begin:].find('_lr')+len('_lr')
    lr_end = lr_begin + line[lr_begin:].find('_')
    lr = float(line[lr_begin:lr_end])

    # find dim
    dim_begin = line.find('dim')+len('dim')
    dim_end = dim_begin + line[dim_begin:].find('_')
    dim = int(line[dim_begin:dim_end])

    x += ' Layers ' + str(layers) + ' Dim ' + str(dim) + ' LR ' + str(lr)
    # x += ' Dim ' + str(dim) + ' LR ' + str(lr)

    line = line.replace('_a', '')

    print "('" + line.strip().replace('_search','') + "', '" + x + "'),"


# for line in open('scratch2.txt','r').readlines():
#     # find layers
#     layers_begin = line.find('layers')+len('layers')
#     layers_end = layers_begin + line[layers_begin:].find('_')
#     layers = int(line[layers_begin:layers_end])

#     # find learning rate
#     lr_begin = line.find('_lr')+len('_lr')
#     lr_end = lr_begin + line[lr_begin:].find('_')
#     lr = float(line[lr_begin:lr_end])

#     # find dim
#     dim_begin = line.find('dim')+len('dim')
#     dim_end = dim_begin + line[dim_begin:].find('_')
#     dim = int(line[dim_begin:dim_end])

#     print "(" + line.strip().replace('_search','') + " '" + str(layers) + ' Layers, Dim ' + str(dim) + ' Learning Rate ' + str(lr) + "'),"


