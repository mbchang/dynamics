import os
import numpy as np
import re
import pprint
import h5py


def convert_file(path):
    """
        input
            :type path: string 
            :param path: path of particular instance of a configuration of a world file
        output
            write a hdf5 file of data
    """
    fileobject = open(path)
    data = fileobject.readlines()

    ## Note that the input file has to be 'comma-ed' and the brackets fixed, since Scheme gives us data without commas.
    configuration   = eval(fixInputSyntax(data[0])) 
    forces          = np.array(configuration[0])
    particles       = [{attr[0]: attr[1] for attr in p} for p in configuration[1]]  # configuration is what it originally was 
    goos            = np.array(configuration[2])
    initial_pos     = np.array(eval(fixInputSyntax(data[1])))  # (numObjects, [px, py])
    initial_vel     = np.array(eval(fixInputSyntax(data[2])))  # (numObjects, [vx, vy])
    observedPath    = np.array(eval(fixInputSyntax(data[3])))  # (numSteps, [pos, vel], numObjects, [x, y])

    return particles, goos, observedPath

def construct_example(particles, goos, observedPath, starttime, windowsize):
    """
        input
            :particles: list of dictionaries (each dictionary is a particle)
                dict keys are ['elastic', 'color', 'field-color', 'mass', 'field-strength', 'size']
            :goos: list of lists
                each list is [[left, top], [right, bottom], gooStrength, color]
            :observedPath: (numSteps, [pos, vel], numObjects, [x, y])
            :starttime: start time of the example (inclusive)
            :windowsize: how many time steps this example will cover (10 in 10 out has win size of 20)

        constraints
            :starttime + windowsize < 400

        output
            :path_slice: np array (numObjects, numSteps, [px, py, vx, vy, mass])
            :goos: np array [left, top, right, bottom, gooStrength]
    """
    assert starttime + windowsize < len(observedPath)  # 400 is the total length of the video

    path_slice = observedPath[starttime:starttime+windowsize]  # (windowsize, [pos, vel], numObjects, [x,y])

    # turn it into (numObjects, numSteps, [pos, vel], [x,y])
    path_slice = np.transpose(path_slice, (2,0,1,3))

    # turn it into (numObjects, numSteps, [px, py, vx, vy])
    path_slice = path_slice.reshape(path_slice.shape[0], path_slice.shape[1], path_slice.shape[2]*path_slice.shape[3])
    num_objects, num_steps = path_slice.shape[:2]

    # get masses
    masses = tuple(np.array([p['mass'] for p in particles]) for i in xrange(num_steps))
    masses = np.column_stack(masses)  # (numObjects, numSteps)

    # turn it into (numObjects, numSteps, [px, py, vx, vy, mass])
    path_slice = np.dstack((path_slice, masses))
    assert path_slice.shape == (num_objects, num_steps, 5)

    # get goos
    goos = np.array([[goo[0][0],goo[0][1], goo[1][0], goo[1][1], goo[2]] for goo in goos])  # [left, top, right, bottom, gooStrength]

    path_slice = np.asarray(path_slice, dtype=np.float64)
    goos = np.asarray(goos, dtype=np.float64)

    return (path_slice, goos)

def get_examples_for_video(video_path, num_samples, windowsize):
    """
        Returns a list of examples for this particular video

        input 
            :video_path: str, full path to the "video"file
            :num_samples: int, number of samples from this video to get
            :windowsize: k-in-m-out means the windowsize is k+m

        output
            :list of randomly chosen examples in this video
                - each example is a list of two np arrays: [path_slice, goos]
                - number of examples is dictated by num_samples

            # stack(video_sample_particles): (num_samples_in_video, num_objects, windowsize, 5)
            # stack(video_sample_goos): (num_samples_in_video, num_goos, 5)
    """
    particles, goos, observedPath = convert_file(video_path)

    # sample randomly
    samples_idxs = np.random.choice(range(len(observedPath)-windowsize), num_samples, replace=False)  # indices
    print 'video', video_path[video_path.rfind('/')+1:]
    print 'video samples:', samples_idxs

    # separate here
    video_sample_particles = []
    video_sample_goos = []
    for starttime in samples_idxs:
        sample_particles, sample_goos = construct_example(particles, goos, observedPath, starttime, windowsize)
        video_sample_particles.append(sample_particles)
        video_sample_goos.append(sample_goos)

    return stack(video_sample_particles), stack(video_sample_goos) 

def get_examples_for_config(config_path, config_sample_idxs, num_samples_per_video, windowsize):
    """
        Returns a list of examples for this particular config

        input 
            :config_path: str, full path to the folder for this configuration
                a configuration will be something like world_m1_np=2_ng=3
            :config_sample_idxs: np array of indices of videos in the folder config_path
                config_sample_idxs were randomly chosen from the parent function
            :num_samples_per_video: int, number of samples we want to sample from each video
            :windowsize: k-in-m-out means the windowsize is k+m

        output
            :list of randomly chosen examples in randomly chosen videos
                - each example is a list of two np arrays: [path_slice, goos]
                - number of examples is dictated by num_samples

            # config_sample_particles: (num_samples_in_config, num_objects, windowsize, 5)
            # config_sample_goos: (num_samples_in_config, num_goos, 5)
    """
    config_sample_particles = []
    config_sample_goos = []
    print 'config samples idxes:', np.array(os.listdir(config_path))[config_sample_idxs]
    for video in np.array(os.listdir(config_path))[config_sample_idxs]:
        video_sample_particles, video_sample_goos = get_examples_for_video(os.path.join(config_path, video), num_samples_per_video, windowsize)
        config_sample_particles.append(video_sample_particles)
        config_sample_goos.append(video_sample_goos)

    # Concatenate along first dimension
    config_sample_particles = np.vstack(config_sample_particles)
    config_sample_goos = np.vstack(config_sample_goos)

    return config_sample_particles, config_sample_goos 


def create_datasets(data_root, num_train_samples_per, num_val_samples_per, num_test_samples_per, windowsize):
    """
        4 worlds * 30 configs * 500 videos * 400 timesteps * 1-6 particles

        # Train: 30 30 = 30^2
        # Val: 10 10 = 10^2
        # Test: 10 10 = 10^2

        # the directory hierarchy is 
            root 
                configs
                    videofiles

        Train, val, test are split on the video level, not within the video
    """
    # Hardcoded
    num_videos = 500
    num_timesteps = 400

    # Number of Examples
    num_world_configs = len(os.listdir(data_root))  # assume the first world in data_root is representative

    print 'Number of train examples:', num_world_configs * (num_train_samples_per ** 2)
    print 'Number of validation examples:', num_world_configs * (num_val_samples_per ** 2)
    print 'Number of test examples:', num_world_configs * (num_test_samples_per ** 2)

    # Number of total videos to sample per config
    num_sample_videos_per_config = num_train_samples_per + num_test_samples_per + num_test_samples_per  # 30 + 10 + 10

    # Containers
    trainset    = {}  # a dictionary of train examples, with 120 keys for world-config
    valset      = {}  # a dictionary of val examples, with 120 keys for world-config
    testset     = {}  # a dictionary of test examples, with 120 keys for world-config

    for world_config in os.listdir(data_root):
        print '\n########################################################################'
        print 'WORLD CONFIG:', world_config
        config_path = os.path.join(data_root, world_config)
        num_videos_per_config = len(os.listdir(config_path))

        # sample random videos
        sampled_videos_idxs = np.random.choice(range(num_videos_per_config), num_sample_videos_per_config, replace=False)

        # split into train and val and test. This is where we split train, val, set: on the video level
        train_sample_idxs   = sampled_videos_idxs[:num_train_samples_per]  # first part  
        val_sample_idxs     = sampled_videos_idxs[num_train_samples_per:num_train_samples_per+num_val_samples_per]  # middle part
        test_sample_idxs    = sampled_videos_idxs[num_train_samples_per+num_val_samples_per:]  # last part

        # check sizes. We defined the number of videos sampled will also be the number of samples in that video
        assert len(train_sample_idxs) == num_train_samples_per
        assert len(val_sample_idxs) == num_test_samples_per
        assert len(test_sample_idxs) == num_test_samples_per

        # add to dictionary. The values returned by get_examples_for_config are tuples!
        print '\nTRAINSET'
        trainset[world_config]    = get_examples_for_config(config_path, train_sample_idxs, num_train_samples_per, windowsize)
        print '\nVALSET'
        valset[world_config]      = get_examples_for_config(config_path, val_sample_idxs, num_val_samples_per, windowsize)
        print '\nTESTSET'
        testset[world_config]     = get_examples_for_config(config_path, test_sample_idxs, num_test_samples_per, windowsize)

    # flatten the datasets and add masks
    trainset = flatten_dataset(trainset)
    valset = flatten_dataset(valset)
    testset = flatten_dataset(testset)

    # save each dictionary as a separate h5py file
    return trainset, valset, testset

def flatten_dataset(dataset):
    flattened_dataset = {}
    for k in dataset.keys():
        flattened_dataset[k+'particles'] = dataset[k][0]  # (num_samples, num_particles, windowsize, [px, py, vx, vy, mass])
        flattened_dataset[k+'goos'] = dataset[k][1]  # (num_samples, num_goos, [left, top, right, bottom, gooStrength])
        mask = np.zeros(5)  # max number of particles is 6, so mask is 5
        num_particles = dataset[k][0].shape[1]
        mask[:num_particles-1] = 1
        flattened_dataset[k+'mask'] = mask
    return flattened_dataset

def stack(list_of_nparrays):
    """
        input
            :nparray: list of numpy arrays
        output
            :stack each numpy array along a new dimension: axis=0
    """
    stacked = np.vstack(([np.expand_dims(x,axis=0) for x in list_of_nparrays]))
    assert stacked.shape[0] == len(list_of_nparrays)
    assert stacked.shape[1:] == list_of_nparrays[0].shape
    return stacked


def save_dict_to_hdf5(dataset, dataset_name, dataset_folder):
    print '\nSaving', dataset_name
    h = h5py.File(os.path.join(dataset_folder, dataset_name + '.h5'), 'w')
    print dataset.keys()
    for k, v in dataset.items():
        print 'Saving', k
        h.create_dataset(k, data=v, dtype='float64')
    h.close()
    print 'Reading saved file'
    g = h5py.File(os.path.join(dataset_folder, dataset_name + '.h5'), 'r')
    for k in g.keys():
        print k
        print g[k][:].shape
    g.close()

def save_all_datasets(data_root, dataset_files_folder):
    """
        data_root: folder containing data
        dataset_files_folder: folder to save the dataset files in
    """
    windowsize = 20  # 2
    num_train_samples_per = 30  # 3
    num_val_samples_per = 10  # 1
    num_test_samples_per = 10  # 1

    # windowsize = 10  # 20
    # num_train_samples_per = 3  # 30
    # num_val_samples_per = 1  # 10
    # num_test_samples_per = 1  # 10

    trainset, valset, testset = create_datasets(data_root, num_train_samples_per, num_val_samples_per, num_test_samples_per, windowsize)
    
    # save
    print '\n########################################################################'
    print 'SAVING'
    save_dict_to_hdf5(trainset, 'trainset', dataset_files_folder)
    save_dict_to_hdf5(valset, 'valset', dataset_files_folder)
    save_dict_to_hdf5(testset, 'testset', dataset_files_folder)

    
def fixInputSyntax(l):
    """
    # helper function for putting commas in the
    # files we get from Church, which don't contain
    # them, a fact Python does not care for.
    # also changes lists () to tuples [], which is
    # important for indexing and other handling
    """
    # remove multiple contiguous whitespaces
    l = re.sub( '\s+', ' ', l ).strip()  

    l = re.sub(r'([a-z])(\))', r'\1"\2', l)  # put a quotation after a word before parentheses
    l = re.sub(r'([a-z])(\s)', r'\1"\2', l)  # put a quotation after a word before space
    l = re.sub(r'(\()([a-z])', r'\1"\2', l)  # put a quotation before a word after parentheses
    l = re.sub(r'(\s)([a-z])', r'\1"\2', l)  # put a quotation before a word after space
    l = re.sub(r'(\")(\d+\.*\d*)(\")', r'\2', l)  # remove quotations around numbers

    # convert to list representation with commas
    l = l.replace(' ', ',').replace('(', '[').replace(')', ']')

    # find index of first "'" and index of last ',' to get list representation
    begin = l.find("'")+1
    end = l.rfind(",")
    l = l[begin:end]

    # remove all "'"
    l = l.replace("'","")

    return l


def pythonToGraphics(path, framerate, movie_folder, movieName):
    """
    Convert data into a numpy array where shape is (numSteps, 2, numObjects, 2)
        shape[0]: number of timesteps
        shape[1] = 2 means that array[:,0,:,:] are the positions, array[:,1,:,:] are the velocities
        shape[2]: number of of objects
        shape[3]: 2 means that array[:,:,:,0] are the x coordinates, array[:,:,:,1] are the y coordinates

    """
    ## Get the data.
    ## data is the x-y coordinates of the particles over times, organized as (STEP0, STEP1, STEP2...)
    ## Where each 'Step' consists of (PARTICLE0, PARTICLE1...)
    ## and each PARTICLE consists of (x, y).
    ## So, for example, in order to get the x-coordinates of particle 1 in time-step 3, we would do data[3][1][0]

    WINSIZE = 640,480
    pygame.init()
    screen = pygame.display.set_mode(WINSIZE)
    clock = pygame.time.Clock()
    screen.fill(THECOLORS["white"])
    pygame.draw.rect(screen, THECOLORS["black"], (3,1,639,481), 45)
        
    fileobject = open(path)
    data = fileobject.readlines()

    ## Note that the input file has to be 'comma-ed' and the brackets fixed, since Scheme gives us data without commas.
    configuration   = eval(fixInputSyntax(data[0])) 
    forces          = np.array(configuration[0])
    particles       = [{attr[0]: attr[1] for attr in p} for p in configuration[1]]  # configuration is what it originally was 
    goos            = np.array(configuration[2])
    initial_pos     = np.array(eval(fixInputSyntax(data[1])))  # (numObjects, [px, py])
    initial_vel     = np.array(eval(fixInputSyntax(data[2])))  # (numObjects, [vx, vy])
    observedPath    = np.array(eval(fixInputSyntax(data[3])))  # (numSteps, [pos, vel], numObjects, [x, y])


    # Set up masses, their number, color, and size
    numberOfParticles   = len(particles)
    sizes               = [p['size'] for p in particles]
    particleColors      = [p['color'] for p in particles]
    fieldColors         = [p['field-color'] for p in particles]

    ## Set up the goo patches, if any
    ## (a goo is list of [ul-corner, br-corner, resistence, color])
    gooList = goos

    ## Set up obstacles, if any
    ## (an obstacle is list of [ul-corner, br-corner, color])
    obstacleColor = "black"
    obstacleList = [] #fixedInput[3]

    ## Set up clouds, if any
    pausePoint = 149
    useCloud = False
    testCloud = cloud.Cloud([210,170], 350, 350, max(0,pausePoint - 190), pausePoint)
        
    ## Create particle objects using a loop over the particle class
    for particleIndex in range(numberOfParticles):
        pcolor = THECOLORS[particleColors[particleIndex]]
        fcolor = THECOLORS[fieldColors[particleIndex]]
        exec('particle' + str(particleIndex) + \
             ' = particle.Particle( screen, (sizes[' + str(particleIndex) + '],sizes[' + str(particleIndex) + ']), getParticleCoords(observedPath,' + \
             str(particleIndex) + '), THECOLORS["white"],' + str(pcolor)+ ',' + str(fcolor) + ')') 

    movieFrame = 0
    madeMovie = False
    frameAllocation = 4
    basicString = '0'*frameAllocation
    # if not os.path.isdir("movies/" + movieName):
    #     os.mkdir("movies/" + movieName)

    # when to pause

    if useCloud == False:
        maxPath = len(observedPath)
    else:
        maxPath = pausePoint
    # The Main Event Loop
    done = False    
    while not done:
        clock.tick(float(framerate))
        screen.fill(THECOLORS["white"])
        pygame.draw.rect(screen, THECOLORS["black"], (3,1,639,481), 45)  # draw border


        # fill the background with goo, if there is any
        if len(gooList) > 0:
            for goo in gooList:
                pygame.draw.rect(screen, THECOLORS[goo[3]], \
                                 Rect(goo[0][0], goo[0][1], abs(goo[1][0]-goo[0][0]), abs(goo[1][1]-goo[0][1])))


        # fill in the obstacles, if there is any
        if len(obstacleList) > 0:
            for obstacle in obstacleList:
                pygame.draw.rect(screen, THECOLORS[obstacle[2]], \
                                 Rect(obstacle[0][0], obstacle[0][1], \
                                      abs(obstacle[1][0]-obstacle[0][0]), abs(obstacle[1][1]-obstacle[0][1])))
                 
        # Drawing handled with exec since we don't know the number of particles in advance:
        for i in range(numberOfParticles):
            if (eval('particle' + str(i) + '.frame >=' + str(maxPath-1))):
                exec('particle' + str(i) + '.frame = ' + str(maxPath-1))
            exec('particle' + str(i) + '.draw()')
            
            # Cloud handling
            if useCloud == True:
                if particle0.frame > testCloud.appearancePoint and particle0.frame < pausePoint:
                    testCloud.update()
                    testCloud.render(screen)
                if particle0.frame == pausePoint:
                    pygame.draw.rect(screen, (255,0,0), (10,450,40,470))
                    pygame.draw.rect(screen, (100,100,100), testCloud.orect)
                    pause = True
                else:
                    pygame.draw.polygon(screen, (0,255,0), [(10,470), (40,460), (10,450)])

        pygame.draw.rect(screen, THECOLORS["black"], (3,1,639,481), 45)  # draw border
        
        # Drawing finished this iteration?  Update the screen
        pygame.display.flip()

        # make movie
        if movieFrame <= (len(observedPath)-1):
            imageName = basicString[0:len(basicString) - len(str(movieFrame))] + str(movieFrame)
            imagefile = movie_folder + "/" + movieName + '-' + imageName + ".png"
            print imagefile
            pygame.image.save(screen, imagefile)
            movieFrame += 1
        elif movieFrame > (len(observedPath)-1):
            done = True


def create_all_videos():
    root = '/Users/MichaelChang/Documents/SuperUROPlink/Code/tomer_pe/physics-andreas/saved-worlds'
    movie_root = '/Users/MichaelChang/Documents/SuperUROPlink/Code/tomer_pe/physics-andreas/movies'
    framerate = 100
    for folder in os.listdir(root):
        folder_abs_path = os.path.join(root, folder)  # each folder here is a world configuration
        world_config_folder = os.path.join(movie_root, folder)
        if not os.path.isdir(world_config_folder): os.mkdir(world_config_folder)
        for worldfile in os.listdir(folder_abs_path):  # each worldfile is an instance of the world configuration = movie
            path = os.path.join(folder_abs_path, worldfile)
            movieName = worldfile[:worldfile.rfind('.ss')]  # each world file is a particular movie
            movie_folder = os.path.join(world_config_folder, movieName)
            if not os.path.isdir(movie_folder): os.mkdir(movie_folder)
            pythonToGraphics(path=path, framerate=framerate, movie_folder = movie_folder, movieName = movieName)


if __name__ == "__main__":
    save_all_datasets(data_root, dataset_files_folder)

    
