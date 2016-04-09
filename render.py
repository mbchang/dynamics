from convert_data import *

# save_all_datasets(True)

# create_all_videos('/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/data/physics-data', 'movie_root_debug')
# assert False

# visualize_results('worldm1_np=6_ng=5_[15,15].h5', 0)
# visualize_results('model_predictions/worldm1_np=6_ng=5_[3,3].h5', 0)


# FOR THIS EXAMPLE:
# h5_file = 'openmind/results_batch_size=100_seq_length=10_layers=2_rnn_dim=100_max_epochs=20floatnetwork/predictions/lr=0.0005_worldm3_np=3_ng=1_[101,200].h5'

# visualize_results(h5_file, 2)  # fail
# visualize_results(h5_file, 99)  # okay
# visualize_results(h5_file, 98)  # bounce off wall: knows boundaries
# visualize_results(h5_file, 96)  # moves in space, but noisily: it'd be nice to have crisp movement
# visualize_results(h5_file, 93)  # moves in space, but there is a slight glitch
# visualize_results(h5_file, 89)  # wobbles around: pure noise. Knows to stay close to where it's supposed to be
# visualize_results(h5_file, 79)  # KNOWS HOW TO BOUNCE OFF WALLS! (predicted after bounce though)
# visualize_results(h5_file, 65)  # KNOWS HOW TO BOUNCE OFF WALLS! (almost)
# visualize_results(h5_file, 55)  # particle-particle fail
# visualize_results(h5_file, 4)  # particle-particle fail

# # FOR THIS EXAMPLE:
# h5_file = 'openmind/results_batch_size=100_seq_length=10_layers=2_rnn_dim=100_max_epochs=20floatnetworkcurriculum/predictions/lr=0.0005_worldm3_np=2_ng=2_[101,200].h5'
#
# # visualize_results(h5_file, 2)  # fail
# # visualize_results(h5_file, 99)  # okay
# # visualize_results(h5_file, 98)  # bounce off wall: knows boundaries
# visualize_results(h5_file, 96)  # moves in space, but noisily: it'd be nice to have crisp movement
# # visualize_results(h5_file, 93)  # moves in space, but there is a slight glitch
# visualize_results(h5_file, 89)  # wobbles around: pure noise. Knows to stay close to where it's supposed to be
# visualize_results(h5_file, 79)  # KNOWS HOW TO BOUNCE OFF WALLS!

# 1/20/16: in summary: it can handle going straight, but it cannot bounce off objects
# h5_file = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/oplogs2/baselinesubsampled_opt_adam_lr_0.001/predictions/worldm1_np=5_ng=4_[1,50].h5'
# visualize_results(h5_file, 1)  # can bounce off walls
# visualize_results(h5_file, 2)  # does not learn to bounce off other objects
# visualize_results(h5_file, 3)  # does not learn to bounce off other objects Need a crisper way to model collisions
# visualize_results(h5_file, 4)  # can definitely bounce off walls. I think we just need more training examples of particle collisions
# visualize_results(h5_file, 5)    # soft "bounce"
# visualize_results(h5_file, 6)    # reproduces linear motion very nicely
# visualize_results(h5_file, 7)    # bounces off imaginary wall, soft bounce. Note though that a lot of the ground truth also have soft bounces
# visualize_results(h5_file, 8)    # instead of bouncing, it slows down
# visualize_results(h5_file, 9)    # no obj-obj bouncing interaction
# visualize_results(h5_file, 10)    # bounces off imaginary wall
# visualize_results(h5_file, 11)     # bounces off imaginary wall. How do something that is crisp?
# visualize_results(h5_file, 12)     # bounces off imaginary wall. How do something that is crisp?
# visualize_results(h5_file, 15)     #  GREAT EXAMPLE OF BOUNCING OFF WALL
# visualize_results(training_samples_hdf5=h5_file, sample_num=16, vidsave=False, imgsave=False)     #  did not bounce against other object; good example
# visualize_results(h5_file, 18)     #  definitive example of NOT BOUNCING OFF OBJECTS
# visualize_results(h5_file, 19)     # GREAT EXAMPLE OF BOUNCING OFF WALL
# visualize_results(h5_file, 26)     # DOES NOT BOUNCE OFF OTHER OBJECTS
# visualize_results(h5_file, 27)     # Bounces off corner
# visualize_results(h5_file, 34)     # Example of a bad ground truth rendering

# h5_file = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/oplogs2/baselinesubsampled_opt_adam_lr_0.001/predictions/worldm1_np=1_ng=4_[1,50].h5'
# visualize_results(training_samples_hdf5=h5_file, sample_num=16, vidsave=False, imgsave=False)     #  moved slower
# visualize_results(training_samples_hdf5=h5_file, sample_num=7, vidsave=False, imgsave=False)     #  bounces well off wall
# visualize_results(training_samples_hdf5=h5_file, sample_num=2, vidsave=True, imgsave=False)     #  straight line

# h5_file = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/oplogs2/baselinesubsampled_opt_adam_lr_0.001/predictions/worldm1_np=2_ng=0_[1,50].h5'
# visualize_results(training_samples_hdf5=h5_file, sample_num=2, vidsave=False, imgsave=False)       # soft bounce off wall
# visualize_results(training_samples_hdf5=h5_file, sample_num=3, vidsave=True, imgsave=False)        # moves straight
# visualize_results(training_samples_hdf5=h5_file, sample_num=6, vidsave=False, imgsave=False)        # moves wrong direction
# visualize_results(training_samples_hdf5=h5_file, sample_num=9, vidsave=True, imgsave=False)         # CANNOT BOUNCE               # SAVED
# visualize_results(training_samples_hdf5=h5_file, sample_num=18, vidsave=False, imgsave=False)         # Great bounce
# visualize_results(training_samples_hdf5=h5_file, sample_num=20, vidsave=False, imgsave=False)         # moves wrong direction

# h5_file = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/oplogs2/baselinesubsampled_opt_adam_lr_0.001/predictions/worldm1_np=3_ng=0_[1,50].h5'
# visualize_results(training_samples_hdf5=h5_file, sample_num=1, vidsave=True, imgsave=False)       # CANNOT BOUNCE                 # SAVED
# visualize_results(training_samples_hdf5=h5_file, sample_num=2, vidsave=False, imgsave=False)       # CANNOT BOUNCE                # SAVED
# visualize_results(training_samples_hdf5=h5_file, sample_num=7, vidsave=False, imgsave=False)       # CANNOT BOUNCE

# h5_file = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/oplogs2/baselinesubsampled_opt_adam_lr_0.001/predictions/worldm4_np=5_ng=4_[1,50].h5'
# visualize_results(training_samples_hdf5=h5_file, sample_num=1, vidsave=False, imgsave=False)       # Stays stationary if stationary
# visualize_results(training_samples_hdf5=h5_file, sample_num=5, vidsave=False, imgsave=False)       # Friction seems to have an effect?
# visualize_results(training_samples_hdf5=h5_file, sample_num=9, vidsave=False, imgsave=False)       # Cannot bounce off objects
# visualize_results(training_samples_hdf5=h5_file, sample_num=11, vidsave=False, imgsave=False)       # An example that performs well
# visualize_results(training_samples_hdf5=h5_file, sample_num=12, vidsave=False, imgsave=False)       # Cannot bounce off objects
# visualize_results(training_samples_hdf5=h5_file, sample_num=13, vidsave=True, imgsave=False)       # CANNOT BOUNCE OFF OBJECTS    # SAVED


# 1/25/16 only 2 balls
# h5_file = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/oplogs/baselinesubsampledcontig_opt_optimrmsprop_testcfgs_[:-2:2-:]_traincfgs_[:-2:2-:]_lr_0.001/predictions/worldm1_np=2_ng=0_[1,50].h5'
# visualize_results(training_samples_hdf5=h5_file, sample_num=34, vidsave=False, imgsave=False)        # inertia good
# visualize_results(training_samples_hdf5=h5_file, sample_num=7, vidsave=False, imgsave=False)        # CANNOT BOUNCE
# visualize_results(training_samples_hdf5=h5_file, sample_num=10, vidsave=False, imgsave=False)        # Bad bounce off wall
# visualize_results(training_samples_hdf5=h5_file, sample_num=13, vidsave=False, imgsave=False)        # Soft bounce off corner
# visualize_results(training_samples_hdf5=h5_file, sample_num=29, vidsave=True, imgsave=False)        # Great bounce off wall
# visualize_results(training_samples_hdf5=h5_file, sample_num=33, vidsave=False, imgsave=False)        # can bounce off objects (maybe?)
# visualize_results(training_samples_hdf5=h5_file, sample_num=38, vidsave=False, imgsave=False)           # cannot bounce off objects (it seems to tweak physics such that it doesn't have to bounce off the other guy)

# h5_file = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/oplogs/baselinesubsampledcontig_opt_optimrmsprop_testcfgs_[:-2:2-:]_traincfgs_[:-2:2-:]_lr_0.001/predictions/worldm1_np=2_ng=0_[51,100].h5'
# visualize_results(training_samples_hdf5=h5_file, sample_num=2, vidsave=False, imgsave=False)        # bad bounce off wall
# visualize_results(training_samples_hdf5=h5_file, sample_num=7, vidsave=True, imgsave=False)        # CANNOT BOUNCE          # Saved
# visualize_results(training_samples_hdf5=h5_file, sample_num=22, vidsave=True, imgsave=False)       # Knows that there was an obj obj bounce in past    #SAVED
# visualize_results(training_samples_hdf5=h5_file, sample_num=26, vidsave=True, imgsave=False)       # Remembers one wall, but not really the other    #SAVED
# visualize_results(training_samples_hdf5=h5_file, sample_num=32, vidsave=False, imgsave=False)       # Switches direction somehow
# visualize_results(training_samples_hdf5=h5_file, sample_num=38, vidsave=True, imgsave=False)       # DEFINITIVE CANNOT BOUNCE      # SAVED  # SHOW THIS
# visualize_results(training_samples_hdf5=h5_file, sample_num=46, vidsave=True, imgsave=False)       # cannot bounce
# visualize_results(training_samples_hdf5=h5_file, sample_num=49, vidsave=False, imgsave=False)       # knows that there is a corner

# h5_file = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/oplogs/baselinesubsampledcontigdense_opt_adam_testcfgs_[:-2:2-:]_traincfgs_[:-2:2-:]_lr_0.001_batch_size_260/predictions/worldm1_np=2_ng=0_[1,260].h5'
# h5_file = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/oplogs/baselinesubsampledcontigdense2_opt_adam_traincfgs_[:-2:2-:]_shuffle_false_lrdecay_1_batch_size_260_testcfgs_[:-2:2-:]_lr_0.001/predictions/worldm1_np=2_ng=0_[1,260].h5'
# visualize_results(training_samples_hdf5=h5_file, sample_num=5, vidsave=True, imgsave=False)        # CANNOT BOUNCE OFF OBJECTS
# visualize_results(training_samples_hdf5=h5_file, sample_num=14, vidsave=True, imgsave=False)        # CANNOT BOUNCE OFF OBJECTS
# visualize_results(training_samples_hdf5=h5_file, sample_num=26, vidsave=True, imgsave=False)        # Fast movement
# visualize_results(training_samples_hdf5=h5_file, sample_num=30, vidsave=True, imgsave=False)        # CANNOT BOUNCE OFF OBJECTS
# visualize_results(training_samples_hdf5=h5_file, sample_num=43, vidsave=True, imgsave=False)        # CANNOT BOUNCE OFF OBJECTS
# visualize_results(training_samples_hdf5=h5_file, sample_num=53, vidsave=False, imgsave=False)        # CANNOT BOUNCE OFF OBJECTS


# h5_file ='/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/oplogs/baselinesubsampledcontigdense3_opt_adam_traincfgs_[:-2:2-:]_shuffle_true_lrdecay_0.99_batch_size_260_testcfgs_[:-2:2-:]_lr_0.005/predictions/worldm1_np=2_ng=0_[1,260].h5'
# visualize_results(training_samples_hdf5=h5_file, sample_num=5, vidsave=False, imgsave=False)
# for i in range(1, 20):
    # print(len(subsample_range(80, 2, i))), subsample_range(80, 20, i)
# print(len(subsample_range(80, 2, 60))), subsample_range(80, 2, 60)
# render_from_scheme_output('/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/data/physics-data/worldm1_np=1_ng=0/worldm1_np=1_ng=0_324.ss', 3, 'heyhey', 'hihi', False)

# h5_file = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/oplogs/5_Sl1BCELinearReLU_opt_optimrmsprop_lr_0.001/predictions/worldm1_np=2_ng=0_[1,65].h5'
# visualize_results(training_samples_hdf5=h5_file, sample_num=14, vidsave=False, imgsave=False)   # CANNOT BOUNCE OFF OBJECTS

# h5_file = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/oplogs/2_TanhReLU_opt_optimrmsprop_layers_2_traincfgs_[:-2:2-:]_shuffle_true_lrdecay_0.99_batch_size_65_testcfgs_[:-2:2-:]_lr_0.001_max_epochs_20/predictions/worldm1_np=2_ng=0_[1,65].h5'
# visualize_results(training_samples_hdf5=h5_file, sample_num=5, vidsave=False, imgsave=False)        # CANNOT BOUNCE OFF OBJECTS

#
# h5_file = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/oplogs/4_SL1TanhReLU_opt_adam_lr_0.001/predictions/worldm1_np=2_ng=0_[1,65].h5'
# visualize_results(training_samples_hdf5=h5_file, sample_num=5, vidsave=False, imgsave=False)        # CANNOT BOUNCE OFF OBJECTS

# print subsample_range(80, 20, 60)

# h5_file = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/oplogs/6_SL1BCELinearReLURel_opt_optimrmsprop_lr_0.001/predictions/worldm1_np=2_ng=0_[1,65].h5'
# visualize_results(training_samples_hdf5=h5_file, sample_num=15, vidsave=False, imgsave=False)        # CANNOT BOUNCE OFF OBJECTS ON TRAINING DATA

# h5_file = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/oplogs/9_opt_optimrmsprop_layers_2_lr_0.005/predictions/worldm1_np=2_ng=0_[1,80].h5'
# visualize_results(training_samples_hdf5=h5_file, sample_num=9, vidsave=False, imgsave=False)

# h5_file = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/oplogs/10_opt_optimrmsprop_layers_2_rnn_dim_256_lr_0.0005/predictions/worldm1_np=2_ng=0_[1,400].h5'
# visualize_results(training_samples_hdf5=h5_file, sample_num=3, vidsave=False, imgsave=False)    # weird movement
# visualize_results(training_samples_hdf5=h5_file, sample_num=40, vidsave=False, imgsave=False)    # cannot bounce
# visualize_results(training_samples_hdf5=h5_file, sample_num=130, vidsave=False, imgsave=False)    # cannot bounce

# running simulation
# h5_file = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/oplogs/11_opt_optimrmsprop_layers_1_rnn_dim_256_lr_0.001/predictions/worldm1_np=2_ng=0_[1,80].h5'
# visualize_results(training_samples_hdf5=h5_file, sample_num=20, vidsave=False, imgsave=False)  # possible bounce? look at 20, 21, 22
# visualize_results(training_samples_hdf5=h5_file, sample_num=53, vidsave=False, imgsave=False)  # very soft bounce off wall

# h5_file = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/oplogs/11_opt_optimrmsprop_layers_1_rnn_dim_256_lr_0.001/predictions/worldm1_np=2_ng=0_[81,160].h5'
# # visualize_results(training_samples_hdf5=h5_file, sample_num=40, vidsave=False, imgsave=False)  # inertia, simulation
# visualize_results(training_samples_hdf5=h5_file, sample_num=50, vidsave=False, imgsave=False)

# h5_file = '/Users/MichaelChang/Dropbox (MIT Solar Car Team)/MacHD/Documents/Important/MIT/Research/SuperUROP/Code/dynamics/oplogs/12_opt_optimrmsprop_layers_1_rnn_dim_256_lr_0.005/predictions/worldm1_np=2_ng=0_[31,60].h5'
# visualize_results(training_samples_hdf5=h5_file, sample_num=29, vidsave=False, imgsave=False)


# 2/23/16: experiments 12 tested priority sampling
# h5_file = '/Users/MichaelChang/Dropbox (MIT Solar Car Team)/MacHD/Documents/Important/MIT/Research/SuperUROP/Code/dynamics/oplogs/12c_opt_optimrmsprop_layers_2_rnn_dim_256_lr_0.001_lrdecay_0.97/predictions/worldm1_np=2_ng=0_[1,30].h5'
# visualize_results(training_samples_hdf5=h5_file, sample_num=3, vidsave=False, imgsave=False)  # NO BOUNCE?  for [1,30]
# visualize_results(training_samples_hdf5=h5_file, sample_num=15, vidsave=False, imgsave=False)  # good crisp wall bounce for [31, 60]
# visualize_results(training_samples_hdf5=h5_file, sample_num=10, vidsave=False, imgsave=False)  # seems hesitant to move when close to other ball for [91,120]


# h5_file = '/Users/MichaelChang/Dropbox (MIT Solar Car Team)/MacHD/Documents/Important/MIT/Research/SuperUROP/Code/dynamics/oplogs/12_opt_optimrmsprop_layers_1_rnn_dim_256_lr_0.005_sharpen_2/predictions/worldm1_np=2_ng=0_[91,120].h5'
# visualize_results(training_samples_hdf5=h5_file, sample_num=29, vidsave=False, imgsave=False)  # NO BOUNCE?  for [1,30]

# 3/2/16: experiments 13 tested velocity only
# h5_file = '/Users/MichaelChang/Dropbox (MIT Solar Car Team)/MacHD/Documents/Important/MIT/Research/SuperUROP/Code/dynamics/oplogs/13_layers_3_sharpen_1_lr_0.005/predictions/worldm1_np=2_ng=0_[5521,5580].h5'
# visualize_results(training_samples_hdf5=h5_file, sample_num=45, vidsave=False, imgsave=False)   # corner bounce

# h5_file = '/Users/MichaelChang/Dropbox (MIT Solar Car Team)/MacHD/Documents/Important/MIT/Research/SuperUROP/Code/dynamics/oplogs/13_layers_3_sharpen_1_lr_0.005/predictions/worldm1_np=2_ng=0_[1,60].h5'
# visualize_results(training_samples_hdf5=h5_file, sample_num=58, vidsave=False, imgsave=False)

# Stationary test
# h5_file = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/logs/stattestpos/predictions/worldm5_np=2_ng=0_[1,50].h5'
# visualize_results(training_samples_hdf5=h5_file, sample_num=23, vidsave=False, imgsave=False)

# h5_file = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/logs/stattestpos2/predictions/worldm5_np=2_ng=0_[1321,1380].h5'
# # h5_file = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/logs/stattestpos2/predictions/worldm5_np=2_ng=0_[1,60].h5'
# visualize_results(training_samples_hdf5=h5_file, sample_num=7, vidsave=False, imgsave=False)

# Test on stationary, constrained window, nonoverlapping
# h5_file = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/logs/ffnonoverlaptest/predictions/worldm5_np=2_ng=0_nonoverlap_[1,10].h5'
# visualize_results(training_samples_hdf5=h5_file, sample_num=0, vidsave=False, imgsave=False)

# Test on stationary, constrained window, nonoverlapping, openmind
h5_file = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/oplogs/16__layers_4_sharpen_1_lr_0.0003_lrdecay_0.99/predictions/worldm5_np=2_ng=0_nonoverlap_[351,400].h5'
visualize_results(training_samples_hdf5=h5_file, sample_num=7, vidsave=False, imgsave=False)  # can do object bounces! (batch 8)
