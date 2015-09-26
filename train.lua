require 'torch'
--require 'optim'
require 'image'
--require 'datasets'
require 'pl' -- this is somehow responsible for lapp working in qlua mode
require 'paths'
ok, DISP = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
ADVERSARIAL = require 'adversarial'
DATASET = require 'dataset'
NN_UTILS = require 'utils.nn_utils'


----------------------------------------------------------------------
-- parse command-line options
OPT = lapp[[
  -s,--save          (default "logs")      subdirectory to save logs
  --saveFreq         (default 10)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  --V_network        (default "logs/v.net")
  -p,--plot                                plot while training
  --SGD_lr           (default 0.02)        SGD learning rate
  -b,--batchSize     (default 16)          batch size
  --SGD_momentum     (default 0)           SGD momentum
  --N_epoch          (default 1000)        Number of examples per epoch (-1 means all)
  --G_L1             (default 0)           L1 penalty on the weights of G
  --G_L2             (default 0e-6)        L2 penalty on the weights of G
  --D_L1             (default 1e-7)        L1 penalty on the weights of D
  --D_L2             (default 0e-6)        L2 penalty on the weights of D
  --D_iterations     (default 1)           number of iterations to optimize D for
  --G_iterations     (default 1)           number of iterations to optimize G for
  --D_maxAcc         (default 1.01)
  --D_clamp          (default 1)
  --G_clamp          (default 5)
  --D_optmethod      (default "adam")      adam|adagrad
  --G_optmethod      (default "adam")      adam|adagrad
  -t,--threads       (default 8)           number of threads
  -g,--gpu           (default -1)          gpu to run on (default cpu)
  -d,--noiseDim      (default 100)         dimensionality of noise vector
  -w, --window       (default 3)           window id of sample image
  --scale            (default 32)          scale of images to train on
  --grayscale                              grayscale mode on/off
  --autoencoder      (default "")          path to autoencoder to load weights from
  --rebuildOptstate  (default 0)           whether to force a rebuild of the optimizer state
  --seed             (default 1)           seed for the RNG
  --weightsVisFreq   (default 0)
  --aws                                    run in AWS mode
]]

START_TIME = os.time()

if OPT.gpu < 0 or OPT.gpu > 3 then OPT.gpu = false end
print(OPT)

-- fix seed
torch.manualSeed(OPT.seed)

-- threads
torch.setnumthreads(OPT.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- possible output of disciminator
CLASSES = {"0", "1"}
Y_GENERATOR = 0
Y_NOT_GENERATOR = 1

-- axis of images: 3 channels, <scale> height, <scale> width
if OPT.grayscale then
    IMG_DIMENSIONS = {1, OPT.scale, OPT.scale}
else
    IMG_DIMENSIONS = {3, OPT.scale, OPT.scale}
end
-- size in values/pixels per input image (channels*height*width)
INPUT_SZ = IMG_DIMENSIONS[1] * IMG_DIMENSIONS[2] * IMG_DIMENSIONS[3]

----------------------------------------------------------------------
-- get/create dataset
----------------------------------------------------------------------
DATASET.nbChannels = IMG_DIMENSIONS[1]
DATASET.setFileExtension("jpg")
DATASET.setScale(OPT.scale)

-- 199,840 in 10k cats
-- 111,344 in flickr cats
if OPT.aws then
    DATASET.setDirs({"/mnt/datasets/out_faces_64x64", "/mnt/datasets/images_faces_aug"})
else
    --DATASET.setDirs({"/media/aj/ssd2a/ml/datasets/10k_cats/out_faces_64x64", "/media/aj/ssd2a/ml/datasets/flickr-cats/images_faces_aug"})
    DATASET.setDirs({"/media/aj/ssd2a/ml/datasets/10k_cats/out_faces_64x64"})
end
----------------------------------------------------------------------

-- run on gpu if chosen
if OPT.gpu then
    print("<trainer> starting gpu support...")
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(OPT.gpu + 1)
    cutorch.manualSeed(OPT.seed)
    print(string.format("<trainer> using gpu device %d", OPT.gpu))
else
    require 'nn'
end
require 'dpnn'
require 'LeakyReLU'
--require 'AddGaussianNoise'
torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------
-- Load / Define network
----------------------------------------------------------------------
local tmp = torch.load(OPT.V_network)
MODEL_V = tmp.V
MODEL_V:evaluate()

function rateWithV(images)
    local imagesTensor
    local N
    if type(images) == 'table' then
        N = #images
        imagesTensor = torch.Tensor(N, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
        for i=1,N do
            imagesTensor[i] = images[i]
        end
    else
        N = images:size(1)
        imagesTensor = images
    end
    
    local predictions = MODEL_V:forward(imagesTensor)
    local sm = 0
    for i=1,N do
        -- first neuron in V signals whether the image is fake (1=yes, 0=no)
        sm = sm + predictions[i][1]
    end
    
    local fakiness = sm / N
    
    -- higher values for better images
    return (1 - fakiness)
end

-- load previous networks (D and G)
-- or initialize them new
if OPT.network ~= "" then
    print(string.format("<trainer> reloading previously trained network: %s", OPT.network))
    local tmp = torch.load(OPT.network)
    MODEL_D = tmp.D
    MODEL_G = tmp.G
    OPTSTATE = tmp.optstate
    EPOCH = tmp.epoch
else
  --------------
  -- D
  --------------
  
  --[[
  local activation = nn.PReLU
  local branch_conv = nn.Sequential()
  --branch_conv:add(nn.SpatialDropout())
  branch_conv:add(nn.SpatialConvolution(IMG_DIMENSIONS[1], 128, 3, 3, 1, 1, (3-1)/2))
  branch_conv:add(activation())
  branch_conv:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
  branch_conv:add(activation())
  branch_conv:add(nn.SpatialMaxPooling(2, 2))
  branch_conv:add(nn.SpatialDropout())
  branch_conv:add(nn.View(128, 0.25*32*32))
  
  local parallel = nn.Parallel(2, 2)
  for i=1,128 do
    parallel:add(nn.Linear(0.25*32*32, 64))
  end
  branch_conv:add(parallel)
  branch_conv:add(activation())
  branch_conv:add(nn.Dropout())
  --branch_conv:add(nn.AddGaussianNoise(0.0, 0.10))
  
  --branch_conv:add(nn.View(64 * 32 * 32))
  --branch_conv:add(nn.Linear(64 * 32 * 32, 2048))
  branch_conv:add(nn.Linear(128*64, 2048))
  branch_conv:add(activation())
  branch_conv:add(nn.Dropout())
  --branch_conv:add(nn.AddGaussianNoise(0.0, 0.10))
  branch_conv:add(nn.Linear(2048, 2048))
  branch_conv:add(activation())
  branch_conv:add(nn.Dropout())
  --branch_conv:add(nn.AddGaussianNoise(0.0, 0.10))
  branch_conv:add(nn.Linear(2048, 1))
  branch_conv:add(nn.Sigmoid())
  MODEL_D = branch_conv
  --]]
  
  local activation = nn.PReLU
  local branch_conv = nn.Sequential()
  
  branch_conv:add(nn.Dropout(0.1))
  
  branch_conv:add(nn.SpatialConvolution(IMG_DIMENSIONS[1], 32, 3, 3, 1, 1, (3-1)/2))
  branch_conv:add(activation())
  --branch_conv:add(nn.Dropout())
  branch_conv:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, (3-1)/2))
  branch_conv:add(activation())
  --branch_conv:add(nn.SpatialMaxPooling(2, 2))
  --branch_conv:add(nn.SpatialDropout())
  --branch_conv:add(nn.Dropout())
  
  branch_conv:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, (3-1)/2))
  branch_conv:add(activation())
  branch_conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2))
  branch_conv:add(activation())
  --branch_conv:add(nn.SpatialDropout())
  branch_conv:add(nn.SpatialMaxPooling(2, 2))
  branch_conv:add(nn.Dropout())
  branch_conv:add(nn.View(128, 16 * 16))
  
  local parallel = nn.Parallel(2, 2)
  for i=1,128 do
    local lin = nn.Sequential()
    lin:add(nn.Linear(16*16, 64))
    lin:add(activation())
    lin:add(nn.Dropout())
    lin:add(nn.Linear(64, 16))
    lin:add(activation())
    lin:add(nn.TotalDropout())
    parallel:add(lin)
  end
  branch_conv:add(parallel)
  
  branch_conv:add(nn.Linear(128*16, 1024))
  branch_conv:add(activation())
  branch_conv:add(nn.Dropout())
  
  branch_conv:add(nn.Linear(1024, 1024))
  branch_conv:add(activation())
  branch_conv:add(nn.Dropout())
  
  branch_conv:add(nn.Linear(1024, 1))
  branch_conv:add(nn.Sigmoid())
  
  MODEL_D = branch_conv
  
  --------------
  -- G
  --------------
  if OPT.autoencoder ~= "" then
      local left = nn.Sequential()
      left:add(nn.View(INPUT_SZ))
      local right = nn.Sequential()
      right:add(nn.View(INPUT_SZ))
      right:add(nn.Linear(INPUT_SZ, 1024))
      right:add(nn.PReLU())
      right:add(nn.BatchNormalization(1024))
      right:add(nn.Linear(1024, 1024))
      right:add(nn.PReLU())
      right:add(nn.BatchNormalization(1024))
      right:add(nn.Linear(1024, INPUT_SZ))
      right:add(nn.Tanh())
      right:add(nn.MulConstant(0.25))
      
      local concat = nn.ConcatTable()
      concat:add(left)
      concat:add(right)
      MODEL_G = nn.Sequential()
      MODEL_G:add(concat)
      MODEL_G:add(nn.CAddTable())
      MODEL_G:add(nn.View(IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3]))
  else
      MODEL_G = nn.Sequential()
      MODEL_G:add(nn.Linear(OPT.noiseDim, 2048))
      MODEL_G:add(nn.PReLU())
      MODEL_G:add(nn.Linear(2048, INPUT_SZ))
      MODEL_G:add(nn.Sigmoid())
      MODEL_G:add(nn.View(IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3]))
  end
  
  --initializeWeights(MODEL_D)
  --initializeWeights(MODEL_G)
  MODEL_D = require('weight-init')(MODEL_D, 'heuristic')
  MODEL_G = require('weight-init')(MODEL_G, 'heuristic')
end

if OPT.autoencoder == "" then
    print("[INFO] No Autoencoder network specified, will not use an autoencoder.")
else
    print("<trainer> Loading autoencoder")
    local tmp = torch.load(OPT.autoencoder)
    local savedAutoencoder = tmp.AE

    MODEL_AE = nn.Sequential()
    MODEL_AE:add(nn.Linear(OPT.noiseDim, 256))
    MODEL_AE:add(nn.ReLU())
    MODEL_AE:add(nn.Linear(256, INPUT_SZ))
    MODEL_AE:add(nn.Sigmoid())
    MODEL_AE:add(nn.View(OPT.geometry[1], OPT.geometry[2], OPT.geometry[3]))

    local mapping = {{1,6+1}, {3,6+3}, {5,6+5}}
    for i=1, #mapping do
        print(string.format("Loading AE layer %d from autoencoder layer %d ...", mapping[i][1], mapping[i][2]))
        local mapTo = mapping[i][1]
        local mapFrom = mapping[i][2]
        if MODEL_AE.modules[mapTo].weight and savedAutoencoder.modules[mapFrom].weight then
            MODEL_AE.modules[mapTo].weight = savedAutoencoder.modules[mapFrom].weight
        end
        if MODEL_AE.modules[mapTo].bias and savedAutoencoder.modules[mapFrom].bias then
            MODEL_AE.modules[mapTo].bias = savedAutoencoder.modules[mapFrom].bias
        end
    end
end

if OPT.gpu then
    print("Copying model to gpu...")
    MODEL_D = NN_UTILS.activateCuda(MODEL_D)
    MODEL_G = NN_UTILS.activateCuda(MODEL_G)
end

-- loss function: negative log-likelihood
CRITERION = nn.BCECriterion()

-- retrieve parameters and gradients
PARAMETERS_D, GRAD_PARAMETERS_D = MODEL_D:getParameters()
PARAMETERS_G, GRAD_PARAMETERS_G = MODEL_G:getParameters()

-- print networks
print("Autoencoder network:")
print(MODEL_AE)
print('Discriminator network:')
print(MODEL_D)
print('Generator network:')
print(MODEL_G)
print('Validator network:')
print(MODEL_V)

-- this matrix records the current confusion across classes
CONFUSION = optim.ConfusionMatrix(CLASSES)

-- log results to files
--TRAIN_LOGGER = optim.Logger(paths.concat(OPT.save, 'train.log'))
--TEST_LOGGER = optim.Logger(paths.concat(OPT.save, 'test.log'))

-- Set optimizer state if it hasn't been loaded from file
if OPTSTATE == nil or OPT.rebuildOptstate == 1 then
    OPTSTATE = {
        adagrad = {
            D = { learningRate = 1e-3 },
            G = { learningRate = 1e-3 * 3 }
        },
        adam = {
            --D = {learningRate = 0.0005},
            --G = {learningRate = 0.0010}
            D = {},
            G = {}
        },
        rmsprop = {D = {}, G = {}},
        sgd = {
            D = {learningRate = OPT.SGD_lr, momentum = OPT.SGD_momentum},
            G = {learningRate = OPT.SGD_lr, momentum = OPT.SGD_momentum}
        }
    }
end

if EPOCH == nil then
    EPOCH = 1
end
PLOT_DATA = {}
VIS_NOISE_INPUTS = NN_UTILS.createNoiseInputs(100)

-- training loop
while true do
    if EPOCH == 1 or EPOCH % 1 == 0 then
        print('Loading new training data...')
        TRAIN_DATA = DATASET.loadRandomImages(OPT.N_epoch * 1)
    end

    if OPT.plot then
        NN_UTILS.visualizeProgress(VIS_NOISE_INPUTS)
        --TRAIN_LOGGER:style{['% mean class accuracy (train set)'] = '-'}
        --TEST_LOGGER:style{['% mean class accuracy (test set)'] = '-'}
    end

    ADVERSARIAL.train(TRAIN_DATA, OPT.D_maxAcc, math.max(20, math.min(1000/OPT.batchSize, 250)))
    
    
    
    --OPTSTATE.adam.G.learningRate = OPTSTATE.adam.G.learningRate * 0.99
end
