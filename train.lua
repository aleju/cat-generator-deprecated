require 'torch'
require 'image'
require 'pl' -- this is somehow responsible for lapp working in qlua mode
require 'paths'
ok, DISP = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
ADVERSARIAL = require 'adversarial'
DATASET = require 'dataset'
NN_UTILS = require 'utils.nn_utils'
MODELS = require 'models'


----------------------------------------------------------------------
-- parse command-line options
OPT = lapp[[
  -s,--save          (default "logs")       subdirectory to save logs
  --saveFreq         (default 30)           save every saveFreq epochs
  -n,--network       (default "")           reload pretrained network
  --V_dir            (default "logs")       Directory where V networks are saved
  --G_pretrained_dir (default "logs")
  -p,--plot                                 plot while training
  --D_sgd_lr         (default 0.02)         D SGD learning rate
  --G_sgd_lr         (default 0.02)         G SGD learning rate
  --D_sgd_momentum   (default 0)            D SGD momentum
  --G_sgd_momentum   (default 0)            G SGD momentum
  -b,--batchSize     (default 16)           batch size
  --N_epoch          (default 1000)         Number of examples per epoch (-1 means all)
  --G_L1             (default 0)            L1 penalty on the weights of G
  --G_L2             (default 0e-6)         L2 penalty on the weights of G
  --D_L1             (default 1e-7)         L1 penalty on the weights of D
  --D_L2             (default 0e-6)         L2 penalty on the weights of D
  --D_iterations     (default 1)            number of iterations to optimize D for
  --G_iterations     (default 1)            number of iterations to optimize G for
  --D_maxAcc         (default 1.01)         Deactivate learning of D while above this threshold
  --D_clamp          (default 1)            Clamp threshold for D's gradient (+/- N)
  --G_clamp          (default 5)            Clamp threshold for G's gradient (+/- N)
  --D_optmethod      (default "adam")       adam|adagrad
  --G_optmethod      (default "adam")       adam|adagrad
  -t,--threads       (default 8)            number of threads
  -g,--gpu           (default -1)           gpu to run on (default cpu)
  -d,--noiseDim      (default 100)          dimensionality of noise vector
  -w, --window       (default 3)            window id of sample image
  --scale            (default 32)           scale of images to train on
  --grayscale                               grayscale mode on/off
  --autoencoder      (default "")           path to autoencoder to load weights from
  --rebuildOptstate  (default 0)            whether to force a rebuild of the optimizer state
  --seed             (default 1)            seed for the RNG
  --weightsVisFreq   (default 0)            how often to update the weight visualization (requires starting with qlua, 0 is off)
  --aws                                     run in AWS mode
]]

START_TIME = os.time()

if OPT.gpu < 0 or OPT.gpu > 3 then OPT.gpu = false end
print(OPT)

-- fix seed
math.randomseed(OPT.seed)
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
    DATASET.setDirs({"/media/aj/ssd2a/ml/datasets/10k_cats/out_faces_64x64", "/media/aj/ssd2a/ml/datasets/flickr-cats/images_faces_aug"})
end
----------------------------------------------------------------------

-- run on gpu if chosen
print("<trainer> starting gpu support...")
require 'nn'
require 'cutorch'
require 'cunn'
if OPT.gpu then
    cutorch.setDevice(OPT.gpu + 1)
    cutorch.manualSeed(OPT.seed)
    print(string.format("<trainer> using gpu device %d", OPT.gpu))
end
torch.setdefaulttensortype('torch.FloatTensor')


function main()
    ----------------------------------------------------------------------
    -- Load / Define network
    ----------------------------------------------------------------------
    local filename = paths.concat(OPT.V_dir, string.format('v_%dx%dx%d.net', IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3]))
    local tmp = torch.load(filename)
    MODEL_V = tmp.V
    MODEL_V:float()
    MODEL_V:evaluate() -- deactivate dropout

    -- load previous networks (D and G)
    -- or initialize them new
    if OPT.network ~= "" then
        print(string.format("<trainer> reloading previously trained network: %s", OPT.network))
        local tmp = torch.load(OPT.network)
        MODEL_D = tmp.D
        MODEL_G = tmp.G
        OPTSTATE = tmp.optstate
        EPOCH = tmp.epoch
        
        if OPT.gpu ~= false then
            MODEL_D:float()
            MODEL_G:float()
        end
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
  
        --[[
        local activation = nn.PReLU
        local branch_conv = nn.Sequential()
      
        branch_conv:add(nn.Dropout(0.1))
      
        branch_conv:add(nn.SpatialConvolution(IMG_DIMENSIONS[1], 64, 3, 3, 1, 1, (3-1)/2))
        branch_conv:add(activation())
        --branch_conv:add(nn.Dropout())
        branch_conv:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
        branch_conv:add(activation())
        --branch_conv:add(nn.SpatialMaxPooling(2, 2))
        --branch_conv:add(nn.SpatialDropout())
        branch_conv:add(nn.Dropout())
      
        branch_conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2))
        branch_conv:add(activation())
        branch_conv:add(nn.SpatialMaxPooling(2, 2))
        branch_conv:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
        branch_conv:add(activation())
        branch_conv:add(nn.Dropout())
        --branch_conv:add(nn.SpatialMaxPooling(2, 2))
        --branch_conv:add(nn.Dropout())
        branch_conv:add(nn.View(128, 16*16))
        local parallel = nn.Parallel(2, 2)
        for i=1,128 do
            parallel:add(nn.Linear(16*16, 128))
        end
        branch_conv:add(parallel)
        branch_conv:add(activation())
        branch_conv:add(nn.Dropout())
      
        branch_conv:add(nn.Linear(128*128, 1))
        branch_conv:add(nn.Sigmoid())
  
        MODEL_D = branch_conv
        --]]
  
        MODEL_D = MODELS.create_D(IMG_DIMENSIONS)
      
        --------------
        -- G
        --------------
        local g_pt_filename = paths.concat(OPT.G_pretrained_dir, string.format('g_pretrained_%dx%dx%d_nd%d.net', IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3], OPT.noiseDim))
        if paths.filep(g_pt_filename) then
            print("<trainer> loading pretrained G...")
            local tmp = torch.load(g_pt_filename)
            MODEL_G = tmp.G
            MODEL_G:float()
        else
            print("<trainer> Note: Did not find pretrained G")
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
                MODEL_G = MODELS.create_G(IMG_DIMENSIONS, OPT.noiseDim)
            end
        end
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

    -- this matrix records the current confusion across classes
    CONFUSION = optim.ConfusionMatrix(CLASSES)

    -- log results to files
    --TRAIN_LOGGER = optim.Logger(paths.concat(OPT.save, 'train.log'))
    --TEST_LOGGER = optim.Logger(paths.concat(OPT.save, 'test.log'))

    -- Set optimizer state
    if OPTSTATE == nil or OPT.rebuildOptstate == 1 then
        OPTSTATE = {
            adagrad = {
                D = { learningRate = 1e-3 },
                G = { learningRate = 1e-3 * 3 }
            },
            adam = {
                D = {},
                G = {}
            },
            rmsprop = {D = {}, G = {}},
            sgd = {
                D = {learningRate = OPT.D_sgd_lr, momentum = OPT.D_sgd_momentum},
                G = {learningRate = OPT.G_sgd_lr, momentum = OPT.G_sgd_momentum}
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
        print('Loading new training data...')
        TRAIN_DATA = DATASET.loadRandomImages(OPT.N_epoch)

        if OPT.plot then
            NN_UTILS.visualizeProgress(VIS_NOISE_INPUTS)
            --TRAIN_LOGGER:style{['% mean class accuracy (train set)'] = '-'}
            --TEST_LOGGER:style{['% mean class accuracy (test set)'] = '-'}
        end

        -- Train D and G
        -- ... but train D only while having an accuracy below OPT.D_maxAcc
        -- ... over the last math.max(20, math.min(1000/OPT.batchSize, 250)) batches
        ADVERSARIAL.train(TRAIN_DATA, OPT.D_maxAcc, math.max(20, math.min(1000/OPT.batchSize, 250)))
    end
end

main()
