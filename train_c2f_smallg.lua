require 'torch'
require 'optim'
require 'image'
require 'pl'
require 'paths'
--image_utils = require 'utils.image'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
NN_UTILS = require 'utils.nn_utils'

-- We use here different files compared to train.lua
ADVERSARIAL = require 'adversarial_c2f_smallg'
DATASET = require 'dataset_c2f'
MODELS = require 'models_c2f'

----------------------------------------------------------------------
-- parse command-line options
OPT = lapp[[
  --save             (default "logs")       subdirectory to save logs
  --saveFreq         (default 10)           save every saveFreq epochs
  --network          (default "")           reload pretrained network
  --noplot                                  plot while training
  --D_sgd_lr         (default 0.02)         D SGD learning rate
  --G_sgd_lr         (default 0.02)         G SGD learning rate
  --D_sgd_momentum   (default 0)            D SGD momentum
  --G_sgd_momentum   (default 0)            G SGD momentum
  --batchSize        (default 32)           batch size
  --N_epoch          (default 1000)         Number of examples per epoch (-1 means all)
  --G_L1             (default 0)            L1 penalty on the weights of G
  --G_L2             (default 0e-6)         L2 penalty on the weights of G
  --D_L1             (default 1e-7)         L1 penalty on the weights of D
  --D_L2             (default 0e-6)         L2 penalty on the weights of D
  --D_iterations     (default 1)            number of iterations to optimize D for
  --G_iterations     (default 1)            number of iterations to optimize G for
  --D_clamp          (default 1)            Clamp threshold for D's gradient (+/- N)
  --G_clamp          (default 5)            Clamp threshold for G's gradient (+/- N)
  --D_optmethod      (default "adam")       adam|adagrad|sgd
  --G_optmethod      (default "adam")       adam|adagrad|sgd
  --threads          (default 4)            number of threads
  --gpu              (default 0)            gpu to run on (default cpu)
  --noiseDim         (default 100)          dimensionality of noise vector
  --window           (default 3)            window id of sample image
  --coarseSize       (default 16)           coarse scale
  --fineSize         (default 32)           fine scale
  --parzenImages     (default 250)          Number of images in approximate parzen (nsamples)
  --parzenNeighbours (default 32)           Number of generated refinements per image in approximate parzen (will choose min distance)
  --grayscale                               grayscale mode on/off
  --seed             (default 1)            seed for the RNG
  --aws                                     run in AWS mode
]]

if OPT.fineSize ~= 22 and OPT.fineSize ~= 32 then
    print("[Warning] Models are currently only optimized for fine sizes of 22 and 32.")
end

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
    IMG_DIMENSIONS = {1, OPT.fineSize, OPT.fineSize}
    COND_DIM = {1, OPT.fineSize, OPT.fineSize}
else
    IMG_DIMENSIONS = {3, OPT.fineSize, OPT.fineSize}
    COND_DIM = {3, OPT.fineSize, OPT.fineSize}
end
-- size in values/pixels per input image (channels*height*width)
INPUT_SZ = IMG_DIMENSIONS[1] * IMG_DIMENSIONS[2] * IMG_DIMENSIONS[3]
NOISE_DIM = {1, OPT.fineSize, OPT.fineSize}

----------------------------------------------------------------------
-- get/create dataset
----------------------------------------------------------------------
DATASET.nbChannels = IMG_DIMENSIONS[1]
DATASET.setFileExtension("jpg")
DATASET.setCoarseScale(OPT.coarseSize)
DATASET.setFineScale(OPT.fineSize)

if OPT.aws then
    DATASET.setDirs({"/mnt/datasets/out_aug_64x64"})
else
    DATASET.setDirs({"dataset/out_aug_64x64"})
end
----------------------------------------------------------------------

-- run on gpu if chosen
-- We have to load all kinds of libraries here, otherwise we risk crashes when loading
-- saved networks afterwards
print("<trainer> starting gpu support...")
require 'nn'
require 'cutorch'
require 'cunn'
require 'LeakyReLU'
require 'dpnn'
require 'layers.cudnnSpatialConvolutionUpsample'
require 'stn'
if OPT.gpu then
    cutorch.setDevice(OPT.gpu + 1)
    cutorch.manualSeed(OPT.seed)
    print(string.format("<trainer> using gpu device %d", OPT.gpu))
end
torch.setdefaulttensortype('torch.FloatTensor')

if OPT.network ~= "" then
    -- Load an older network
    print(string.format("<trainer> reloading previously trained network: %s", OPT.network))
    local tmp = torch.load(OPT.network)
    MODEL_D = tmp.D
    MODEL_G = tmp.G
    --OPTSTATE = tmp.optstate
    EPOCH = tmp.epoch
    PLOT_DATA = tmp.plot_data
    NORMALIZE_MEAN = tmp.normalize_mean
    NORMALIZE_STD = tmp.normalize_std
    
    if OPT.gpu == false then
        MODEL_D:float()
        MODEL_G:float()
    end
else
    -- Create new Ds and Gs
    MODEL_D = MODELS.create_D(IMG_DIMENSIONS, OPT.gpu ~= false)
    MODEL_G = MODELS.create_G(IMG_DIMENSIONS, OPT.noiseDim, OPT.gpu ~= false)
end

-- Load a 16x16 base G
-- That base G will be used to generate images that are then used during training (16 to 22 net)
-- or are fed into a trained 16 to 22 net (for training of 22 to 32)
local tmp = torch.load(paths.concat(OPT.save, "adversarial.net"))
MODEL_G_BASE = tmp.G

-- Load a G of a 16 to 22 net if currently training 22 to 32.
-- That G will be used to produce images that are used during training.
if OPT.coarseSize == 22 then
    tmp = torch.load(paths.concat(OPT.save, "adversarial_c2f_16_to_22.net"))
    MODEL_G_C2F22 = tmp.G
end

-- Generate images from all Gs before the currently one.
-- For a 16 to 22 net that is the base 16x16 G.
-- For a 22 to 32 net that is the base 16x16 G and then the 16 to 22 net.
-- @param nbImages Number of images to generate.
-- @returns Tensor of shape (nbImages, channels, height, width)
function sampleFromSmallerGs(nbImages)
    local noise = torch.Tensor(nbImages, OPT.noiseDim)
    noise:uniform(-1.0, 1.0)
    local images = MODEL_G_BASE:forward(noise)
    NN_UTILS.normalize(images)
    local imagesCoarse = torch.Tensor(nbImages, IMG_DIMENSIONS[1], 22, 22)
    for i=1,nbImages do
        imagesCoarse[i] = image.scale(images[i], 22, 22)
    end
    
    if not MODEL_G_C2F22 then
        if math.random() > 0.95 then
            disp.image(imagesCoarse, {win=OPT.window+5, width=5*IMG_DIMENSIONS[3], title=string.format("Sampled from earlier Gs", OPT.save, EPOCH)})
        end
        return imagesCoarse
    else
        local imagesFine = torch.Tensor(nbImages, IMG_DIMENSIONS[1], 22, 22)
        noise = torch.Tensor(nbImages, 1, 22, 22)
        noise:uniform(-1.0, 1.0)
        local imagesDiff = MODEL_G_C2F22:forward({noise, imagesCoarse})
        for i=1,nbImages do
            imagesFine[i] = torch.add(imagesCoarse[i], imagesDiff[i])
            imagesFine[i] = torch.clamp(imagesFine[i], -1.0, 1.0)
        end
        
        local imagesUpscaled = torch.Tensor(nbImages, IMG_DIMENSIONS[1], 32, 32)
        for i=1,nbImages do
            imagesUpscaled[i] = image.scale(imagesFine[i], 32, 32)
        end
        
        if math.random() > 0.95 then
            disp.image(imagesCoarse, {win=OPT.window+5, width=5*IMG_DIMENSIONS[3], title=string.format("Sampled from earlier Gs", OPT.save, EPOCH)})
        end
        return imagesUpscaled
    end
end

-- Copy to GPU
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

-- Show models
print("Model D:")
print(MODEL_D)
print("Model G:")
print(MODEL_G)

-- count free parameters in D/G
local nparams = 0
local dModules = MODEL_D:listModules()
for i=1,#dModules do
  if dModules[i].weight ~= nil then
    nparams = nparams + dModules[i].weight:nElement()
  end
end
print('\nNumber of free parameters in D: ' .. nparams)


local nparams = 0
local gModules = MODEL_G:listModules()
for i=1,#gModules do
  if gModules[i].weight ~= nil then
    nparams = nparams + gModules[i].weight:nElement()
  end
end
print('Number of free parameters in G: ' .. nparams .. '\n')

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

-- Get examples to plot.
-- Returns a list of the pattern
--  [i] Coarse/blurry image,
--  [i+1] Fine image (Coarse + true diff)
--  [i+2] Coarse + diff generated by G
--  [i+3] True diff
--  [i+4] Diff generated by G
-- @param ds Dataset as list of examples that have attributes .coarse .fine and .diff
-- @param N Number of samples to prepare
function getPlottingSamples(ds, N)
    local N = N or 8
    local noiseInputs = torch.Tensor(N, NOISE_DIM[1], NOISE_DIM[2], NOISE_DIM[3])
    local condInputs = torch.Tensor(N, COND_DIM[1], COND_DIM[2], COND_DIM[3])
    local gt_diff = torch.Tensor(N, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
    local gt = torch.Tensor(N, IMG_DIMENSIONS[1], OPT.fineSize, OPT.fineSize)

    -- Generate samples
    noiseInputs:uniform(-1, 1)
    for n = 1,N do
        local rand = math.random(ds:size())
        local example = ds[rand]
        condInputs[n] = example.coarse:clone()
        gt[n] = example.fine:clone()
        gt_diff[n] = example.diff:clone()
    end
    local samples = MODEL_G:forward({noiseInputs, condInputs})

    local to_plot = {}
    for i=1,N do
        local refined = torch.add(condInputs[i]:float(), samples[i]:float())
        to_plot[#to_plot+1] = condInputs[i]:float()
        to_plot[#to_plot+1] = gt[i]:float()
        to_plot[#to_plot+1] = refined
        to_plot[#to_plot+1] = gt_diff[i]:float()
        to_plot[#to_plot+1] = samples[i]:float()
    end
    return to_plot
end

-- Get examples from previous Gs to plot.
-- Shows the upscaled images generated by the chain of previous Gs,
-- as well as their diffs and refined images generated by the currently trained G.
-- @param N Number of images to generate.
-- @returns Tensor of shape (N, channels, height, width)
function getPlottingSamplesSmallerGs(N)
    local N = N or 8
    local images = sampleFromSmallerGs(N)
    local noiseInputs = torch.Tensor(N, NOISE_DIM[1], NOISE_DIM[2], NOISE_DIM[3])
    noiseInputs:uniform(-1, 1)
    
    local diffs = MODEL_G:forward({noiseInputs, images})
    
    local to_plot = {}
    for i=1,N do
        local refined = torch.add(images[i], diffs[i])
        to_plot[#to_plot+1] = images[i]
        to_plot[#to_plot+1] = refined
        to_plot[#to_plot+1] = diffs[i]
    end
    return to_plot
end

-- Save the current models G and D to a file.
-- @param filename The path to the file
function saveAs(filename)
    os.execute(string.format("mkdir -p %s", sys.dirname(filename)))
    if paths.filep(filename) then
      os.execute(string.format("mv %s %s.old", filename, filename))
    end
    print(string.format("<trainer> saving network to %s", filename))
    torch.save(filename, {D = MODEL_D, G = MODEL_G, opt = OPT, plot_data = PLOT_DATA, epoch = EPOCH+1, normalize_mean=NORMALIZE_MEAN, normalize_std=NORMALIZE_STD})
end

if EPOCH == nil then EPOCH = 1 end

-- Normalize images, currently only to -1.0 to +1.0 range
if NORMALIZE_MEAN == nil then
    TRAIN_DATA = DATASET.loadRandomImages(10000, OPT.parzenImages)
    NORMALIZE_MEAN, NORMALIZE_STD = TRAIN_DATA.normalize()
end
VAL_DATA = DATASET.loadImages(0, OPT.parzenImages)
VAL_DATA.normalize(NORMALIZE_MEAN, NORMALIZE_STD)

if PLOT_DATA == nil then
    -- No plotting data loaded, initialize to empty values
    PLOT_DATA = {}
    LAST_PARZEN_APPROX = nil
    BEST_PARZEN_APPROX = nil
else
    -- Process loaded plotting data with (epoch, parzen approximation) values.
    LAST_PARZEN_APPROX = PLOT_DATA[#PLOT_DATA][2]

    -- we start at epoch 20,
    -- that way we avoid getting stuck at a low value from an early epoch (where low means that G
    -- simply didnt change anything about the image)
    BEST_PARZEN_APPROX = LAST_PARZEN_APPROX
    for i=20,#PLOT_DATA do
        if PLOT_DATA[i][2] > BEST_PARZEN_APPROX then
            BEST_PARZEN_APPROX = PLOT_DATA[i][2]
        end
    end
end

-- training loop
while true do
    -- load new data
    print('Loading new training data...')
    TRAIN_DATA = DATASET.loadRandomImages(OPT.N_epoch, VAL_DATA:size() + 1)
    TRAIN_DATA.normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    
    -- plot
    if not OPT.noplot then
        -- Show images and their refinements for the validation and training set
        -- as well as images from previous Gs (upscaled) and their refinements
        local to_plot_val = getPlottingSamples(VAL_DATA, 20)
        local to_plot_train = getPlottingSamples(TRAIN_DATA, 20)
        local to_plot_smaller = getPlottingSamplesSmallerGs(24)
        disp.image(to_plot_val, {win=OPT.window, width=2*10*IMG_DIMENSIONS[3], title=string.format("[VAL] Coarse, GT, G img, GT diff, G diff (%s epoch %d)", OPT.save, EPOCH-1)})
        disp.image(to_plot_train, {win=OPT.window+1, width=2*10*IMG_DIMENSIONS[3], title=string.format("[TRAIN] Coarse, GT, G img, GT diff, G diff (%s epoch %d)", OPT.save, EPOCH-1)})
        disp.image(to_plot_smaller, {win=OPT.window+6, width=2*10*IMG_DIMENSIONS[3], title=string.format("[SMALLER Gs] Coarse, Refined, Diff (%s epoch %d)", OPT.save, EPOCH-1)})
        
        -- Show what images and refinements looked like for the net with the best parzen approximation.
        -- Usually they didnt look good.
        if BEST_PARZEN_APPROX ~= nil and LAST_PARZEN_APPROX == BEST_PARZEN_APPROX then
            disp.image(to_plot_val, {win=OPT.window+3, width=2*10*IMG_DIMENSIONS[3], title=string.format("[Best] parzen dist=%.3f (%s at epoch %d)", BEST_PARZEN_APPROX, OPT.save, EPOCH-1)})
        end
        
        -- Show a plot of the parzen approximations.
        if #PLOT_DATA > 0 then
            local best = BEST_PARZEN_APPROX
            if best == nil then
                best = -1
            end
            disp.plot(PLOT_DATA, {win=OPT.window+4, labels={'epoch', 'parzen'}, title=string.format('Parzen Approximation Rating (min=%.3f) (lower is better)', best)})
        end
    end
    
    -- train
    ADVERSARIAL.train(TRAIN_DATA)
    
    -- approximate parzen evaluation
    local dist = ADVERSARIAL.approxParzen(VAL_DATA, OPT.parzenNeighbours)
    LAST_PARZEN_APPROX = dist:mean()
    table.insert(PLOT_DATA, {EPOCH, LAST_PARZEN_APPROX})
    
    -- if best value, then save network and log this value
    if BEST_PARZEN_APPROX == nil or LAST_PARZEN_APPROX < BEST_PARZEN_APPROX then
        if EPOCH > 20 then
            BEST_PARZEN_APPROX = LAST_PARZEN_APPROX
            local filename = paths.concat(OPT.save, string.format('adversarial_c2f_%d_to_%d.bestnet', OPT.coarseSize, OPT.fineSize))
            saveAs(filename)
        end
    end
    
    -- save every N epochs
    if EPOCH % OPT.saveFreq == 0 then
        local filename = paths.concat(OPT.save, string.format('adversarial_c2f_%d_to_%d.net', OPT.coarseSize, OPT.fineSize))
        saveAs(filename)
        
        -- Show images and their refinements from the time of the last save
        if not OPT.noplot then
            local to_plot_save = getPlottingSamples(VAL_DATA, 20)
            disp.image(to_plot_save, {win=OPT.window+2, width=2*10*IMG_DIMENSIONS[3], title=string.format("[LAST SAVE VAL] Coarse, GT, G img, GT diff, G diff (%s epoch %d)", OPT.save, EPOCH)})
        end
    end
    
    EPOCH = EPOCH + 1
end
