require 'torch'
require 'image'
require 'paths'
require 'pl' -- this is somehow responsible for lapp working in qlua mode
require 'optim'
ok, DISP = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
DATASET = require 'dataset'
NN_UTILS = require 'utils.nn_utils'
MODELS = require 'models'

OPT = lapp[[
    --save          (default "logs")
    --batchSize     (default 128)
    --noplot                            Whether to not plot
    --window        (default 23)
    --seed          (default 1)
    --aws                               run in AWS mode
    --saveFreq      (default 50)        
    --gpu           (default 0)
    --threads       (default 8)         number of threads
    --grayscale                         activate grayscale mode
    --scale         (default 32)
    --G_clamp       (default 5)
    --G_L1          (default 0)
    --G_L2          (default 0)
    --N_epoch       (default 10000)
    --noiseDim      (default 100)
]]

if OPT.gpu < 0 or OPT.gpu > 3 then OPT.gpu = false end
print(OPT)

math.randomseed(OPT.seed)
torch.manualSeed(OPT.seed)
torch.setnumthreads(OPT.threads)

if OPT.grayscale then
    IMG_DIMENSIONS = {1, OPT.scale, OPT.scale}
else
    IMG_DIMENSIONS = {3, OPT.scale, OPT.scale}
end

INPUT_SZ = IMG_DIMENSIONS[1] * IMG_DIMENSIONS[2] * IMG_DIMENSIONS[3]

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
torch.setdefaulttensortype('torch.FloatTensor')


function main()
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
    
    G_AUTOENCODER = MODELS.create_G_autoencoder(IMG_DIMENSIONS, OPT.noiseDim)
    
    if OPT.gpu then
        G_AUTOENCODER = NN_UTILS.activateCuda(G_AUTOENCODER)
    end
    
    print("G autoencoder:")
    print(G_AUTOENCODER)
    
    CRITERION = nn.MSECriterion()
    PARAMETERS_G_AUTOENCODER, GRAD_PARAMETERS_G_AUTOENCODER = G_AUTOENCODER:getParameters()
    OPTSTATE = {adam={}}
    
    EPOCH = 1
    while true do
        TRAIN_DATA = DATASET.loadRandomImages(OPT.N_epoch)
        print(string.format("<trainer> Epoch %d", EPOCH))
        epoch()
        visualizeProgress()
    end
end



function epoch()
    local startTime = sys.clock()
    local batchIdx = 0
    local trained = 0
    while trained < OPT.N_epoch do
        local thisBatchSize = math.min(OPT.batchSize, OPT.N_epoch - trained)
        local inputs = torch.Tensor(thisBatchSize, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
        local targets = torch.Tensor(thisBatchSize, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
        
        for i=1,thisBatchSize do
            inputs[i] = TRAIN_DATA[i]:clone()
            targets[i] = TRAIN_DATA[i]:clone()
        end
        
        local fevalG = function(x)
            collectgarbage()
            if x ~= PARAMETERS_G_AUTOENCODER then PARAMETERS_G_AUTOENCODER:copy(x) end
            GRAD_PARAMETERS_G_AUTOENCODER:zero()

            --  forward pass
            local outputs = G_AUTOENCODER:forward(inputs)
            local f = CRITERION:forward(outputs, targets)

            -- backward pass 
            local df_do = CRITERION:backward(outputs, targets)
            G_AUTOENCODER:backward(inputs, df_do)

            -- penalties (L1 and L2):
            if OPT.G_L1 ~= 0 or OPT.G_L2 ~= 0 then
                -- Loss:
                f = f + OPT.G_L1 * torch.norm(PARAMETERS_G_AUTOENCODER, 1)
                f = f + OPT.G_L2 * torch.norm(PARAMETERS_G_AUTOENCODER, 2)^2/2
                -- Gradients:
                GRAD_PARAMETERS_G_AUTOENCODER:add(torch.sign(PARAMETERS_G_AUTOENCODER):mul(OPT.G_L1) + PARAMETERS_G_AUTOENCODER:clone():mul(OPT.G_L2) )
            end

            -- Clamp G's gradients
            if OPT.G_clamp ~= 0 then
                GRAD_PARAMETERS_G_AUTOENCODER:clamp((-1)*OPT.G_clamp, OPT.G_clamp)
            end
            
            return f,GRAD_PARAMETERS_G_AUTOENCODER
        end
        
        optim.adam(fevalG, PARAMETERS_G_AUTOENCODER, OPTSTATE.adam)
        
        trained = trained + thisBatchSize
        batchIdx = batchIdx + 1
        
        xlua.progress(trained, OPT.N_epoch)
    end
    
    local epochTime = sys.clock() - startTime
    print(string.format("<trainer> time required for this epoch = %d s", epochTime))
    print(string.format("<trainer> time to learn 1 sample = %f ms", 1000 * epochTime/OPT.N_epoch))
    print(string.format("<trainer> loss: %.4f", CRITERION.output))
    
    if EPOCH % OPT.saveFreq == 0 then
        local filename = paths.concat(OPT.save, string.format('g_pretrained_%dx%dx%d_nd%d.net', IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3], OPT.noiseDim))
        os.execute(string.format("mkdir -p %s", sys.dirname(filename)))
        --if paths.filep(filename) then
        --    os.execute(string.format("mv %s %s.old", filename, filename))
        --end
        print(string.format("<trainer> saving network to %s", filename))
        
        -- apparently something in the OPTSTATE is a CudaTensor, so saving it and then loading
        -- in CPU mode would cause an error
        -- :get(2) means here "get the decoder part of the autoencoder, dont save the encoder"
        torch.save(filename, {G=NN_UTILS.deactivateCuda(G_AUTOENCODER):get(2), opt=OPT, EPOCH=EPOCH+1}) --, optstate=OPTSTATE
    end
    
    EPOCH = EPOCH + 1
end

function visualizeProgress()
    -- deactivate dropout
    G_AUTOENCODER:evaluate()
    
    if PLOT_DATA == nil then PLOT_DATA = {} end
    
    local imagesReal = DATASET.loadRandomImages(100)
    local imagesRealTensor = torch.Tensor(imagesReal:size(), IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
    for i=1,imagesReal:size() do imagesRealTensor[i] = imagesReal[i] end
    
    local imagesAfterG = G_AUTOENCODER:forward(imagesRealTensor)
    table.insert(PLOT_DATA, {EPOCH, CRITERION.output})
    
    DISP.image(imagesRealTensor, {win=OPT.window+0, width=IMG_DIMENSIONS[3]*15, title="Original images (before Autoencoder) (EPOCH " .. EPOCH .. ")"})
    DISP.image(imagesAfterG, {win=OPT.window+1, width=IMG_DIMENSIONS[3]*15, title="Images after autoencoder G (EPOCH " .. EPOCH .. ")"})
    DISP.plot(PLOT_DATA, {win=OPT.window+2, labels={'epoch', 'G Loss'}, title='G Loss'})
    
    -- reactivate dropout
    G_AUTOENCODER:training()
end

main()
