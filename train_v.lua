require 'torch'
require 'image'
require 'paths'
ok, DISP = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
DATASET = require 'dataset'

OPT = lapp[[
    --save          (default "logs")
    --batchSize     (default 128)
    --noplot                            Whether to not plot
    --window        (default 13)
    --seed          (default 1)
    --aws                               run in AWS mode
    --saveFreq      (default 50)        
    --gpu           (default 0)
    --threads       (default 8)         number of threads
    --grayscale                         activate grayscale mode
    --scale         (default 32)
    --V_clamp       (default 5)
    --V_L1          (default 0)
    --V_L2          (default 0)
    --N_epoch       (default 1000)
]]

if OPT.gpu < 0 or OPT.gpu > 3 then OPT.gpu = false end
print(OPT)

torch.manualSeed(OPT.seed)
torch.setnumthreads(OPT.threads)

Y_FAKE = 0
Y_REAL = 1
CLASSES = {"0", "1"}

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
        --DATASET.setDirs({"/media/aj/ssd2a/ml/datasets/10k_cats/out_faces_64x64", "/media/aj/ssd2a/ml/datasets/flickr-cats/images_faces_aug"})
        DATASET.setDirs({"/media/aj/ssd2a/ml/datasets/10k_cats/out_faces_64x64"})
    end
    ----------------------------------------------------------------------
    
    local V = create_V()
    
    if OPT.gpu then
        V:cuda()
        
        local tmp = nn.Sequential()
        tmp:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
        tmp:add(V)
        tmp:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'))
        V = tmp
    end
    
    print("network V:")
    print(V)
    
    CRITERION = nn.BCECriterion()
    PARAMETERS_V, GRAD_PARAMETERS_V = V:getParameters()
    CONFUSION = optim.ConfusionMatrix(CLASSES)
    OPTSTATE = {adam:{}}
    
    EPOCH = 1
    while true do
        TRAIN_DATA = DATASET.loadRandomImages(OPT.N_epoch)
        epoch(V)
    end
end

function create_V()
    local model = nn.Sequential()
    local activation = nn.LeakyReLU
    
    model:add(nn.SpatialConvolution(IMG_DIMENSIONS[1], 32, 3, 3, 1, 1, (3-1)/2))
    model:add(activation())
    model:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, (3-1)/2))
    model:add(activation())
    model:add(nn.SpatialBatchNormalization(32))
    model:add(nn.SpatialDropout())
  
    model:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, (3-1)/2))
    model:add(activation())
    model:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))
    model:add(nn.SpatialBatchNormalization(64))
    model:add(nn.SpatialDropout())
    model:add(nn.View(64, 16 * 16))
  
    local parallel = nn.Parallel(2, 2)
    for i=1,128 do
        local lin = nn.Sequential()
        lin:add(nn.Linear(16*16, 128))
        lin:add(activation())
        lin:add(nn.BatchNormalization(128))
        lin:add(nn.Dropout())
        lin:add(nn.Linear(128, 8))
        lin:add(activation())
        parallel:add(lin)
    end
    model:add(parallel)
    model:add(nn.BatchNormalization(128*8))
  
    model:add(nn.Linear(128*8, 128))
    model:add(activation())
    model:add(nn.BatchNormalization(128))
    model:add(nn.Dropout())
  
    model:add(nn.Linear(128, 128))
    model:add(activation())
    model:add(nn.BatchNormalization(128))
    model:add(nn.Dropout())
  
    model:add(nn.Linear(128, 2))
    model:add(nn.Softmax())
  
    model = require('weight-init')(model, 'heuristic')
  
    return model
end

function epoch(model)
    local startTime = sys.clock()
    local batchIdx = 0
    local trained = 0
    while trained < OPT.N_epoch do
        local thisBatchSize = math.min(OPT.batchSize, N_epoch - t + 1)
        local inputs = torch.Tensor(thisBatchSize, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
        local targets = torch.Tensor(thisBatchSize, 2)
        
        local fevalV = function(x)
            collectgarbage()
            if x ~= PARAMETERS_V then PARAMETERS_V:copy(x) end
            GRAD_PARAMETERS_V:zero()

            --  forward pass
            local outputs = model:forward(inputs)
            local f = CRITERION:forward(outputs, targets)

            -- backward pass 
            local df_do = CRITERION:backward(outputs, targets)
            model:backward(inputs, df_do)

            -- penalties (L1 and L2):
            if OPT.V_L1 ~= 0 or OPT.V_L2 ~= 0 then
                -- Loss:
                f = f + OPT.V_L1 * torch.norm(PARAMETERS_V, 1)
                f = f + OPT.V_L2 * torch.norm(PARAMETERS_V, 2)^2/2
                -- Gradients:
                GRAD_PARAMETERS_V:add(torch.sign(PARAMETERS_V):mul(OPT.V_L1) + PARAMETERS_D:clone():mul(OPT.V_L2) )
            end

            -- update confusion (add 1 since targets are binary)
            for i = 1,thisBatchSize do
                local c
                if outputs[i][1] > 0.5 then c = 1 else c = 2 end
                CONFUSION:add(c, targets[i][1]+1)
            end

            -- Clamp V's gradients
            if OPT.V_clamp ~= 0 then
                GRAD_PARAMETERS_V:clamp((-1)*OPT.V_clamp, OPT.V_clamp)
            end
            
            return f,GRAD_PARAMETERS_V
        end
        
        --------------------------------------
        -- Collect Batch
        --------------------------------------
        -- Real data 
        local exampleIdx = 1
        local realDataSize = thisBatchSize / 2
        for i=1,thisBatchSize/2 do
            local randomIdx = math.random(TRAIN_DATA:size())
            inputs[exampleIdx] = TRAIN_DATA[randomIdx]:clone()
            targets[exampleIdx][Y_REAL+1] = 1
            targets[exampleIdx][Y_FAKE+1] = 0
            exampleIdx = exampleIdx + 1
        end

        -- Fake data
        local images = createSyntheticImages(thisBatchSize/2, model)
        for i = 1, realDataSize do
            inputs[exampleIdx] = images[i]:clone()
            targets[exampleIdx][Y_REAL+1] = 0
            targets[exampleIdx][Y_FAKE+1] = 1
            exampleIdx = exampleIdx + 1
        end
        
        optim.adam(fevalV, PARAMETERS_V, OPTSTATE.adam)
        
        
        trained = trained + thisBatchSize
        batchIdx = batchIdx + 1
        
        xlua.progress(trained, N_epoch)
    end
    
    local epochTime = sys.clock() - startTime
    print(string.format("<trainer> time required for this epoch = %d s", epochTime))
    print(string.format("<trainer> time to learn 1 sample = %f ms", 1000 * epochTime/N_epoch))
    print("Confusion of V:")
    print(CONFUSION)
    CONFUSION:zero()
    
    if EPOCH % OPT.saveFreq == 0 then
        local filename = paths.concat(OPT.save, 'v.net')
        os.execute(string.format("mkdir -p %s", sys.dirname(filename)))
        if paths.filep(filename) then
            os.execute(string.format("mv %s %s.old", filename, filename))
        end
        print(string.format("<trainer> saving network to %s", filename))
        
        torch.save(filename, {V=model, opt=OPT, optstate=OPTSTATE, epoch=EPOCH+1})
    end
    
    EPOCH = EPOCH + 1
end

function createSyntheticImages(N)
    local images = {}
    
    for i=1,N do
        local p = 0.5
        local img1 = TRAIN_DATA[math.random(TRAIN_DATA:size())]:clone()
        local img2 = TRAIN_DATA[math.random(TRAIN_DATA:size())]:clone()
        local image = torch.Tensor(img1:size())
        image:zero()
        
        for y=1,img1:size(2) do
            for x=1,img1:size(3) do
                -- loop over channel
                for c=1,img1:size(1) do
                    if math.random() > p then
                        image[c][y][x] = img1[c][y][x]
                    end
                    
                    if math.random() > 0.5 then
                        p = math.max(p - 0.05, 0)
                    else
                        p = math.min(p + 0.05, 1.0)
                    end
                end
            end
        end
        
        table.insert(images, image)
    end
    
    return images
end

main()
