require 'torch'
require 'image'
require 'paths'
require 'pl' -- this is somehow responsible for lapp working in qlua mode
require 'optim'
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

math.randomseed(OPT.seed)
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
        --DATASET.setDirs({"/media/aj/ssd2a/ml/datasets/10k_cats/out_faces_64x64", "/media/aj/ssd2a/ml/datasets/flickr-cats/images_faces_aug"})
        DATASET.setDirs({"/media/aj/ssd2a/ml/datasets/10k_cats/out_faces_64x64"})
    end
    ----------------------------------------------------------------------
    
    V = create_V()
    
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
    OPTSTATE = {adam={}}
    
    EPOCH = 1
    while true do
        TRAIN_DATA = DATASET.loadRandomImages(OPT.N_epoch)
        print(string.format("<trainer> Epoch %d", EPOCH))
        epoch(V)
        visualizeProgress()
    end
end

function create_V()
    local model = nn.Sequential()
    local activation = nn.LeakyReLU
    
    --[[
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
    for i=1,64 do
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
    model:add(nn.BatchNormalization(64*8))
  
    model:add(nn.Linear(64*8, 128))
    model:add(activation())
    model:add(nn.BatchNormalization(128))
    model:add(nn.Dropout())
  
    model:add(nn.Linear(128, 128))
    model:add(activation())
    model:add(nn.BatchNormalization(128))
    model:add(nn.Dropout())
  
    model:add(nn.Linear(128, 2))
    model:add(nn.SoftMax())
    --]]
  
    model:add(nn.SpatialConvolution(IMG_DIMENSIONS[1], 64, 3, 3, 1, 1, (3-1)/2))
    model:add(activation())
    model:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))
    model:add(nn.SpatialBatchNormalization(64))
    model:add(nn.Dropout())
  
    model:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2))
    model:add(activation())
    model:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(nn.SpatialDropout())
    model:add(nn.View(128 * 8 * 8))
    model:add(nn.BatchNormalization(128*8*8))
  
    model:add(nn.Linear(128*8*8, 1024))
    model:add(activation())
    model:add(nn.BatchNormalization(1024))
    model:add(nn.Dropout())
  
    model:add(nn.Linear(1024, 1024))
    model:add(activation())
    model:add(nn.BatchNormalization(1024))
    model:add(nn.Dropout())
  
    model:add(nn.Linear(1024, 2))
    model:add(nn.SoftMax())
  
    model = require('weight-init')(model, 'heuristic')
  
    return model
end

function epoch(model)
    local startTime = sys.clock()
    local batchIdx = 0
    local trained = 0
    while trained < OPT.N_epoch do
        local thisBatchSize = math.min(OPT.batchSize, OPT.N_epoch - trained)
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
                local predictedClass
                local realClass
                if outputs[i][1] > 0.5 then predictedClass = 0 else predictedClass = 1 end
                if targets[i][1] == 1 then realClass = 0 else realClass = 1 end
                CONFUSION:add(predictedClass+1, realClass+1)
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
        
        xlua.progress(trained, OPT.N_epoch)
    end
    
    local epochTime = sys.clock() - startTime
    print(string.format("<trainer> time required for this epoch = %d s", epochTime))
    print(string.format("<trainer> time to learn 1 sample = %f ms", 1000 * epochTime/OPT.N_epoch))
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

function visualizeProgress()
    -- deactivate dropout
    V:evaluate()
    
    local imagesReal = DATASET.loadRandomImages(50)
    local imagesFake = createSyntheticImages(50)
    local both = torch.Tensor(100, IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
    for i=1,imagesReal:size(1) do
        both[i] = imagesReal[i]
    end
    for i=1,#imagesFake do
        both[imagesReal:size(1) + i] = imagesFake[i]
    end
    
    local predictions = V:forward(both)
    local goodImages = {}
    local badImages = {}
    for i=1,predictions:size(1) do
        if torch.any(both[i]:gt(1.0)) then
            print("[WARNING] bad values in image")
            print(both[i][both[i]:gt(1.0)])
            print("image i=", i, " is ge1")
        end
        if torch.any(both[i]:lt(0.0)) then
            print("[WARNING] bad values in image")
            print(both[i][both[i]:lt(0.0)])
            print("image i=", i, " is lt0")
        end
    
        --print("i=", i, "predictions[i][1]=", predictions[i][1])
        if predictions[i][1] < 0.5 then
            goodImages[#goodImages+1] = both[i]
        else
            badImages[#badImages+1] = both[i]
        end
    end
    
    if #goodImages > 0 then
        DISP.image(goodImages, {win=OPT.window+0, width=IMG_DIMENSIONS[3]*15, title="V: rated as real images (Epoch " .. EPOCH .. ")"})
    end
    if #badImages > 0 then
        DISP.image(badImages, {win=OPT.window+1, width=IMG_DIMENSIONS[3]*15, title="V: rated as fake images (EPOCH " .. EPOCH .. ")"})
    end
    
    -- reactivate dropout
    V:training()
end

function createSyntheticImages(N, allowSubcalls)
    if allowSubcalls == nil then allowSubcalls = true end
    local images
    local p = math.random()
    if p < 1/4 then
        --print("mix")
        images = createSyntheticImagesMix(N)
    elseif p >= 1/4 and p < 2/4 then
        --print("warp")
        images = createSyntheticImagesWarp(N)
    elseif p >= 2/4 and p < 3/4 then
        --print("stamp")
        images = createSyntheticImagesStamp(N)
    else
        --print("random")
        images = createSyntheticImagesRandom(N)
    end
    
    
    if allowSubcalls and math.random() < 0.33 then
        --print("sub")
        local otherImages = createSyntheticImages(N, false)
        images = mixImageLists(images, otherImages)
    end
    --print("done")
    
    --for i=1,#images do
    --    image.display(images[i])
    --    io.read()
    --end
    
    return images
end

function mixImages(img1, img2, overlay)
    local img = torch.Tensor(IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
    img:zero()
    
    if overlay == nil then
        if math.random() < 0.5 then
            --overlay = createGaussianOverlay(img1:size(2), img1:size(3), 400+math.random(900))
            overlay = getGaussianOverlay()
        else
            overlay = createPixelwiseOverlay(IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
        end
    end
    
    overlay = torch.repeatTensor(overlay, 3, 1, 1)
    --    overlay * img1     + (1 - overlay) * img2
    img = overlay:clone():cmul(img1) + overlay:clone():mul(-1):add(1):cmul(img2)
    img:div(torch.max(img))
    
    --image.display(img1)
    --image.display(img2)
    --image.display(overlay)
    --image.display(img)
    --io.read()
    
    --[[
    for y=1,img1:size(2) do
        for x=1,img1:size(3) do
            for c=1,img1:size(1) do
                img[c][y][x] = overlay[y][x] * img1[c][y][x] + (1 - overlay[y][x]) * img2[c][y][x]
            end
        end
    end
    --]]
    
    return img
end

function mixImageLists(images1, images2)
    local images = {}
    local overlay
    local p = math.random()
    if p < 0.5 then
        --overlay = createGaussianOverlay(IMG_DIMENSIONS[2], IMG_DIMENSIONS[3], 400+math.random(900))
        overlay = getGaussianOverlay()
    else
        overlay = createPixelwiseOverlay(IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
    end
    
    for i=1,#images1 do
        images[i] = mixImages(images1[i]:clone(), images2[i]:clone(), overlay)
    end
    
    return images
end

function createSyntheticImagesMix(N)
    local images = {}
    local img1 = {}
    local img2 = {}
    
    for i=1,N do
        table.insert(img1, TRAIN_DATA[math.random(TRAIN_DATA:size())])
        table.insert(img2, TRAIN_DATA[math.random(TRAIN_DATA:size())])
    end
    
    return mixImageLists(img1, img2)
end

function createSyntheticImagesStamp(N)
    local images = {}
    
    local nbChannels = TRAIN_DATA[1]:size(1)
    local maxY = TRAIN_DATA[1]:size(2)
    local maxX = TRAIN_DATA[1]:size(3)
    
    --local overlay = createGaussianOverlay(maxY, maxX, 1000+math.random(1000))
    local overlay = getGaussianOverlay()
    
    for i=1,N do
        local p = math.random()
        local img1 = TRAIN_DATA[math.random(TRAIN_DATA:size())]
        local img = torch.Tensor(nbChannels, maxY, maxX)
        img:zero()
        local direction = {math.random(10), math.random(10)}
        
        for y=1,maxY do
            for x=1,maxX do
                local coords = withinImageCoords(y + direction[1], x + direction[2], img:size(2), img:size(3))
                
                for c=1,nbChannels do
                    local usualVal = img1[c][y][x]
                    local sourceVal = img1[c][coords[1]][coords[2]]
                    --print("overlay:", overlay[y][x], "usualval:", usualVal, "sourceval:", sourceVal, "result:", (1 - overlay[y][x]) * usualVal + overlay[y][x] * sourceVal)
                    img[c][y][x] = (1 - overlay[y][x]) * usualVal + overlay[y][x] * sourceVal
                end
            end
        end
        img:div(torch.max(img))
        
        table.insert(images, img)
    end
    
    return images
end

function withinImageCoords(y, x, maxY, maxX)
    y = y % maxY
    if y < 1 then
        y = y * (-1)
        y = maxY - y
    end
    
    x = x % maxX
    if x < 1 then
        x = x * (-1)
        x = maxX - x
    end
    
    return {y, x}
end

function createSyntheticImagesWarp(N)
    local images = {}
    
    --local overlay1 = createGaussianOverlay(OPT.scale, OPT.scale)
    --local overlay2 = createGaussianOverlay(OPT.scale, OPT.scale)
    local overlay1 = getGaussianOverlay()
    local overlay2 = getGaussianOverlay()
    overlay1:mul(2.0)
    overlay1:add(-1.0)
    overlay2:mul(2.0)
    overlay2:add(-1.0)
    
    for i=1,N do
        local img1 = TRAIN_DATA[math.random(TRAIN_DATA:size())]:clone()
        local flow = torch.Tensor(2, img1:size(2), img1:size(3))
        flow:zero()
        
        local direction = {1, 0}
        local length = 1 + math.random(4.0)
        --local length = 3.0
        
        for y=1,img1:size(2) do
            for x=1,img1:size(3) do
                flow[1][y][x] = overlay1[y][x] * length
                flow[2][y][x] = overlay2[y][x] * length
            end
        end
        
        local img = image.warp(img1, flow)
        img:div(torch.max(img))
        
        table.insert(images, img)
    end
    
    return images
end

function createSyntheticImagesRandom(N)
    local images = {}
    --local overlay1 = createGaussianOverlay(OPT.scale, OPT.scale, 2000, 10)
    --local overlay2 = createGaussianOverlay(OPT.scale, OPT.scale, 2000, 10)
    --local overlay3 = createGaussianOverlay(OPT.scale, OPT.scale, 10000, 4)
    local overlay1 = getGaussianOverlay(10)
    local overlay2 = getGaussianOverlay(10)
    local overlay3 = getGaussianOverlay(4)
    
    for i=1,N do
        local img = torch.Tensor(IMG_DIMENSIONS[1], IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
        local offsetY = math.random(IMG_DIMENSIONS[2])
        local offsetX = math.random(IMG_DIMENSIONS[3])
        local baseVal = {math.random(), math.random(), math.random()}
        
        for y=1,OPT.scale do
            for x=1,OPT.scale do
                for c=1,IMG_DIMENSIONS[1] do
                    local coords = withinImageCoords(y + c*offsetY, x + c*offsetX, IMG_DIMENSIONS[2], IMG_DIMENSIONS[3])
                    img[c][y][x] = baseVal[c] + (overlay1[y][x] * overlay2[coords[1]][coords[2]]) + overlay3[coords[1]][coords[2]]
                end
            end
        end
        
        img:div(torch.max(img))
        table.insert(images, img)
    end
    
    return images
end

function getGaussianOverlay(blurSize)
    if blurSize == nil then blurSize = 4 end
    
    if OVERLAYS == nil then
        OVERLAYS = {}
        for i=1,1000 do
            OVERLAYS[i] = createGaussianOverlay(IMG_DIMENSIONS[2], IMG_DIMENSIONS[3], 10000, 0)
        end
    end
    
    local overlay1 = OVERLAYS[math.random(#OVERLAYS)]
    local overlay2 = OVERLAYS[math.random(#OVERLAYS)]
    local overlay3 = OVERLAYS[math.random(#OVERLAYS)]
    local overlay4 = OVERLAYS[math.random(#OVERLAYS)]
    local overlayResult = overlay1:mul(2) - overlay2
    overlayResult = torch.clamp(overlayResult, 0.0, 1.0)
    overlayResult = overlayResult + overlay3:cmul(overlay4):mul(2)
    overlayResult = torch.clamp(overlayResult, 0.0, 1.0)
    
    if blurSize > 0 then
        overlayResult = image.convolve(overlayResult, image.gaussian(blurSize), "same")
        overlayResult:div(torch.max(overlayResult))
    end
    
    --image.display(overlayResult)
    --io.read()
    
    return overlayResult
end

function createGaussianOverlay(ySize, xSize, N_points, blurSize)
    N_points = N_points or 1000
    blurSize = blurSize or 6
    local minY = 1
    local maxY = ySize
    local minX = 1
    local maxX = xSize
    
    local overlay = torch.Tensor(maxY, maxX)
    overlay:zero()
    
    local directions = {
        {-1, 0},
        {-1, 1},
        {0, 1},
        {1, 1},
        {1, 0},
        {1, -1},
        {0, -1},
        {-1, -1}
    }
    
    local currentY = math.random(ySize)
    local currentX = math.random(xSize)
    local lastY = math.random(ySize)
    local lastX = math.random(xSize)
    
    for i=1,N_points do
        --print(i)
        local p = math.random()
        if p < 0.02 then
            lastY = currentY
            lastX = currentX
            currentY = math.random(maxY)
            currentX = math.random(maxX)
        elseif math.random() < 0.10 then
            currentY = lastY
            currentX = lastX
        else
            lastY = currentY
            lastX = currentX
            
            local found = false
            while not found do
                local direction = directions[math.random(#directions)]
                currentY = lastY + direction[1]
                currentX = lastX + direction[2]
                
                if (currentY >= minY and currentY <= maxY) and (currentX >= minX and currentX <= maxX) then
                    found = true
                end
            end
        end
        
        --print(currentY, currentX, overlay:size())
        overlay[currentY][currentX] = overlay[currentY][currentX] + 1
    end
    
    overlay:div(torch.max(overlay))
    if blurSize > 0 then
        overlay = image.convolve(overlay, image.gaussian(blurSize), "same")
        overlay:div(torch.max(overlay))
    end
    
    return overlay
end

function createPixelwiseOverlay(ySize, xSize)
    local overlay = torch.Tensor(ySize, xSize)
    overlay:zero()
    
    local p = math.random()
    local pChange = math.random() / 10
    
    for y=1,ySize do
        for x=1,xSize do
            if math.random() > p then
                overlay[y][x] = math.min(2*math.random(), 1)
            else
                overlay[y][x] = 0
            end
            
            if math.random() > 0.5 then
                p = math.max(p - pChange, 0)
            else
                p = math.min(p + pChange, 1.0)
            end
        end
    end
    
    return overlay
end

function minmax(min_x, x, max_x)
    if x < min_x then
        return min_x
    elseif x > max_x then
        return max_x
    else
        return x
    end
end

main()
