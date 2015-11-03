require 'torch'
require 'image'
require 'paths'
require 'pl'
require 'layers.cudnnSpatialConvolutionUpsample'
require 'stn'
require 'LeakyReLU'
NN_UTILS = require 'utils.nn_utils'
DATASET = require 'dataset'

OPT = lapp[[
    --save_base     (default "logs/final")                 directory in which the networks are saved
    --save_c2f22    (default "logs/final")
    --save_c2f32    (default "logs/final")
    --G_base        (default "et1b_adversarial3.net")      
    --D_base        (default "et1b_adversarial3.net")      
    --G_c2f22       (default "e2-3b_adversarial_c2f_16_to_22.net")  
    --D_c2f22       (default "e2-3b_adversarial_c2f_16_to_22.net")  
    --G_c2f32       (default "e2-3d_adversarial_c2f_22_to_32_e650.net")  
    --D_c2f32       (default "e2-3d_adversarial_c2f_22_to_32_e650.net")  
    --neighbours                                           Whether to search for nearest neighbours of generated images in the dataset (takes long)
    --scale         (default 16)
    --grayscale                                            grayscale mode on/off
    --writeto       (default "samples")                    directory to save the images to
    --seed          (default 1)
    --gpu           (default 0)                            GPU to run on
    --runs          (default 1)                           How often to sample and save images
    --noiseDim      (default 100)
    --batchSize     (default 16)
    --aws                                                  run in AWS mode
]]

--[[
    --save_base     (default "logs/final")                 directory in which the networks are saved
    --save_c2f22    (default "logs/final")
    --save_c2f32    (default "logs/final")
    --G_base        (default "et1b_adversarial3.net")      
    --D_base        (default "et1b_adversarial3.net")      
    --G_c2f22       (default "e2-3b_adversarial_c2f_16_to_22.net")  
    --D_c2f22       (default "e2-3b_adversarial_c2f_16_to_22.net")  
    --G_c2f32       (default "e2-3d_adversarial_c2f_22_to_32_e650.net")  
    --D_c2f32       (default "e2-3d_adversarial_c2f_22_to_32_e650.net")  
    --neighbours                                           Whether to search for nearest neighbours of generated images in the dataset (takes long)
    --scale         (default 16)
    --grayscale                                            grayscale mode on/off
    --writeto       (default "samples")                    directory to save the images to
    --seed          (default 1)
    --gpu           (default 0)                            GPU to run on
    --runs          (default 1)                           How often to sample and save images
    --noiseDim      (default 100)
    --batchSize     (default 16)
    --aws                                                  run in AWS mode
--]]

--[[
    --save_base     (default "logs/final")                 directory in which the networks are saved
    --save_c2f22    (default "logs")
    --save_c2f32    (default "logs/final")
    --G_base        (default "e43e_adversarial.net")      
    --D_base        (default "e43e_adversarial.net")      
    --G_c2f22       (default "adversarial_c2f_16_to_22.net")  
    --D_c2f22       (default "adversarial_c2f_16_to_22.net")  
    --G_c2f32       (default "e2-3d_adversarial_c2f_22_to_32_e650.net")  
    --D_c2f32       (default "e2-3d_adversarial_c2f_22_to_32_e650.net")  
    --neighbours                                           Whether to search for nearest neighbours of generated images in the dataset (takes long)
    --scale         (default 16)
    --grayscale                                            grayscale mode on/off
    --writeto       (default "samples")                    directory to save the images to
    --seed          (default 1)
    --gpu           (default 0)                            GPU to run on
    --runs          (default 1)                           How often to sample and save images
    --noiseDim      (default 100)
    --batchSize     (default 16)
    --aws                                                  run in AWS mode
--]]

--[[
    --save_base     (default "logs/final")                 directory in which the networks are saved
    --save_c2f22    (default "logs/final")
    --save_c2f32    (default "logs/final")
    --G_base        (default "e43e_adversarial.net")      
    --D_base        (default "e43e_adversarial.net")      
    --G_c2f22       (default "e2-1b_adversarial_c2f_16_to_22.net")  
    --D_c2f22       (default "e2-1b_adversarial_c2f_16_to_22.net")  
    --G_c2f32       (default "e2-2b_adversarial_c2f_22_to_32.net")  
    --D_c2f32       (default "e2-2b_adversarial_c2f_22_to_32.net") 
--]]

if OPT.gpu < 0 then
    print("[ERROR] Sample script currently only runs on GPU, set --gpu=x where x is between 0 and 3.")
    exit()
end

print("Starting gpu support...")
require 'cutorch'
require 'cunn'
torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(OPT.seed)
torch.manualSeed(OPT.seed)
cutorch.setDevice(OPT.gpu + 1)
cutorch.manualSeed(OPT.seed)

if OPT.grayscale then
    IMG_DIMENSIONS = {1, OPT.scale, OPT.scale}
else
    IMG_DIMENSIONS = {3, OPT.scale, OPT.scale}
end

DATASET.nbChannels = IMG_DIMENSIONS[1]
DATASET.setFileExtension("jpg")
DATASET.setScale(OPT.scale)

if OPT.aws then
    DATASET.setDirs({"/mnt/datasets/out_aug_64x64"})
else
    DATASET.setDirs({"dataset/out_aug_64x64"})
end


function main()
    local G, D, G_c2f22, D_c2f22, G_c2f32, D_c2f32, normMean, normStd = loadModels()
    MODEL_G = G
    MODEL_D = D
    
    print("Sampling...")
    for run=1,OPT.runs do
        -- save 64 randomly selected images from the training set
        local imagesTrainList = DATASET.loadRandomImages(64)
        local imagesTrain = torch.Tensor(#imagesTrainList, imagesTrainList[1]:size(1), imagesTrainList[1]:size(2), imagesTrainList[1]:size(3))
        for i=1,#imagesTrainList do
            imagesTrain[i] = imagesTrainList[i]:clone()
        end
        image.save(paths.concat(OPT.writeto, string.format('trainset_s1_%04d_base.jpg', run)), toGrid(imagesTrain, 8))
        
        -- sample 1024 new images from G
        local images = NN_UTILS.createImages(1024, false)
        
        -- validate image dimensions
        if images[1]:size(1) ~= IMG_DIMENSIONS[1] or images[1]:size(2) ~= IMG_DIMENSIONS[2] or images[1]:size(3) ~= IMG_DIMENSIONS[3] then
            print("[WARNING] dimension mismatch between images generated by G and command line parameters, --grayscale falsly on/off or --scale not set correctly")
            print("Dimension G:", images[1]:size())
            print("Settings:", IMG_DIMENSIONS)
        end
        
        -- save a big image of those 1024 random images
        image.save(paths.concat(OPT.writeto, string.format('random1024_%04d_base.jpg', run)), toGrid(images, 32))
        
        -- Collect the best and worst images (according to D) from these images
        -- Save: 64 best images, 64 worst images, 64 randomly selected images
        local imagesBest, predictions = NN_UTILS.sortImagesByPrediction(images, false, 64)
        local imagesWorst, predictions = NN_UTILS.sortImagesByPrediction(images, true, 64)
        local imagesRandom = selectRandomImagesFrom(images, 64)
        imagesBest = imageListToTensor(imagesBest)
        imagesWorst = imageListToTensor(imagesWorst)
        imagesRandom = imageListToTensor(imagesRandom)
        image.save(paths.concat(OPT.writeto, string.format('best_%04d_base.jpg', run)), toGrid(imagesBest, 8))
        image.save(paths.concat(OPT.writeto, string.format('worst_%04d_base.jpg', run)), toGrid(imagesWorst, 8))
        image.save(paths.concat(OPT.writeto, string.format('random_%04d_base.jpg', run)), toGrid(imagesRandom, 8))
        
        -- Extract the 16 best images and find their closest neighbour in the training set
        if OPT.neighbours then
            local searchFor = {}
            for i=1,16 do
                table.insert(searchFor, imagesBest[i]:clone())
            end
            local neighbours = findClosestNeighboursOf(searchFor)
            image.save(paths.concat(OPT.writeto, string.format('best_%04d_neighbours_base.jpg', run)), toNeighboursGrid(neighbours, 8))
        end
        
        -- Upsample from 16 to 22px and refine with G_c2f22
        --[[
        imagesBest = torch.Tensor(64, 1, 16, 16)
        for i=1,64 do
            imagesBest[i] = imageTensor[i]:clone()
        end
        --]]
        
        imagesTrain = upscale(imagesTrain, 22, 22)
        image.save(paths.concat(OPT.writeto, string.format('trainset_s2_up22_%04d.jpg', run)), toGrid(imagesTrain, 8))
        imagesBest = upscale(imagesBest, 22, 22)
        imagesWorst = upscale(imagesWorst, 22, 22)
        imagesRandom = upscale(imagesRandom, 22, 22)
        imagesTrain = normalize(imagesTrain, normMean, normStd)
        image.save(paths.concat(OPT.writeto, string.format('trainset_s3_up22norm_%04d.jpg', run)), toGrid(imagesTrain, 8))
        imagesBest = normalize(imagesBest, normMean, normStd)
        imagesWorst = normalize(imagesWorst, normMean, normStd)
        imagesRandom = normalize(imagesRandom, normMean, normStd)
        local imagesTrainC2F22 = c2f(imagesTrain, G_c2f22, D_c2f22)
        local imagesBestC2F22 = c2f(imagesBest, G_c2f22, D_c2f22)
        local imagesWorstC2F22 = c2f(imagesWorst, G_c2f22, D_c2f22)
        local imagesRandomC2F22 = c2f(imagesRandom, G_c2f22, D_c2f22)
        image.save(paths.concat(OPT.writeto, string.format('trainset_s4_c2f_22_%04d.jpg', run)), toGrid(imagesTrainC2F22, 8))
        image.save(paths.concat(OPT.writeto, string.format('best_%04d_c2f_22.jpg', run)), toGrid(imagesBestC2F22, 8))
        image.save(paths.concat(OPT.writeto, string.format('worst_%04d_c2f_22.jpg', run)), toGrid(imagesWorstC2F22, 8))
        image.save(paths.concat(OPT.writeto, string.format('random_%04d_c2f_22.jpg', run)), toGrid(imagesRandomC2F22, 8))
        
        -- Upsample from 22 to 32px and refine with G_c2f32
        imagesTrainC2F22 = upscale(imagesTrainC2F22, 32, 32)
        image.save(paths.concat(OPT.writeto, string.format('trainset_s5_up32_%04d_base.jpg', run)), toGrid(imagesTrainC2F22, 8))
        imagesBestC2F22 = upscale(imagesBestC2F22, 32, 32)
        imagesWorstC2F22 = upscale(imagesWorstC2F22, 32, 32)
        imagesRandomC2F22 = upscale(imagesRandomC2F22, 32, 32)
        local imagesTrainC2F32 = c2f(imagesTrainC2F22, G_c2f32, D_c2f32)
        local imagesBestC2F32 = c2f(imagesBestC2F22, G_c2f32, D_c2f32)
        local imagesWorstC2F32 = c2f(imagesWorstC2F22, G_c2f32, D_c2f32)
        local imagesRandomC2F32 = c2f(imagesRandomC2F22, G_c2f32, D_c2f32)
        image.save(paths.concat(OPT.writeto, string.format('trainset_s6_c2f_32_%04d.jpg', run)), toGrid(imagesTrainC2F32, 8))
        image.save(paths.concat(OPT.writeto, string.format('best_%04d_c2f_32.jpg', run)), toGrid(imagesBestC2F32, 8))
        image.save(paths.concat(OPT.writeto, string.format('worst_%04d_c2f_32.jpg', run)), toGrid(imagesWorstC2F32, 8))
        image.save(paths.concat(OPT.writeto, string.format('random_%04d_c2f_32.jpg', run)), toGrid(imagesRandomC2F32, 8))
        
        xlua.progress(run, OPT.runs)
    end
    
    print("Finished.")
end

function findClosestNeighboursOf(images)
    local result = {}
    local trainingSet = DATASET.loadImages(0, 9999999)
    for i=1,#images do
        local img = images[i]
        local closestDist = nil
        local closestImg = nil
        for j=1,trainingSet:size() do
            local dist = torch.dist(trainingSet[j], img)
            if closestDist == nil or dist < closestDist then
                closestDist = dist
                closestImg = trainingSet[j]:clone()
            end
        end
        table.insert(result, {img, closestImg, closestDist})
    end
    
    return result
end

function c2f(images, G, D)
    local fineSize = images[1]:size(2)
    local triesPerImage = 10
    local result = {}
    for i=1,images:size(1) do
        local imgTensor = torch.Tensor(triesPerImage, images[1]:size(1), fineSize, fineSize)
        local img = images[i]:clone()
        local height = img:size(2)
        local width = img:size(3)
        
        for j=1,triesPerImage do
            imgTensor[j] = img:clone()
        end
        
        local noiseInputs = torch.Tensor(triesPerImage, 1, fineSize, fineSize)
        noiseInputs:uniform(-1, 1)
        
        local diffs = G:forward({noiseInputs, imgTensor})
        --diffs:float()
        local predictions = D:forward({diffs, imgTensor})
        
        local maxval = nil
        local maxdiff = nil
        for j=1,triesPerImage do
            if maxval == nil or predictions[j][1] > maxval then
                maxval = predictions[j][1]
                maxdiff = diffs[j]
            end
        end
        
        local imgRefined = torch.add(img, maxdiff)
        imgRefined = torch.clamp(imgRefined, -1.0, 1.0)
        table.insert(result, imgRefined)
    end
    
    return imageListToTensor(result)
end

function upscale(images, newHeight, newWidth)
    local newImages = torch.Tensor(images:size(1), images[1]:size(1), newHeight, newWidth)
    for i=1,images:size(1) do
        newImages[i] = image.scale(images[i], newHeight, newWidth)
    end
    return newImages
    --[[
    local newImages = {}
    for i=1,#images do
        table.insert(newImages, image.scale(images[i], newHeight, newWidth))
    end
    return newImages
    --]]
end

function imageListToTensor(images)
    local newImages = torch.Tensor(#images, images[1]:size(1), images[1]:size(2), images[1]:size(3))
    for i=1,#images do
        newImages[i] = images[i]
    end
    return newImages
end

function normalize(images, mean_, std_)
    --local images2 = torch.Tensor(images:size(1), images:size(2), images:size(3), images:size(4))
    

    -- normalizes in-place
    NN_UTILS.normalize(images, mean_, std_)
    --NN_UTILS.normalize(images, nil, nil)
    return images
end

function toGrid(images, nrow)
    return image.toDisplayTensor{input=images, nrow=nrow}
end

function toNeighboursGrid(imagesWithNeighbours)
    local img = imagesWithNeighbours[1][1]
    local imgpairs = torch.Tensor(#imagesWithNeighbours*2, img:size(1), img:size(2), img:size(3))
    
    local imgpairs_idx = 1
    for i=1,#imagesWithNeighbours do
        imgpairs[imgpairs_idx] = imagesWithNeighbours[i][1]
        imgpairs[imgpairs_idx + 1] = imagesWithNeighbours[i][2]
        imgpairs_idx = imgpairs_idx + 2
    end
    
    return image.toDisplayTensor{input=imgpairs, nrow=#imagesWithNeighbours}
end

function selectRandomImagesFrom(tensor, n)
    local shuffle = torch.randperm(tensor:size(1))
    local result = {}
    for i=1,math.min(n, tensor:size(1)) do
        table.insert(result, tensor[ shuffle[i] ])
    end
    return result
end



function loadModels()
    local file
    
    -- load G base
    file = torch.load(paths.concat(OPT.save_base, OPT.G_base))
    local G = file.G2
    G:evaluate()
    
    -- load D base
    file = torch.load(paths.concat(OPT.save_base, OPT.D_base))
    local D = file.D
    D:evaluate()
    
    -- load G c2f 16 to 22
    file = torch.load(paths.concat(OPT.save_c2f22, OPT.G_c2f22))
    local G_c2f22 = file.G
    G_c2f22:evaluate()
    local normMean = file.normalize_mean
    local normStd = file.normalize_std
    
    -- load D c2f 16 to 22
    file = torch.load(paths.concat(OPT.save_c2f22, OPT.D_c2f22))
    local D_c2f22 = file.D
    D_c2f22:evaluate()
    
    -- load G c2f 22 to 32
    file = torch.load(paths.concat(OPT.save_c2f32, OPT.G_c2f32))
    local G_c2f32 = file.G
    G_c2f32:evaluate()
    
    -- load D c2f 22 to 32
    file = torch.load(paths.concat(OPT.save_c2f32, OPT.D_c2f32))
    local D_c2f32 = file.D
    D_c2f32:evaluate()
    
    return G, D, G_c2f22, D_c2f22, G_c2f32, D_c2f32, normMean, normStd
end

main()
