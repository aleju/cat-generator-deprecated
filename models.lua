require 'torch'
require 'nn'
require 'LeakyReLU'
require 'dpnn'

local models = {}

-- Creates the encoder part of G as an autoencoder.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G_encoder(dimensions, noiseDim)
    local model = nn.Sequential()
    local activation = nn.LeakyReLU
  
    model:add(nn.SpatialConvolution(dimensions[1], 32, 3, 3, 1, 1, (3-1)/2))
    model:add(activation())
    model:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, (3-1)/2))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))
    model:add(nn.SpatialBatchNormalization(32))
    --model:add(nn.SpatialDropout())
  
    model:add(nn.View(32 * 0.25 * dimensions[2] * dimensions[3]))
    model:add(nn.Linear(32 * 0.25 * dimensions[2] * dimensions[3], 512))
    model:add(activation())
    model:add(nn.Dropout())
    model:add(nn.Linear(512, 512))
    model:add(activation())
    model:add(nn.Dropout())
    model:add(nn.Linear(512, noiseDim))
    model:add(nn.Dropout(0.2))

    model = require('weight-init')(model, 'heuristic')
  
    return model
end

-- Creates the decoder part of G as an autoencoder.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G_decoder(dimensions, noiseDim)
    local model = nn.Sequential()
    local activation = nn.PReLU
  
    model = nn.Sequential()
    model:add(nn.Linear(noiseDim, 2048))
    model:add(activation())
    model:add(nn.Linear(2048, INPUT_SZ))
    model:add(nn.Sigmoid())
    model:add(nn.View(dimensions[1], dimensions[2], dimensions[3]))

    model = require('weight-init')(model, 'heuristic')

    return model
end

-- Creates G, which is identical to the decoder part of G as an autoencoder.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G(dimensions, noiseDim)
    return models.create_G_decoder(dimensions, noiseDim)
end

-- Creates the G as an autoencoder (encoder+decoder).
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @param noiseDim Size of the hidden layer between encoder and decoder.
-- @returns nn.Sequential
function models.create_G_autoencoder(dimensions, noiseDim)
    local model = nn.Sequential()
    model:add(models.create_G_encoder(dimensions, noiseDim))
    model:add(models.create_G_decoder(dimensions, noiseDim))
    return model
end

-- Creates the D
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @returns nn.Sequential
--[[
function models.create_D(dimensions)
    local activation = nn.PReLU
    local branch_conv = nn.Sequential()
  
    --local parallel = nn.Parallel(2, 2)
    local parallel = nn.Concat(2)
    local submodel = nn.Sequential()
    submodel:add(nn.Dropout(0.25))
    submodel:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2))
    submodel:add(activation())
    submodel:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    submodel:add(activation())
    submodel:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    submodel:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2))
    submodel:add(activation())
    submodel:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    submodel:add(activation())
    submodel:add(nn.SpatialDropout())
    submodel:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    submodel:add(nn.View(128 * (1/4)*(1/4) * dimensions[2] * dimensions[3]))
    submodel:add(nn.Linear(128 * (1/4)*(1/4) * dimensions[2] * dimensions[3], 512))
    submodel:add(activation())
    submodel:add(nn.Dropout())
    submodel:add(nn.Linear(512, 128))
    submodel:add(activation())
    parallel:add(submodel)
    for i=2,4 do parallel:add(submodel:sharedClone()) end
  
    branch_conv:add(parallel)
    branch_conv:add(nn.Dropout())
    branch_conv:add(nn.Linear(128*4, 1))
    branch_conv:add(nn.Sigmoid())

    branch_conv = require('weight-init')(branch_conv, 'heuristic')

    return branch_conv
end
--]]

--[[
function models.create_D(dimensions)
    local conv = nn.Sequential()
    local concat = nn.Concat(2)
    concat:add(nn.SpatialConvolution(dimensions[1], 32, 3, 3, 1, 1, (3-1)/2))
    concat:add(nn.SpatialConvolution(dimensions[1], 32, 5, 5, 1, 1, (5-1)/2))
    conv:add(concat)
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(64, 2048, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(2048, 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.View(128 * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear(128 * dimensions[2] * dimensions[3], 1024))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(1024, 1024))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(1024, 1))
    conv:add(nn.Sigmoid())

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end
--]]

--[[
function models.create_D(dimensions)
    local conv = nn.Sequential()
    conv:add(nn.SpatialConvolution(dimensions[1], 32, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialMaxPooling(2, 2))
    conv:add(nn.Dropout())
    conv:add(nn.View(64 * (1/4) * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear(64 * (1/4) * dimensions[2] * dimensions[3], 512))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(512, 512))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(512, 1))
    conv:add(nn.Sigmoid())

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end
--]]

--[[
-- D1
function models.create_D(dimensions)
    local conv = nn.Sequential()
    conv:add(nn.SpatialConvolution(dimensions[1], 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialMaxPooling(2, 2))
    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(256, 2048, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialMaxPooling(2, 2))
    conv:add(nn.SpatialDropout())
    conv:add(nn.View(2048 * (1/4)*(1/4) * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear(2048 * (1/4)*(1/4) * dimensions[2] * dimensions[3], 1024))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(1024, 1024))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(1024, 1))
    conv:add(nn.Sigmoid())

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end
--]]

-- D2
--[[
function models.create_D(dimensions)
    local conv = nn.Sequential()
    conv:add(nn.Dropout(0.1))
    conv:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialMaxPooling(2, 2))
    conv:add(nn.SpatialConvolution(128, 512, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialMaxPooling(2, 2))
    conv:add(nn.SpatialDropout())
    conv:add(nn.View(512 * (1/4)*(1/4) * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear(512 * (1/4)*(1/4) * dimensions[2] * dimensions[3], 2048))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(2048, 1024))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(1024, 1))
    conv:add(nn.Sigmoid())

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end
--]]

-- D3
function models.create_D(dimensions)
    require 'dpnn'
    
    local conv = nn.Sequential()
    local concat = nn.Concat(2)
    
    local sm1size = 256
    local submodel1 = nn.Sequential()
    submodel1:add(nn.View(dimensions[1] * dimensions[2] * dimensions[3]))
    submodel1:add(nn.Linear(dimensions[1] * dimensions[2] * dimensions[3], sm1size))
    submodel1:add(nn.PReLU())
    
    local sm2size = 8 * dimensions[2] * dimensions[3]
    local submodel2 = nn.Sequential()
    submodel2:add(nn.SpatialConvolution(dimensions[1], 8, 3, 3, 1, 1, (3-1)/2))
    submodel2:add(nn.PReLU())
    submodel2:add(nn.View(sm2size))
    submodel2:add(nn.Linear(sm2size, 1024))
    submodel2:add(nn.PReLU())
    -- 8 * 16 * 16 = 8 * 256 = 2048
    
    local sm3size = 1024 * 0.25 * dimensions[2] * dimensions[3]
    local submodel3 = nn.Sequential()
    submodel3:add(nn.SpatialConvolution(dimensions[1], 32, 3, 3, 1, 1, (3-1)/2))
    submodel3:add(nn.PReLU())
    submodel3:add(nn.SpatialConvolution(32, 1024, 3, 3, 1, 1, (3-1)/2))
    submodel3:add(nn.PReLU())
    submodel3:add(nn.SpatialMaxPooling(2, 2))
    submodel3:add(nn.SpatialDropout())
    submodel3:add(nn.View(16, (1024/16)*0.25*dimensions[2]*dimensions[3]))
    local sm3parallel = nn.Parallel(2, 2)
    for i=1,16 do
        sm3parallel:add(nn.Linear((1024/16)*0.25*dimensions[2]*dimensions[3], 512))
    end
    submodel3:add(nn.PReLU())
    submodel3:add(sm3parallel)
    --submodel3:add(nn.PReLU())
    --submodel3:add(nn.Linear(16*1024, 1024))
    --submodel3:add(nn.PReLU())
    -- 1024 * 0.25 * 16 * 16 = 256 * 16 * 16 = 256**2 = 65536
    
    concat:add(submodel1)
    concat:add(submodel2)
    concat:add(submodel3)
    
    conv:add(concat)
    --conv:add(nn.View(sm2size + sm3size))
    --conv:add(nn.Linear(sm2size + sm3size, 1024))
    conv:add(nn.Dropout())
    conv:add(nn.Linear(sm1size + 1024 + (16*512), 1024))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(1024, 1))
    conv:add(nn.Sigmoid())

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

--[[
function models.create_D(dimensions)
    local activation = nn.PReLU
    local branch_conv = nn.Sequential()
  
    --local parallel = nn.Parallel(2, 2)
    local parallel = nn.Concat(2)
    local submodel = nn.Sequential()
    submodel:add(nn.Dropout(0.25))
    submodel:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2))
    submodel:add(activation())
    submodel:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    submodel:add(activation())
    submodel:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    submodel:add(nn.SpatialConvolution(64, 256, 3, 3, 1, 1, (3-1)/2))
    submodel:add(activation())
    submodel:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, (3-1)/2))
    submodel:add(activation())
    submodel:add(nn.SpatialDropout())
    submodel:add(nn.View(256 * (1/4) * dimensions[2] * dimensions[3]))
    submodel:add(nn.Linear(256 * (1/4) * dimensions[2] * dimensions[3], 1024))
    submodel:add(activation())
    submodel:add(nn.Dropout())
    submodel:add(nn.Linear(1024, 512))
    submodel:add(activation())
    parallel:add(submodel)
    for i=2,4 do parallel:add(submodel:sharedClone()) end
  
    branch_conv:add(parallel)
    branch_conv:add(nn.Dropout())
    branch_conv:add(nn.Linear(512*4, 1))
    branch_conv:add(nn.Sigmoid())

    branch_conv = require('weight-init')(branch_conv, 'heuristic')

    return branch_conv
end
--]]

--[[
function models.create_D(dimensions)
    local model = nn.Sequential()
    model:add(nn.View(dimensions[1] * dimensions[2] * dimensions[3]))
    model:add(nn.Linear(dimensions[1] * dimensions[2] * dimensions[3], 2048))
    model:add(nn.PReLU())
    model:add(nn.Linear(2048, 2048))
    model:add(nn.PReLU())
    model:add(nn.Linear(2048, 1))
    model:add(nn.Sigmoid())

    model = require('weight-init')(model, 'heuristic')

    return model
end
--]]

-- Creates V.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @returns nn.Sequential
function models.create_V(dimensions)
    local model = nn.Sequential()
    local activation = nn.LeakyReLU
  
    model:add(nn.SpatialConvolution(dimensions[1], 128, 3, 3, 1, 1, (3-1)/2))
    model:add(activation())
    model:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    model:add(activation())
    --model:add(nn.SpatialMaxPooling(2, 2))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(nn.Dropout())
  
    model:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2))
    model:add(activation())
    model:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, (3-1)/2))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(nn.SpatialDropout())
    local imgSize = 0.25 * dimensions[2] * dimensions[3]
    model:add(nn.View(256 * imgSize))
    model:add(nn.BatchNormalization(256 * imgSize))
  
    model:add(nn.Linear(256 * imgSize, 1024))
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

return models
