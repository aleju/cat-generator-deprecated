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
    model:add(nn.Dropout(0.1))

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
    model:add(nn.Linear(noiseDim, 1024))
    model:add(activation())
    model:add(nn.Linear(1024, INPUT_SZ))
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

function models.create_D(dimensions)
    local conv = nn.Sequential()
    conv:add(nn.SpatialConvolution(dimensions[1], 32, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(32, 1024, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(1024, 64, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout())
    conv:add(nn.View(64 * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear(64 * dimensions[2] * dimensions[3], 1024))
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

-- Creates V.
-- @param dimensions The dimensions of each image as {channels, height, width}.
-- @returns nn.Sequential
function models.create_V(dimensions)
    local model = nn.Sequential()
    local activation = nn.LeakyReLU
  
    model:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2))
    model:add(activation())
    model:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    model:add(activation())
    --model:add(nn.SpatialMaxPooling(2, 2))
    model:add(nn.SpatialBatchNormalization(64))
    model:add(nn.Dropout())
  
    model:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2))
    model:add(activation())
    model:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(nn.SpatialDropout())
    local imgSize = 0.25 * dimensions[2] * dimensions[3]
    model:add(nn.View(128 * imgSize))
    model:add(nn.BatchNormalization(128 * imgSize))
  
    model:add(nn.Linear(128 * imgSize, 1024))
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
