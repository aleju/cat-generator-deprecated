require 'torch'
require 'nn'
require 'LeakyReLU'
--require 'dpnn'

local models = {}

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

function models.create_G(dimensions, noiseDim)
    return models.create_G_decoder(dimensions, noiseDim)
end

function models.create_G_autoencoder(dimensions, noiseDim)
    local model = nn.Sequential()
    model:add(models.create_G_encoder(dimensions, noiseDim))
    model:add(models.create_G_decoder(dimensions, noiseDim))
    return model
end

function create_V(dimensions)
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
  
    model:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2))
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
    model:add(nn.View(128 * 0.25 * dimensions[2] * dimensions[3]))
    model:add(nn.BatchNormalization(128 * 0.25 * dimensions[2] * dimensions[3]))
  
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

return models
