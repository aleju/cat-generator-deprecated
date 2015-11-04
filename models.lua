require 'torch'
require 'nn'
require 'LeakyReLU'
require 'dpnn'
require 'layers.cudnnSpatialConvolutionUpsample'
require 'stn'

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
    local inputSz = dimensions[1] * dimensions[2] * dimensions[3]
    local activation = nn.PReLU
  
    local model = nn.Sequential()
    model:add(nn.Linear(noiseDim, 2048))
    model:add(activation())
    model:add(nn.Linear(2048, inputSz))
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

-- D1b
function models.create_D(dimensions)
    local conv = nn.Sequential()
    conv:add(nn.SpatialConvolution(dimensions[1], 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialMaxPooling(2, 2))
    conv:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(256, 1024, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialMaxPooling(2, 2))
    conv:add(nn.SpatialDropout())
    conv:add(nn.View(1024 * (1/4)*(1/4) * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear(1024 * (1/4)*(1/4) * dimensions[2] * dimensions[3], 1024))
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

-- D with spatial transformer
function models.create_D_st(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end
    conv:add(models.createSpatialTransformer(true, false, false, dimensions[2], dimensions[1], cuda))
    conv:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    
    local branch1 = nn.Sequential()
    branch1:add(models.createSpatialTransformer(false, true, true, dimensions[2], 64, cuda))
    branch1:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2))
    branch1:add(nn.PReLU())
    branch1:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    branch1:add(nn.PReLU())
    branch1:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    
    local branch2 = nn.Sequential()
    branch2:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    branch2:add(nn.SpatialConvolution(64, 128, 5, 5, 1, 1, (5-1)/2))
    branch2:add(nn.PReLU())
    branch2:add(nn.SpatialConvolution(128, 128, 7, 7, 1, 1, (7-1)/2))
    branch2:add(nn.PReLU())
    
    local concy = nn.Concat(2)
    concy:add(branch1)
    concy:add(branch2)
    
    conv:add(concy)
    conv:add(nn.SpatialConvolution(128+128, 256, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialDropout())
    conv:add(nn.View(256 * 0.25 * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear(256 * 0.25 * dimensions[2] * dimensions[3], 256))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(256, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

-- D with a spatial transformer at the start and then 3 branches of which 2 also contain
-- spatial transformers and another one has none (focuses on the whole image).
-- Also has dropout at the start, right after the first spatial transformer.
function models.create_D_st2(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end
    conv:add(models.createSpatialTransformer(true, false, false, dimensions[2], dimensions[1], cuda))
    conv:add(nn.Dropout(0.2))
    conv:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    
    local branch1 = nn.Sequential()
    branch1:add(models.createSpatialTransformer(true, true, true, dimensions[2], 64, cuda))
    branch1:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    branch1:add(nn.PReLU())
    branch1:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    branch1:add(nn.PReLU())
    branch1:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    
    local branch2 = nn.Sequential()
    branch2:add(models.createSpatialTransformer(true, true, true, dimensions[2], 64, cuda))
    branch2:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    branch2:add(nn.PReLU())
    branch2:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    branch2:add(nn.PReLU())
    branch2:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    
    local branch3 = nn.Sequential()
    branch3:add(models.createSpatialTransformer(true, true, true, dimensions[2], 64, cuda))
    branch3:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    branch3:add(nn.PReLU())
    branch3:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    branch3:add(nn.PReLU())
    branch3:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    
    local branch4 = nn.Sequential()
    branch4:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    branch4:add(nn.SpatialConvolution(64, 128, 5, 5, 1, 1, (5-1)/2))
    branch4:add(nn.PReLU())
    branch4:add(nn.SpatialConvolution(128, 128, 7, 7, 1, 1, (7-1)/2))
    branch4:add(nn.PReLU())
    
    local concy = nn.Concat(2)
    concy:add(branch1)
    concy:add(branch2)
    concy:add(branch3)
    concy:add(branch4)
    
    conv:add(concy)
    conv:add(nn.SpatialDropout())
    conv:add(nn.View((64+64+64+128) * 0.25 * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear((64+64+64+128) * 0.25 * dimensions[2] * dimensions[3], 256))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(256, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

-- D with one spatial transformer at the start and then 3 branches of which 2 also have spatial
-- transformers. One branch doesn't have a spatial transformer.
-- In contrast to st2 this doesn't have dropout at the start and also no pooling layers.
function models.create_D_st3(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end
    conv:add(models.createSpatialTransformer(true, false, false, dimensions[2], dimensions[1], cuda))
    conv:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    
    local branch1 = nn.Sequential()
    branch1:add(models.createSpatialTransformer(true, true, true, dimensions[2], 64, cuda))
    branch1:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    branch1:add(nn.PReLU())
    branch1:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    branch1:add(nn.PReLU())
    
    local branch2 = nn.Sequential()
    branch2:add(models.createSpatialTransformer(true, true, true, dimensions[2], 64, cuda))
    branch2:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    branch2:add(nn.PReLU())
    branch2:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    branch2:add(nn.PReLU())
    
    local branch3 = nn.Sequential()
    branch3:add(models.createSpatialTransformer(true, true, true, dimensions[2], 64, cuda))
    branch3:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    branch3:add(nn.PReLU())
    branch3:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    branch3:add(nn.PReLU())
    
    local branch4 = nn.Sequential()
    branch4:add(nn.SpatialConvolution(64, 128, 5, 5, 1, 1, (5-1)/2))
    branch4:add(nn.PReLU())
    branch4:add(nn.SpatialConvolution(128, 128, 7, 7, 1, 1, (7-1)/2))
    branch4:add(nn.PReLU())
    
    local concy = nn.Concat(2)
    concy:add(branch1)
    concy:add(branch2)
    concy:add(branch3)
    concy:add(branch4)
    
    conv:add(concy)
    conv:add(nn.SpatialDropout())
    conv:add(nn.View((64+64+64+128) * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear((64+64+64+128) * dimensions[2] * dimensions[3], 256))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(256, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

-- Larger version of D_st3 (i.e. convolution kernels increased).
function models.create_D_st3b(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end
    conv:add(models.createSpatialTransformer(true, false, false, dimensions[2], dimensions[1], cuda))
    conv:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    
    local branch1 = nn.Sequential()
    branch1:add(models.createSpatialTransformer(true, true, true, dimensions[2], 64, cuda))
    branch1:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2))
    branch1:add(nn.PReLU())
    branch1:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    branch1:add(nn.PReLU())
    branch1:add(nn.SpatialMaxPooling(2, 2))
    
    local branch2 = nn.Sequential()
    branch2:add(models.createSpatialTransformer(true, true, true, dimensions[2], 64, cuda))
    branch2:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2))
    branch2:add(nn.PReLU())
    branch2:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    branch2:add(nn.PReLU())
    branch2:add(nn.SpatialMaxPooling(2, 2))
    
    local branch3 = nn.Sequential()
    branch3:add(models.createSpatialTransformer(true, true, true, dimensions[2], 64, cuda))
    branch3:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, (3-1)/2))
    branch3:add(nn.PReLU())
    branch3:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    branch3:add(nn.PReLU())
    branch3:add(nn.SpatialMaxPooling(2, 2))
    
    local branch4 = nn.Sequential()
    branch4:add(nn.SpatialConvolution(64, 128, 5, 5, 1, 1, (5-1)/2))
    branch4:add(nn.PReLU())
    branch4:add(nn.SpatialMaxPooling(2, 2))
    branch4:add(nn.SpatialConvolution(128, 128, 7, 7, 1, 1, (7-1)/2))
    branch4:add(nn.PReLU())
    
    local concy = nn.Concat(2)
    concy:add(branch1)
    concy:add(branch2)
    concy:add(branch3)
    concy:add(branch4)
    
    conv:add(concy)
    conv:add(nn.SpatialDropout())
    conv:add(nn.View((128*4) * 0.25 * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear((128*4) * 0.25 * dimensions[2] * dimensions[3], 256))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(256, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

-- D_st with more convolution kernels and larger linear layer.
function models.create_D_st3c(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end
    conv:add(models.createSpatialTransformer(true, false, false, dimensions[2], dimensions[1], cuda))
    conv:add(nn.SpatialConvolution(dimensions[1], 64, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    
    local branch1 = nn.Sequential()
    branch1:add(models.createSpatialTransformer(true, true, true, dimensions[2], 64, cuda))
    branch1:add(nn.SpatialConvolution(64, 256, 3, 3, 1, 1, (3-1)/2))
    branch1:add(nn.PReLU())
    branch1:add(nn.SpatialMaxPooling(2, 2))
    branch1:add(nn.SpatialConvolution(256, 64, 3, 3, 1, 1, (3-1)/2))
    branch1:add(nn.PReLU())
    branch1:add(nn.SpatialMaxPooling(2, 2))
    
    local branch2 = nn.Sequential()
    branch2:add(models.createSpatialTransformer(true, true, true, dimensions[2], 64, cuda))
    branch2:add(nn.SpatialConvolution(64, 256, 3, 3, 1, 1, (3-1)/2))
    branch2:add(nn.PReLU())
    branch2:add(nn.SpatialMaxPooling(2, 2))
    branch2:add(nn.SpatialConvolution(256, 64, 3, 3, 1, 1, (3-1)/2))
    branch2:add(nn.PReLU())
    branch2:add(nn.SpatialMaxPooling(2, 2))
    
    local branch3 = nn.Sequential()
    branch3:add(models.createSpatialTransformer(true, true, true, dimensions[2], 64, cuda))
    branch3:add(nn.SpatialConvolution(64, 256, 3, 3, 1, 1, (3-1)/2))
    branch3:add(nn.PReLU())
    branch3:add(nn.SpatialMaxPooling(2, 2))
    branch3:add(nn.SpatialConvolution(256, 64, 3, 3, 1, 1, (3-1)/2))
    branch3:add(nn.PReLU())
    branch3:add(nn.SpatialMaxPooling(2, 2))
    
    local branch4 = nn.Sequential()
    branch4:add(nn.SpatialMaxPooling(2, 2))
    branch4:add(nn.SpatialConvolution(64, 512, 5, 5, 1, 1, (5-1)/2))
    branch4:add(nn.PReLU())
    branch4:add(nn.SpatialConvolution(512, 128, 7, 7, 1, 1, (7-1)/2))
    branch4:add(nn.PReLU())
    branch4:add(nn.SpatialMaxPooling(2, 2))
    
    local concy = nn.Concat(2)
    concy:add(branch1)
    concy:add(branch2)
    concy:add(branch3)
    concy:add(branch4)
    
    conv:add(concy)
    conv:add(nn.SpatialDropout())
    conv:add(nn.View((64+64+64+128) * 0.25 * 0.25 * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear((64+64+64+128) * 0.25 * 0.25 * dimensions[2] * dimensions[3], 1024))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(1024, 1024))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(1024, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

-- D_st3c with more convolution kernels and less nodes in linear layer.
function models.create_D_st3d(dimensions, cuda)
    local conv = nn.Sequential()
    if cuda then
        conv:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    end
    conv:add(models.createSpatialTransformer(true, false, false, dimensions[2], dimensions[1], cuda))
    conv:add(nn.SpatialConvolution(dimensions[1], 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    conv:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, (3-1)/2))
    conv:add(nn.PReLU())
    
    local branch1 = nn.Sequential()
    branch1:add(models.createSpatialTransformer(true, true, true, dimensions[2], 128, cuda))
    branch1:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2))
    branch1:add(nn.PReLU())
    branch1:add(nn.SpatialConvolution(256, 64, 3, 3, 1, 1, (3-1)/2))
    branch1:add(nn.PReLU())
    
    local branch2 = nn.Sequential()
    branch2:add(models.createSpatialTransformer(true, true, true, dimensions[2], 128, cuda))
    branch2:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2))
    branch2:add(nn.PReLU())
    branch2:add(nn.SpatialConvolution(256, 64, 3, 3, 1, 1, (3-1)/2))
    branch2:add(nn.PReLU())
    
    local branch3 = nn.Sequential()
    branch3:add(models.createSpatialTransformer(true, true, true, dimensions[2], 128, cuda))
    branch3:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2))
    branch3:add(nn.PReLU())
    branch3:add(nn.SpatialConvolution(256, 64, 3, 3, 1, 1, (3-1)/2))
    branch3:add(nn.PReLU())
    
    local branch4 = nn.Sequential()
    branch4:add(nn.SpatialConvolution(128, 256, 5, 5, 1, 1, (5-1)/2))
    branch4:add(nn.PReLU())
    branch4:add(nn.SpatialConvolution(256, 64, 7, 7, 1, 1, (7-1)/2))
    branch4:add(nn.PReLU())
    
    local concy = nn.Concat(2)
    concy:add(branch1)
    concy:add(branch2)
    concy:add(branch3)
    concy:add(branch4)
    
    conv:add(concy)
    conv:add(nn.SpatialDropout())
    conv:add(nn.View(64*4 * dimensions[2] * dimensions[3]))
    conv:add(nn.Linear(64*4 * dimensions[2] * dimensions[3], 512))
    conv:add(nn.PReLU())
    conv:add(nn.Dropout())
    conv:add(nn.Linear(512, 1))
    conv:add(nn.Sigmoid())

    if cuda then
        conv:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
        conv:cuda()
    end

    conv = require('weight-init')(conv, 'heuristic')

    return conv
end

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
    model:add(nn.SpatialBatchNormalization(128))
    model:add(nn.Dropout())
  
    model:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, (3-1)/2))
    model:add(activation())
    model:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, (3-1)/2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(activation())
    model:add(nn.SpatialMaxPooling(2, 2))
    model:add(nn.SpatialDropout())
    local imgSize = 0.25 * dimensions[2] * dimensions[3]
    model:add(nn.View(256 * imgSize))
  
    model:add(nn.Linear(256 * imgSize, 1024))
    model:add(nn.BatchNormalization(1024))
    model:add(activation())
    model:add(nn.Dropout())
  
    model:add(nn.Linear(1024, 1024))
    model:add(nn.BatchNormalization(1024))
    model:add(activation())
    model:add(nn.Dropout())
  
    model:add(nn.Linear(1024, 2))
    model:add(nn.SoftMax())
  
    model = require('weight-init')(model, 'heuristic')
  
    return model
end


-- Create a new spatial transformer network.
-- From: https://github.com/Moodstocks/gtsrb.torch/blob/master/networks.lua
-- @param allow_rotation Whether to allow the spatial transformer to rotate the image.
-- @param allow_rotation Whether to allow the spatial transformer to scale (zoom) the image.
-- @param allow_rotation Whether to allow the spatial transformer to translate (shift) the image.
-- @param input_size Height/width of input images.
-- @param input_channels Number of channels of the image.
-- @param cuda Whether to activate cuda mode.
function models.createSpatialTransformer(allow_rotation, allow_scaling, allow_translation, input_size, input_channels, cuda)
    if cuda == nil then
        cuda = true
    end

    -- Get number of params and initial state
    local init_bias = {}
    local nbr_params = 0
    if allow_rotation then
        nbr_params = nbr_params + 1
        init_bias[nbr_params] = 0
    end
    if allow_scaling then
        nbr_params = nbr_params + 1
        init_bias[nbr_params] = 1
    end
    if allow_translation then
        nbr_params = nbr_params + 2
        init_bias[nbr_params-1] = 0
        init_bias[nbr_params] = 0
    end
    if nbr_params == 0 then
        -- fully parametrized case
        nbr_params = 6
        init_bias = {1,0,0,0,1,0}
    end

    -- Create localization network
    local net = nn.Sequential()
    net:add(nn.SpatialAveragePooling(2, 2, 2, 2))
    net:add(nn.SpatialConvolution(input_channels, 16, 3, 3, 1, 1, (3-1)/2))
    net:add(nn.LeakyReLU())
    net:add(nn.SpatialConvolution(16, 16, 3, 3, 1, 1, (3-1)/2))
    net:add(nn.LeakyReLU())
    net:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    local newHeight = input_size * 0.5 * 0.5
    net:add(nn.View(16 * newHeight * newHeight))
    net:add(nn.Linear(16 * newHeight * newHeight, 64))
    net:add(nn.LeakyReLU())
    local classifier = nn.Linear(64, nbr_params)
    net:add(classifier)
    
    net = require('weight-init')(net, 'heuristic')
    -- Initialize the localization network (see paper, A.3 section)
    classifier.weight:zero()
    classifier.bias = torch.Tensor(init_bias)
    
    local localization_network = net

    -- Create the actual module structure
    -- branch1 is basically an identity matrix
    -- branch2 estimates the necessary rotation/scaling/translation (above localization network)
    -- They both feed into the BilinearSampler, which transforms the image
    local ct = nn.ConcatTable()
    local branch1 = nn.Sequential()
    branch1:add(nn.Transpose({3,4},{2,4}))
    -- see (1) below
    if cuda then
        branch1:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
    end
    local branch2 = nn.Sequential()
    branch2:add(localization_network)
    branch2:add(nn.AffineTransformMatrixGenerator(allow_rotation, allow_scaling, allow_translation))
    branch2:add(nn.AffineGridGeneratorBHWD(input_size, input_size))
    -- see (1) below
    if cuda then
        branch2:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))
    end
    ct:add(branch1)
    ct:add(branch2)

    local st = nn.Sequential()
    st:add(ct)
    local sampler = nn.BilinearSamplerBHWD()
    -- (1)
    -- The sampler lead to non-reproducible results on GPU
    -- We want to always keep it on CPU
    -- This does no lead to slowdown of the training
    if cuda then
        sampler:type('torch.FloatTensor')
        -- make sure it will not go back to the GPU when we call
        -- ":cuda()" on the network later
        sampler.type = function(type) return self end
        st:add(sampler)
        st:add(nn.Copy('torch.FloatTensor','torch.CudaTensor', true, true))
    else
        st:add(sampler)
    end
    st:add(nn.Transpose({2,4},{3,4}))

    return st
end

return models
