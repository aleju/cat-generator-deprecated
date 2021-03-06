require 'torch'
require 'image'
require 'paths'

local dataset = {}

-- load data from these directories
dataset.dirs = {}
-- load only images with these file extensions
dataset.fileExtension = ""

-- expected original height/width of images
dataset.originalScale = 64
-- desired height/width of images
dataset.scale = 32
-- desired channels of images (1=grayscale, 3=color)
dataset.nbChannels = 3

-- cache for filepaths to all images
dataset.paths = nil

-- Set directories to load images from
-- @param dirs List of paths to directories
function dataset.setDirs(dirs)
  dataset.dirs = dirs
end

-- Set file extension that images to load must have
-- @param fileExtension the file extension of the images
function dataset.setFileExtension(fileExtension)
  dataset.fileExtension = fileExtension
end

-- Desired height/width of the images (will be resized if necessary)
-- @param scale The height/width of the images
function dataset.setScale(scale)
  dataset.scale = scale
end

-- Set desired number of channels for the images (1=grayscale, 3=color)
-- @param nbChannels The number of channels
function dataset.setNbChannels(nbChannels)
  dataset.nbChannels = nbChannels
end

-- Loads the paths of all images in the defined files
-- (with defined file extensions)
function dataset.loadPaths()
    local files = {}
    local dirs = dataset.dirs
    local ext = dataset.fileExtension

    for i=1, #dirs do
        local dir = dirs[i]
        -- Go over all files in directory. We use an iterator, paths.files().
        for file in paths.files(dir) do
            -- We only load files that match the extension
            if file:find(ext .. '$') then
                -- and insert the ones we care about in our table
                table.insert(files, paths.concat(dir,file))
            end
        end

        -- Check files
        if #files == 0 then
            error('given directory doesnt contain any files of type: ' .. ext)
        end
    end
    
    dataset.paths = files
end

-- Load images from the dataset.
-- @param startAt Number of the first image.
-- @param count Count of the images to load.
-- @return Table of images. You can call :size() on that table to get the number of loaded images.
function dataset.loadImages(startAt, count)
    local endBefore = startAt + count
    if dataset.paths == nil then
        dataset.loadPaths()
    end

    local N = math.min(count, #dataset.paths)
    local images = torch.FloatTensor(N, dataset.nbChannels, dataset.scale, dataset.scale)
    for i=1,N do
        local img = image.load(dataset.paths[i], dataset.nbChannels, "float")
        img = image.scale(img, dataset.scale, dataset.scale)
        images[i] = img
    end

    local result = {}
    result.data = images
    
    function result:size()
        return N
    end

    setmetatable(result, {
        __index = function(self, index) return self.data[index] end,
        __len = function(self) return self.data:size(1) end
    })

    return result
end

-- Loads a defined number of randomly selected images from
-- the cached paths (cached in loadPaths()).
-- @param count Number of random images.
-- @return List of Tensors
function dataset.loadRandomImages(count)
    local images = dataset.loadRandomImagesFromPaths(count)
    local data = torch.FloatTensor(#images, dataset.nbChannels, dataset.scale, dataset.scale)
    for i=1, #images do
        data[i] = image.scale(images[i], dataset.scale, dataset.scale)
    end

    local N = data:size(1)
    local result = {}
    result.scaled = data

    function result:size()
        return N
    end

    function result:normalize(mean, std)
        mean, std = NN_UTILS.normalize(result.scaled, mean, std)
        return mean, std
    end

    setmetatable(result, {
        __index = function(self, index) return self.scaled[index] end,
        __len = function(self) return self.scaled:size(1) end
    })

    return result
end

-- Loads randomly selected images from the cached paths.
-- TODO: merge with loadRandomImages()
-- @param count Number of images to load
-- @returns List of Tensors
function dataset.loadRandomImagesFromPaths(count)
    if dataset.paths == nil then
        dataset.loadPaths()
    end

    local shuffle = torch.randperm(#dataset.paths)    
    
    local images = {}
    for i=1,math.min(shuffle:size(1), count) do
       -- load each image
       table.insert(images, image.load(dataset.paths[shuffle[i]], dataset.nbChannels, "float"))
    end
    
    return images
end

return dataset
