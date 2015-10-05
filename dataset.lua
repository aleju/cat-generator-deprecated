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
dataset.nbChannels = 1

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
    
    print(string.format("<dataset> Loaded %d filepaths", #files))
    
    dataset.paths = files
end

--[[
function dataset.loadImages(startAt, count)
    local endBefore = startAt + count

    local images = dataset.loadImagesFromDirs(dataset.dirs, dataset.fileExtension, startAt, count, false)
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

    setmetatable(result, {__index = function(self, index)
        return self.scaled[index]
    end})

    print(string.format('<dataset> loaded %d examples', N))

    return result
end
--]]

-- Loads a defined number of randomly selected images from
-- the cached paths (cached in loadPaths()).
-- @param count Number of random images.
-- @return List of Tensors
function dataset.loadRandomImages(count)
    --local images = dataset.loadRandomImagesFromDirs(dataset.dirs, dataset.fileExtension, count)
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

    setmetatable(result, {__index = function(self, index)
        return self.scaled[index]
    end})

    print(string.format('<dataset> loaded %d random examples', N))

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

    if OPT.gpu then
    --    torch.setdefaulttensortype('torch.FloatTensor')
    end
    shuffle = torch.randperm(#dataset.paths)
    
    
    images = {}
    for i=1,math.min(shuffle:size(1), count) do
       -- load each image
       table.insert(images, image.load(dataset.paths[shuffle[i]], dataset.nbChannels, "float"))
    end
    
    if OPT.gpu then
    --    torch.setdefaulttensortype('torch.CudaTensor')
    end
    
    return images
end

--[[
function dataset.loadRandomImagesFromDirs(dirs, ext, count)
    local files = {}

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
    end
    
    -- Check files
    if #files == 0 then
        error('given directories dont contain any files of type: ' .. ext)
    end
    
    
    shuffle = torch.randperm(#files)
    images = {}
    for i=1,math.min(shuffle:size(1), count) do
       -- load each image
       table.insert(images, image.load(files[ shuffle[i] ], dataset.nbChannels, "float"))
    end
    
    return images
end
--]]

--[[
function dataset.loadImagesFromDirs(dirs, ext, startAt, count, doSort)
    -- code from: https://github.com/andresy/torch-demos/blob/master/load-data/load-images.lua
    local files = {}

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
    
    ----------------------------------------------------------------------
    -- 3. Sort file names

    -- We sort files alphabetically, it's quite simple with table.sort()
    if doSort then
        table.sort(files, function (a,b) return a < b end)
    end

    ----------------------------------------------------------------------
    -- Extract requested files from startAt to startAt+count
    local filesExtracted = {}
    local endAt = math.min(startAt+count-1, #files)
    for i=startAt, endAt do
        filesExtracted[#filesExtracted+1] = files[i]
    end
    
    ----------------------------------------------------------------------
    -- 4. Finally we load images

    -- Go over the file list:
    local images = {}
    for i,file in ipairs(filesExtracted) do
       -- load each image
       table.insert(images, image.load(file, dataset.nbChannels, "float"))
    end
    
    return images
end
--]]

return dataset
