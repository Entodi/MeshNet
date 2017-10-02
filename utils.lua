require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'string'
require 'csvigo'
require 'image'
require 'math'
require 'randomkit'
require 'Dataframe'
require 'os'
npy4th = require 'npy4th'

local utils = {}

function utils.init_identity(net)
  --[[
  Inits with identity weights

  Args:
    net: network model
  ]]
  local identity3x3x3 = torch.FloatTensor(3,3,3):fill(0)
  identity3x3x3[{1,1,1}] = 1

  local n_layer = 1
  for i = 1, #net.modules do
    local m = net.modules[i]
    if m.__typename == 'nn.VolumetricDilatedConvolution' then
      m.bias = torch.FloatTensor(m.bias:size()):fill(0)
      m.bias = randomkit.normal(m.bias, 0, 2.0/(m.nInputPlane + m.nOutputPlane))
      for out_f = 1, m.nOutputPlane do
        for in_f = 1, m.nInputPlane do
          if n_layer ~= 8 then
            t = torch.FloatTensor(3,3,3):fill(0)
            t = randomkit.normal(t, 0, 2.0/(m.nInputPlane + m.nOutputPlane))
            t[{1,1,1}] = 1 + randomkit.normal(0, 2.0/(m.nInputPlane + m.nOutputPlane))
            m.weight[{out_f, in_f, {}, {}, {}}] = t:clone()
          else
            m.weight[{out_f, in_f, {}, {}, {}}] = 1 + randomkit.normal(0, 2.0/(m.nInputPlane + m.nOutputPlane))
          end
        end
      end
      n_layer = n_layer + 1
    end
  end
end

function utils.init_xavier(net)
  --[[
  Inits with xavier weights

  Args:
    net: network model
  ]]
  local identity3x3x3 = torch.FloatTensor(3,3,3):fill(0)
  identity3x3x3[{1,1,1}] = 1

  local n_layer = 1
  for i = 1, #net.modules do
    local m = net.modules[i]
    if m.__typename == 'nn.VolumetricDilatedConvolution' then
      m.bias = torch.FloatTensor(m.bias:size()):fill(0)
      m.bias = randomkit.normal(m.bias, 0, 2.0/(m.nInputPlane + m.nOutputPlane))
      for out_f = 1, m.nOutputPlane do
        for in_f = 1, m.nInputPlane do
          if n_layer ~= 8 then
            t = torch.FloatTensor(3,3,3):fill(0)
            t = randomkit.normal(t, 0, 2.0/(m.nInputPlane + m.nOutputPlane))
            m.weight[{out_f, in_f, {}, {}, {}}] = t:clone()
          else
            m.weight[{out_f, in_f, {}, {}, {}}] = randomkit.normal(0, 2.0/(m.nInputPlane + m.nOutputPlane))
          end
        end
      end
      n_layer = n_layer + 1
    end
  end
end

function utils.train(net, criterion, optimMethod, data, coordinates, amount, nPerBrain, batchSize, subsizes, lossInfo)
  --[[
  Inits with identity weights

  Args:
    net: network model 
    criterion: criterion
    optimMethod: optimization method
    data: data with brains and labels
    coordinates: generated coordinate grid for subvolumes
    amount: amount of subvolumes per epoch
    nPerBrain: amount of subvolumes generated per brain,
    batchSize: mini-batch size
    subsizes: subvolumes sizes
    lossInfo: table with mean and std of loss function values per epoch
  ]]
  net:training()
  print 'Training'
  local time = sys.clock()
  local overall_train_loss = torch.Tensor(amount / batchSize)
  local i = 1
  for t = 1, amount, batchSize do
    local inputs, targets = utils.create_cuda_mini_batch(
      data, coordinates, batchSize, subsizes, nPerBrain, 0, 'train')
    local trainFunc = function(x)
      if x ~= parameters then
        parameters:copy(x)
      end
      gradParameters:zero()
      local outputs = net:forward(inputs)
      local loss = criterion:forward(outputs, targets)
      overall_train_loss[i] = loss
      i = i + 1
      local df_do = criterion:backward(outputs, targets)
      net:backward(inputs, df_do)
      return loss, gradParameters
    end
    optimMethod(trainFunc, parameters, optimState)
  end
  table.insert(lossInfo.trainMean, overall_train_loss:mean())
  table.insert(lossInfo.trainStd, overall_train_loss:std())
  time = sys.clock() - time
  print("time to learn 1 epoch = " .. (time * 1000) .. 'ms')
end

function utils.valid(net, criterion, data, coordinates, amount, nPerBrain, batchSize, subsizes, lossInfo)
  --[[
  Inits with identity weights

  Args:
    net: network model 
    criterion: criterion
    data: data with brains and labels
    coordinates: generated coordinate grid for subvolumes
    amount: amount of subvolumes per epoch
    nPerBrain: amount of subvolumes generated per brain,
    batchSize: mini-batch size
    subsizes: subvolumes sizes
    lossInfo: table with mean and std of loss function values per epoch
  ]]
  net:evaluate()
  print 'Validating'
  local time = sys.clock()
  local overall_valid_loss = torch.Tensor(amount / batchSize)
  local k = 1
  for t = 1, amount, batchSize do
    local inputs, targets = utils.create_cuda_mini_batch(
      data, coordinates, batchSize, subsizes, nPerBrain, t, 'valid')
    local outputs = net:forward(inputs)
    overall_valid_loss[k] = criterion:forward(outputs, targets)
    k = k + 1
  end
  table.insert(lossInfo.validMean, overall_valid_loss:mean())
  table.insert(lossInfo.validStd, overall_valid_loss:std())
  time = sys.clock() - time
  time = time / amount
  print("time to valid 1 sample = " .. (time*1000) .. 'ms')
end


function utils.calculate_metrics(prediction, target, nClasses)
  --[[
  Calculates metrics from prediction

  Args:
    prediction: model prediction
    target: ground thruth labels
    nClasses: number of classes
  Returns:
    brain_metrics: calculated metrics
  ]]
  local splitted_output = utils.split_classes(prediction, nClasses)  
  local splitted_target = utils.split_classes(target, nClasses)
  local brain_metrics = {}
  brain_metrics.f1_score = {}
  brain_metrics.avd = {}
  for c = 1, nClasses do
    brain_metrics.f1_score[c] = 
      utils.f1_score(splitted_output[c], splitted_target[c])
    brain_metrics.avd[c] = 
      utils.average_volumetric_difference(splitted_output[c], splitted_target[c])
  end
  return brain_metrics
end

function utils.save_metrics(foldList, brain_metrics, nClasses, outputFile)
  --[[
  Save metrics to csv

  Args:
    brain_metrics: table with metrics
    nClasses: number of classes
    outputFile: filename to save
  ]]
  local model_csv = {}
  model_csv['brain'] = {}
  model_csv['time'] = {}
  local first_run = true
  for c = 1, nClasses do
    model_csv['f1_' .. tostring(c)] = {}
    model_csv['avd_' .. tostring(c)] = {}
    for b = 1, #brain_metrics do
      if first_run then
        model_csv.brain[b] = foldList[b]
        model_csv.time[b] = brain_metrics[b].time 
      end
      model_csv['f1_' .. tostring(c)][b] = brain_metrics[b].f1_score[c]
      model_csv['avd_' .. tostring(c)][b] = brain_metrics[b].avd[c]  
    end
    first_run = false
  end
  local df = Dataframe()
  print (model_csv)
  df:load_table{data=Df_Dict(model_csv)}
  df:to_csv(outputFile)
end

function utils.predict(brain, model, opt)
  --[[
  Predicts segmentation.

  Args:
    brain: brain with input and target (loaded using utils.load_brain)
    model: model weights
    opt: table with options
  Returns:
    segmentation: predicted segmentation
  ]]
  -- define gathering function
  local gather_function = {}
  if opt.predType == 'maxclass' then
    gather_function = utils.gather_maxclass
  elseif opt.predType == 'maxsoftmax' then
    gather_function = utils.gather_maxsoftmax
  else
    print('Invalid prediction type. Should be maxclass or maxsoftmax.')
  end
  -- define extend
  local extend = opt.extend or {{0, 0}, {0, 0}, {0, 0}}
  -- define volume sizes
  local sizes = brain.input:size()
    -- define subvolumes sizes
  local subsizes = {sizes[1], opt.zLen, opt.yLen, opt.xLen}
  -- define mean and std for gaussian sampling
  local mean = opt.mean or {sizes[2]/2,  sizes[3]/2,  sizes[4]/2}
  local std = opt.std or {sizes[2]/6 + 8, sizes[3]/6 + 8, sizes[4]/6 + 8}
  -- define softmax layer
  local softmax = cudnn.VolumetricLogSoftMax():cuda()
  -- correct number of subvvolumes based of batchsize
  opt.nSubvolumes = opt.nSubvolumes - opt.nSubvolumes % opt.batchSize
  -- define coordinate grid
  local coords_grid = coords_grid or utils.create_dataset_coords(
    sizes, opt.nSubvolumes, subsizes, extend, opt.sampleType, mean, std)
  -- define output segmentation
  local segmentation = {}
  if opt.predType == 'maxclass' then
    segmentation = torch.IntTensor(opt.nClasses, 
      sizes[2] + extend[1][1] + extend[1][2], 
      sizes[3] + extend[2][1] + extend[2][2], 
      sizes[4] + extend[3][1] + extend[3][2]):fill(0)
  elseif opt.predType == 'maxsoftmax' then
    segmentation = torch.DoubleTensor(opt.nClasses,
      sizes[2] + extend[1][1] + extend[1][2],
      sizes[3] + extend[2][1] + extend[2][2],
      sizes[4] + extend[3][1] + extend[3][2]):fill(0)
  else
    print('Invalid prediction type. Should be maxclass or maxsoftmax.')
  end
  -- predict
  local time = sys.clock()
  for i = 1, #coords_grid, opt.batchSize do
    local inputs, targets = utils.create_cuda_mini_batch(
      {brain}, coords_grid, opt.batchSize, 
      subsizes, opt.nSubvolumes, i, 'test')
    local outputs = model:forward(inputs)
    outputs = softmax:forward(outputs)
    segmentation = gather_function(outputs, segmentation, coords_grid, i)
  end
  local maxs, segmentation = torch.max(segmentation, 1)
  time = sys.clock() - time
  print (time, 'seconds')
  brain = utils.reduceData(brain, extend)
  segmentation = utils.reduceOutput(segmentation, extend)[1]
  return segmentation, time
end

function utils.load_brains(pathes, extend, inputFiles, labelFile)
  --[[
  Load brains from fold.

  Args:
    pathes: table with pathes to brains directories
    inputFiles: filenames with input images (for example: 'T1.npy', 'T2.npy'} for multi modal case)
    labelFile: filename with labels
    extend: table of extensions of MRI image for every axis from left and right sides (Example table to extend from every side of axises MRI image by 10: {{10, 10}, {10, 10}, {10, 10}})
  Returns:
    data: table with brains
  ]]
  if #pathes == 0 then
    print 'No pathes to brains directories'
    return {}
  end
  inputFiles = inputFiles or {'T1.npy'}
  labelFile = labelFile or 'gm_wm.npy'
  extend = extend or {{0, 0}, {0, 0}, {0, 0}}
  local data = {}
  for i = 1, #pathes do
    data[i] = utils.load_brain(pathes[i], extend, inputFiles, labelFile)
  end
  return data
end

function utils.load_brain_nolabel(path, extend, inputFiles)
    --[[
    Load brains from fold.

    Args:
      path: path to brain directory
      inputFiles: filenames with input images (for example: 'T1.npy', 'T2.npy'} for multi modal case)
      extend: table of extensions of MRI image for every axis from left and right sides (Example table to extend from every side of axises MRI image by 10: {{10, 10}, {10, 10}, {10, 10}})
    Returns:
      data: brain data
    ]]
    inputFiles = inputFiles or {'T1.npy'}
    extend = extend or {{0, 0}, {0, 0}, {0, 0}}
    local data = {}
    for j = 1, #inputFiles do
      local t = npy4th.loadnpy(path .. inputFiles[j]):float()
      -- scale to unit interval
      t = (t - t:min()) / (t:max() - t:min())
      if j == 1 then
        data.input = torch.FloatTensor(#inputFiles, t:size()[1], t:size()[2], t:size()[3])
      end
      data.input[{j, {}, {}, {}}] = t
    end
    data = utils.extendData(data, extend)
    return data
end

function utils.load_brain(path, extend, inputFiles, labelFile)
    --[[
    Load brains from fold.

    Args:
      path: path to brain directory
      inputFiles: filenames with input images (for example: 'T1.npy', 'T2.npy'} for multi modal case)
      labelFile: filename with labels
      extend: table of extensions of MRI image for every axis from left and right sides (Example table to extend from every side of axises MRI image by 10: {{10, 10}, {10, 10}, {10, 10}})
    Returns:
      data: brain data
    ]]
    inputFiles = inputFiles or {'T1.npy'}
    labelFile = labelFile or 'gm_wm.npy'
    extend = extend or {{0, 0}, {0, 0}, {0, 0}}
    local data = {}
    for j = 1, #inputFiles do
      local t = npy4th.loadnpy(path .. inputFiles[j]):float()
      -- scale to unit interval
      t = (t - t:min()) / (t:max() - t:min())
      if j == 1 then
        data.input = torch.FloatTensor(#inputFiles, t:size()[1], t:size()[2], t:size()[3])
      end
      data.input[{j, {}, {}, {}}] = t
    end
    data.target = {}
    data.target = npy4th.loadnpy(path .. labelFile):int()
    -- torch labels start from 1, not from 0
    data.target = data.target:add(1)
    data = utils.extendData(data, extend)
    return data
end

function utils.nooverlapCoordinates(sizes, subsizes, extend)
  --[[
  Creates nonoverlap grid

  Args:
    sizes: MRI image side length
    subsizes: subvolume's side lengths
    extend: table of extensions of MRI image for every axis from left and right sides (Example table to extend from every side of axises MRI image by 10: {{10, 10}, {10, 10}, {10, 10}})
  Return:
    coords: table of nonoverlap grid coordinates
  ]]
  local coords = {}
  local k = 1
  for z1 = 1 + extend[1][1], sizes[2] - extend[1][2] - subsizes[2] + 1, subsizes[2] do
    for y1 = 1 + extend[2][1], sizes[3] - extend[2][2] - subsizes[3] + 1, subsizes[3] do
      for x1 = 1 + extend[3][1], sizes[4] - extend[3][2] - subsizes[4] + 1, subsizes[4] do
        coords[k] = {}
        coords[k].z1 = z1
        coords[k].y1 = y1
        coords[k].x1 = x1
        coords[k].z2 = coords[k].z1 + subsizes[2] - 1
        coords[k].y2 = coords[k].y1 + subsizes[3] - 1
        coords[k].x2 = coords[k].x1 + subsizes[4] - 1
        k = k + 1
      end
    end
  end
  return coords
end
  
function utils.gaussianCoordinates(sizes, subsizes, amount, mean, std)
  --[[
  Creates gaussian grid

  Args:
    sizes: MRI image side length
    subsizes: subvolume's side lengths
    amount: amount of subvolumes
    mean: table with mean values for every axis
    std: table with std values for every axis

  Return:
    coords: table of gaussian grid coordinates
  ]]
  mean = mean or {sizes[2] / 2,
    sizes[3] / 2,
    sizes[4] / 2}
  std = std or {50, 50, 50}
  local coords = {}
  local half_subsizes = {subsizes[2] / 2, subsizes[3] / 2, subsizes[4] / 2}
  local left_bound = {half_subsizes[1], half_subsizes[2], half_subsizes[3]}
  local right_bound = {sizes[2] - half_subsizes[1] + 1,
    sizes[3] - half_subsizes[2] + 1, sizes[4] - half_subsizes[3] + 1}
  local k = 1
  while k < amount + 1 do
    local rc = {
      torch.round(randomkit.normal(mean[1], std[1])), 
      torch.round(randomkit.normal(mean[2], std[2])),
      torch.round(randomkit.normal(mean[3], std[3]))
    }
    if rc[1] >= left_bound[1] and rc[2] >= left_bound[2] and rc[3] >= left_bound[3] and 
      rc[1] < right_bound[1] and rc[2] < right_bound[2] and rc[3] < right_bound[3] then
      coords[k] = {}
      coords[k].z1 = rc[1] - half_subsizes[1] + 1
      coords[k].y1 = rc[2] - half_subsizes[2] + 1
      coords[k].x1 = rc[3] - half_subsizes[3] + 1
      coords[k].z2 = coords[k].z1 + subsizes[2] - 1
      coords[k].y2 = coords[k].y1 + subsizes[3] - 1
      coords[k].x2 = coords[k].x1 + subsizes[4] - 1
      k = k + 1
    end
  end
  return coords
end

function utils.extendData(data, extend)
  --[[
  Extend MRI image with extend values. Equivalent to padding with zeros for input and 1 for target.

  Args:
    data: brain data with input and target
    extend:  extend: table of extensions of MRI image for every axis from left and right sides (Example table to extend from every side of axises MRI image by 10: {{10, 10}, {10, 10}, {10, 10}})

  Returns:
    extend_data: extended version of data
  ]]
  local extend_data = {}
  extend_data.input = torch.FloatTensor(data.input:size()[1], 
    data.input:size()[2] + extend[1][1] + extend[1][2], 
    data.input:size()[3] + extend[2][1] + extend[2][2], 
    data.input:size()[4] + extend[3][1] + extend[3][2]):fill(0)
  extend_data.input[{{},
    {1 + extend[1][1], data.input:size()[2] + extend[1][2]},
    {1 + extend[2][1], data.input:size()[3] + extend[2][2]},
    {1 + extend[3][1], data.input:size()[4] + extend[3][2]}}] = data.input
  if data.target then
    extend_data.target = torch.IntTensor(
      data.target:size()[1] + extend[1][1] + extend[1][2], 
      data.target:size()[2] + extend[2][1] + extend[2][2], 
      data.target:size()[3] + extend[3][1] + extend[3][2]):fill(1)
    extend_data.target[{
      {1 + extend[1][1], data.target:size()[1] + extend[1][2]},
      {1 + extend[2][1], data.target:size()[2] + extend[2][2]},
      {1 + extend[3][1], data.target:size()[3] + extend[3][2]}}] = data.target
  end
  return extend_data
end

function utils.reduceOutput(data, extend)
  --[[
  Reduces output with extend amount after extending input.

  Args:
    data: output from MeshNety
    extend:  extend: table of extensions of MRI image for every axis from left and right sides (Example table to extend from every side of axises MRI image by 10: {{10, 10}, {10, 10}, {10, 10}})

  Returns:
    reduced_data: reduces version of data
  ]]
  local reduced_data = {}
  reduced_data = torch.IntTensor(data:size()[1],
    data:size()[2] - extend[1][1] - extend[1][2], 
    data:size()[3] - extend[2][1] - extend[2][2], 
    data:size()[4] - extend[3][1] - extend[3][2]):fill(1)
  reduced_data = data[{{},
    {1 + extend[1][1], data:size()[2] - extend[1][2]},
    {1 + extend[2][1], data:size()[3] - extend[2][2]},
    {1 + extend[3][1], data:size()[4] - extend[3][2]}}]
  return reduced_data
end

function utils.reduceData(data, extend)
  --[[
  Reduces MRI image with extend amount after extending.

  Args:
    data: brain data with input and target
    extend:  extend: table of extensions of MRI image for every axis from left and right sides (Example table to extend from every side of axises MRI image by 10: {{10, 10}, {10, 10}, {10, 10}})

  Returns:
    reduced_data: reduces version of data
  ]]
  local reduced_data = {}
  reduced_data.input = torch.FloatTensor(data.input:size()[1], 
    data.input:size()[2] - extend[1][1] - extend[1][2], 
    data.input:size()[3] - extend[2][1] - extend[2][2], 
    data.input:size()[4] - extend[3][1] - extend[3][2]):fill(0)
  reduced_data.input = data.input[{{},
    {1 + extend[1][1], data.input:size()[2] - extend[1][2]},
    {1 + extend[2][1], data.input:size()[3] - extend[2][2]},
    {1 + extend[3][1], data.input:size()[4] - extend[3][2]}}]
  if data.target then
    reduced_data.target = torch.IntTensor(
      data.target:size()[1] - extend[1][1] - extend[1][2], 
      data.target:size()[2] - extend[2][1] - extend[2][2], 
      data.target:size()[3] - extend[3][1] - extend[3][2]):fill(1)
    reduced_data.target = data.target[{
      {1 + extend[1][1], data.target:size()[1] - extend[1][2]},
      {1 + extend[2][1], data.target:size()[2] - extend[2][2]},
      {1 + extend[3][1], data.target:size()[3] - extend[3][2]}}]
  end
  return reduced_data
end

function utils.load_prediction_model(modelFilename)
  --[[
  Loads model for CUDA and in evaluation state

  Args:
    modelFilename: name of a file with a model

  Returns:
    model: loaded model
  ]]
  local model = torch.load(modelFilename)
  model:cuda()
  model:evaluate()
  return model
end

function utils.split_classes(volume, nClasses)
  --[[
  Splits target or prediction tensor by class

  Args:
    volume: input volume
    nClasses: number of classes in volume

  Returns:
    split: table of volumes
  ]]
  local split = {}
  for id = 1, nClasses do
    split[id] = torch.IntTensor(volume:size()):fill(0)
    split[id] = split[id] + volume:eq(id):int()
  end
  return split
end

function utils.file_exists(file)
  --[[
  Checking file existence

  Args:
    file: name of a file

  Returns: true if exist, otherwise false 
  ]]
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

local function gather_prediction(outputCube, splitPrediction, coords)
  --[[
  Combine prediction from splitted prediction to volume by coordinates

  Args:
    outputCube: current full size prediction
    splitPrediction: split on classes prediction with split_classes function
    coords: coordinates for current subvolume

  Returns:
    outputCube: updated full size prediction
  ]]
  for id_class = 1, #splitPrediction do
    outputCube[{id_class, {coords.z1, coords.z2}, 
      {coords.y1, coords.y2}, {coords.x1, coords.x2}}] = 
        outputCube[{id_class, {coords.z1, coords.z2}, 
          {coords.y1, coords.y2}, {coords.x1, coords.x2}}]
        + splitPrediction[id_class]
  end
  return outputCube
end

function utils.gather_maxclass(dnnOutput, outputCube, coords, offset)
  --[[
  Combine prediction from splitted prediction to volume by coordinates based on majority voting

  Args:
    dnnOutput: output from MeshNet
    outputCube: current full size prediction
    splitPrediction: split on classes prediction with split_classes function
    coords: table of coordinates for current subvolume
    offset: current id of coordinates

  Returns:
    outputCube: 'histogram' of classes for majority voting
  ]]
  local max, inds = torch.max(dnnOutput, 2)
  for id = 1, dnnOutput:size()[1] do
    local splitPrediction = utils.split_classes(inds[{id, 1, {}, {}, {}}], dnnOutput:size()[2])
    outputCube = gather_prediction(
      outputCube, splitPrediction, coords[offset])
  end
  return outputCube
end

function utils.gather_maxsoftmax(dnnOutput, outputCube, coords, offset)
   --[[
  Combine softmax values from splitted prediction to volume by coordinates

  Args:
    dnnOutput: output from MeshNet
    outputCube: current full size prediction
    splitPrediction: split on classes prediction with split_classes function
    coords: table of coordinates for current subvolume
    offset: current id of coordinates

  Returns:
    outputCube: aggregated probability for majority voting
  ]]
  for id = 1, dnnOutput:size()[1] do
    local c = coords[offset] 
    outputCube[{{}, {c.z1, c.z2}, {c.y1, c.y2}, {c.x1, c.x2}}] = 
      outputCube[{{}, {c.z1, c.z2}, {c.y1, c.y2}, {c.x1, c.x2}}]
      + dnnOutput[id]:double()
  end
  return outputCube
end

function utils.lines_from(file)
  --[[
  Reads lines from file.

  Args:
    file: name of a file
  ]]
  if not utils.file_exists(file) then return {} end
  lines = {}
  for line in io.lines(file) do 
    lines[#lines + 1] = line
  end
  return lines
end

function utils.save_loss_info_2_csv(lossInfo, logsFilename)
  --[[
  Saves loss inforamtion to a csv file.

  Args:
    lossInfo: table with mean and stf of a loss from training and validating
    logsFilename: name of a csv file to save 
  ]]
  csvigo.save(logsFilename, lossInfo)
end

function utils.model_name_generator()
  --[[
  Creates model name based on a time.

  Returns:
    model name
  ]]
  datetime = os.date():gsub(' ', '_')
  return string.format('model_%s',
    datetime)
end


function utils.true_positive(predCube, targetCube)
  --[[
  Calculates number of True Positive voxels
  ]]
  return torch.sum(predCube:maskedSelect(targetCube:eq(1)))
end

function utils.true_negative(predCube, targetCube)
  --[[
  Calculates number of True Negative voxels
  ]]
  return torch.sum(predCube:maskedSelect(targetCube:eq(0)))
end

function utils.false_positive(predCube, targetCube)
  --[[
  Calculates number of False Positive voxels
  ]]
  return torch.sum(predCube:maskedSelect(targetCube:eq(0)):eq(1))
end

function utils.false_negative(predCube, targetCube)
  --[[
  Calculates number of False Negative voxels
  ]]
  return torch.sum(predCube:maskedSelect(targetCube:eq(1)):eq(0))
end

function utils.precision(predCube, targetCube)
  --[[
  Calculates precision

  Args:
    predCube: predicted volume
    targetCube: ground truth volume

  Returns: 
    precision
  ]]
  local tp = utils.true_positive(predCube, targetCube)
  local fp = utils.false_positive(predCube, targetCube)
  if tp + fp == 0 then 
    return 0
  else
    return tp / (tp + fp)
  end
end

function utils.recall(predCube, targetCube)
  --[[
  Calculates recall

  Args:
    predCube: predicted volume
    targetCube: ground truth volume

  Returns: 
    recall
  ]]
  local tp = utils.true_positive(predCube, targetCube)
  local fn = utils.false_negative(predCube, targetCube)
  if tp + fn == 0 then
    return 0
  else 
    return tp / (tp + fn)
  end
end

function utils.average_volumetric_difference(predCube, targetCube)
  --[[
  Calculates average_volumetric_difference

  Args:
    predCube: predicted volume
    targetCube: ground truth volume

  Returns: 
    average_volumetric_difference
  ]]
  local Vp = torch.sum(predCube:eq(1))
  local Vt = torch.sum(targetCube:eq(1))
  return torch.abs(Vp - Vt) / Vt
end

function utils.f1_score(predCube, targetCube)
  --[[
  Calculates f1_score (equivalent to DICE)

  Args:
    predCube: predicted volume
    targetCube: ground truth volume

  Returns: 
    f1_score
  ]]
  local p = utils.precision(predCube, targetCube)
  local r = utils.recall(predCube, targetCube)
  if p + r == 0 then
    return 0
  else 
    return 2 * p * r / (p + r)
  end
end

function utils.create_dataset_coords(sizes, amount, subsizes, extend, sample_type, mean, std)
  --[[
  Creates dataset of subvolumes using table of coordinates

  Args:
    sizes: size of input volumes
    amount: number of subvolumes
    subsizes: size of subvolumes
    extend:  extend: table of extensions of MRI image for every axis from left and right sides (Example table to extend from every side of axises MRI image by 10: {{10, 10}, {10, 10}, {10, 10}})
    sample_type: subvolume sampling distribution
    mean: table with mean values for every axis
    std: table with std values for every axis

  Returns:
    dataset_coords: table with coordinates with nonoverlap and sampleType grid
  ]]
  extend = extend or {{0, 0}, {0, 0}, {0, 0}}
  sample_type = sample_type or 'uniform'
  local dataset_coords = {} if sample_type == 'gaussian' then
    dataset_coords = utils.gaussianCoordinates(
      sizes, subsizes, amount, mean, std)  
  else
    print(sample_type .. ' is not implemented')
    os.exit()
  end
  noc = utils.nooverlapCoordinates(
    sizes, subsizes, extend)
  -- change first #noc coordinates with non-overlap
  for j = 1, #noc do
      dataset_coords[j] = noc[j]
  end
  return dataset_coords     
end

function utils.create_cuda_mini_batch(data, dataset_coords, batchSize, subsizes, nPerBrain, offset, mode)
  --[[
  Creates CUDA mini batch from data

  Args:
    data: table with brains
    dataset_coords: table with coordinates
    batchSize: size of mini-batch
    subsizes: subolume side lengths
    nPerBrain: number of volumes per brain (need just for mode valid)
    offset: current number of used for train
    mode: mode of creating batch ('train', 'valid' ot 'test')

  Returns: 
    inputs: CUDA batch of input
    targets: in 'train' and 'valid' mode returns CUDA batch, in 'test' mode returns empty table
  ]]
  local inputs, targets = utils.create_mini_batch(
    data, dataset_coords, batchSize, subsizes, nPerBrain, offset, mode)
  if mode ~= 'test' then
    return inputs:cuda(), targets:cuda()
  else
    return inputs:cuda(), {}
  end
end

function utils.create_mini_batch(data, dataset_coords, batchSize, subsizes, nPerBrain, offset, mode)
   --[[
  Creates mini batch from data

  Args:
    data: table with brains
    dataset_coords: table with coordinates
    batchSize: size of mini-batch
    subsizes: subolume side lengths
    nPerBrain: number of volumes per brain (need just for mode valid)
    offset: current number of used for train
    mode: mode of creating batch ('train', 'valid' ot 'test')

    Returns: 
    inputs: batch of input
    targets: in 'train' and 'valid' mode returns batch, in 'test' mode returns empty table
  ]]
  mode = mode or 'train'
  local inputs = torch.FloatTensor(
    batchSize, subsizes[1], subsizes[2], subsizes[3], subsizes[4])
  local targets = {}
  if mode ~= 'test' then
    targets = torch.IntTensor(
      batchSize, subsizes[2], subsizes[3], subsizes[4])
  end
  if mode == 'train' then
    for i = 1, batchSize do
      local bid = randomkit.randint(1, #data)
      local cid = randomkit.randint(1, #dataset_coords)
      inputs[{i, {}, {}, {}, {}}] = data[bid].input[{{},
        {dataset_coords[cid].z1, dataset_coords[cid].z2},
        {dataset_coords[cid].y1, dataset_coords[cid].y2},
        {dataset_coords[cid].x1, dataset_coords[cid].x2}
      }]
      targets[{i, {}, {}, {}}] = data[bid].target[{
        {dataset_coords[cid].z1, dataset_coords[cid].z2},
        {dataset_coords[cid].y1, dataset_coords[cid].y2},
        {dataset_coords[cid].x1, dataset_coords[cid].x2}
      }]
    end
  else
    local k = 1
    for i = offset, offset + batchSize - 1 do
      local bid = torch.floor((i - 1) / nPerBrain) + 1
      local cid = (i - 1) % nPerBrain + 1
      inputs[{k, {}, {}, {}, {}}] = data[bid].input[{{},
        {dataset_coords[cid].z1, dataset_coords[cid].z2},
        {dataset_coords[cid].y1, dataset_coords[cid].y2},
        {dataset_coords[cid].x1, dataset_coords[cid].x2},
      }]
      if mode ~= 'test' then
        targets[{k, {}, {}, {}}] = data[bid].target[{
          {dataset_coords[cid].z1, dataset_coords[cid].z2},
          {dataset_coords[cid].y1, dataset_coords[cid].y2},
          {dataset_coords[cid].x1, dataset_coords[cid].x2},
        }]
      end
      k = k + 1
    end
  end
  return inputs, targets
end

return utils
