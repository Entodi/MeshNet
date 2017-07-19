require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'optim'

require 'string'
npy4th = require 'npy4th'
utils = require 'utils'
require 'randomkit'

-------------------------------------------------------------------------

if not opt then
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Training')
  cmd:text()
  cmd:text('Options:')
  cmd:option('-path2dir', './saved_models/', 'Path to save model directory')
  cmd:option('-trainFold', 'train_gmwm_fold.txt', 'File with train fold')
  cmd:option('-validFold', 'valid_gmwm_fold.txt', 'File with validation fold')
  cmd:option('-nModal', 1, 'The number of modalities')
  cmd:option('-nTrainSubCubesPerBrain', 100000, 'Number of sub-cubes for train')
  cmd:option('-nValidSubCubesPerBrain', 1024, 'Number of sub-cubes for valid')
  cmd:option('-nTrainPerEpoch', 2048, 'Train subvolumes per epoch')
  cmd:option('-nEpochs', 1000, 'Number of epochs')

  cmd:option('-xLen', 68, 'sub-cube side length of brain data cube by x')
  cmd:option('-yLen', 68, 'sub-cube side length of brain data cube by y')
  cmd:option('-zLen', 68, 'sub-cube side length of brain data cube by z')

  cmd:option('-MRI_xLen', 256, 'MRI volume x-axis side length')
  cmd:option('-MRI_yLen', 256, 'MRI volume y-axis side length')
  cmd:option('-MRI_zLen', 256, 'MRI volume z-axis side length')

  cmd:option('-batchSize', 1, 'mini-batch size')
  cmd:option('-modelFile', './models/gmwm_model_vdp.lua', 'File with architecture')
  cmd:option('-optimization', 'adam', 'optimization method: SGD')
  cmd:option('-loss', 'VolumetricCrossEntropyCriterion', 
    'type of loss function to minimize: VolumetricCrossEntropyCriterion')
  cmd:option('-weightInit', 'identity', 
    'Weight initilization of network layers: xavier or identity')
  cmd:option('-seed', 123, 'seed')
  cmd:option('-gpuDevice', 1, 'GPU device id (starting from 1)')
  
  cmd:option('-sampleType', 'gaussian', 'Distribution for sampling subvolumes. gaussian')
  
  cmd:text()
  opt = cmd:parse(arg or {})
  print(opt)
end

-------------------------------------------------------------------------
torch.manualSeed(opt.seed)
cutorch.setDevice(opt.gpuDevice)
print('Training on ', opt.gpuDevice)
-------------------------------------------------------------------------
local net = {}
print 'Loading net'
net = dofile(opt.modelFile)
-------------------------------------------------------------------------
print 'Weight initilization'
if opt.weightInit == 'identity' then
  identity3x3x3 = torch.FloatTensor(3,3,3):fill(0)
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
  net:cuda()
else
  print (opt.weightInit .. ' is not implemented')
end

if net then
  parameters, gradParameters = net:getParameters()
end
-------------------------------------------------------------------------
print 'Configuring optimizer'
if opt.optimization == 'adam' then
  optimMethod = optim.adam
else
  error('Unknown optimization method')
end
--------------------------------------------------------------------------
print 'Configuring Loss'
if opt.loss == 'VolumetricCrossEntropyCriterion' then
  criterion = cudnn.VolumetricCrossEntropyCriterion()
else
  error('Unknown Loss')
end
criterion = criterion:cuda()
print 'The loss function:'
print(criterion)
---------------------------------------------------------------------------
print 'Training procedure'

function train(data, coordinates, amount, nPerBrain, batchSize, nModal)
  net:training()
  print 'Training'
  local time = sys.clock()
  local overall_train_loss = torch.Tensor(amount / batchSize)
  local i = 1
  for t = 1, amount, batchSize do
    local inputs, targets = utils.create_cuda_mini_batch(
      data, coordinates, batchSize, subsizes, nModal, nPerBrain, 0, 'train')
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


function valid(data, coordinates, amount, nPerBrain, batchSize, nModal)
  net:evaluate()
  print 'Validating'
  local time = sys.clock()
  local overall_valid_loss = torch.Tensor(amount / batchSize)
  local k = 1
  for t = 1, amount, batchSize do
    local inputs, targets = utils.create_cuda_mini_batch(
      data, coordinates, batchSize, subsizes, nModal, nPerBrain, t, 'valid')
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

---------------------------------------------------------------------------
-- structure to save loss
lossInfo = {
  epochs = {},
  trainMean = {},
  trainStd = {},
  validMean = {},
  validStd = {}
}

modelName = utils.model_name_generator()
modelName = string.format('%s%s/', opt.path2dir, modelName)
logsFilename = string.format(
  '%s/logs%s', modelName,'.csv')
lossPlotFilename = string.format(
  '%s/plot%s', modelName,'.png')
modelFilenameAdd = ''
subsizes = {opt.zLen, opt.yLen, opt.xLen}
extend = {{opt.zLen/2, opt.zLen/2}, {opt.yLen/2, opt.yLen/2}, {opt.xLen/2, opt.xLen/2}}
sizes = {opt.nModal, opt.MRI_zLen + extend[1][1] + extend[1][2], opt.MRI_yLen + extend[2][1] + extend[2][2], opt.MRI_xLen + extend[3][1] + extend[3][2]}
mean = {opt.MRI_zLen/2,  opt.MRI_yLen/2,  opt.MRI_xLen/2}
std = {opt.MRI_zLen/6 + 8, opt.MRI_yLen/6 + 8, opt.MRI_xLen/6 + 8}


print 'Loading data'
local trainFold = utils.lines_from(opt.trainFold)
local validFold = utils.lines_from(opt.validFold)
local trainData = utils.load_brains(trainFold, opt.nModal, opt.MRI_zLen, opt.MRI_yLen, opt.MRI_xLen, extend)
local validData = utils.load_brains(validFold, opt.nModal, opt.MRI_zLen, opt.MRI_yLen, opt.MRI_xLen, extend)

-- makes training and validation dataset times of batch size
local trainAmount = opt.nTrainPerEpoch - opt.nTrainPerEpoch % opt.batchSize
local validAmount = #validFold * (opt.nValidSubCubesPerBrain - opt.nValidSubCubesPerBrain % opt.batchSize)

print ('Dataset per epoch: train: ', trainAmount, ' valid: ', validAmount)

print 'Creating validation coordinates'
local validDataset = utils.create_dataset_coords(sizes, opt.nValidSubCubesPerBrain, subsizes, extend, opt.sampleType, mean, std)

os.execute("mkdir " .. modelName)

print 'Start training'
for i = 1, opt.nEpochs do
  print('Epoch #' .. i)
  table.insert(lossInfo.epochs, i)
  print 'Creating Training coordinates'
  trainDataset = utils.create_dataset_coords(sizes, opt.nTrainSubCubesPerBrain, subsizes, extend, opt.sampleType, mean, std)
  -- training
  train(trainData, trainDataset, trainAmount, opt.nTrainPerEpoch, opt.batchSize, opt.nModal)
  -- validating
  valid(validData, validDataset, validAmount, opt.nValidSubCubesPerBrain, opt.batchSize, opt.nModal)
  -- saving model
  torch.save(modelName .. modelFilenameAdd .. 'model_' .. i .. '.t7', net:clearState())
  -- saving tables with loss
  utils.save_loss_info_2_csv(lossInfo, logsFilename)
  print('train: ',lossInfo.trainMean[i], lossInfo.trainStd[i])
  print('valid: ',lossInfo.validMean[i], lossInfo.validStd[i])

  collectgarbage()
end
