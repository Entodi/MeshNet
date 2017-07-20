require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'optim'
require 'string'
utils = require 'utils'

-------------------------------------------------------------------------

if not opt then
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Training')
  cmd:text()
  cmd:text('Options:')
  cmd:option('-path2dir', './saved_models/', 'Path to save model directory')
  cmd:option('-trainFold', 'train_fold.txt', 'File with train fold')
  cmd:option('-validFold', 'valid_fold.txt', 'File with validation fold')
  cmd:option('-nModal', 1, 'The number of modalities')
  cmd:option('-nTrainSubCubesPerBrain', 100000, 'Number of sub-cubes to generate to choose for train')
  cmd:option('-nValidSubCubesPerBrain', 1024, 'Number of sub-cubes for valid')
  cmd:option('-nTrainPerEpoch', 2048, 'Train subvolumes per epoch')
  cmd:option('-nEpochs', 1000, 'Number of epochs')

  cmd:option('-xLen', 68, 'sub-cube side length of brain data cube by x')
  cmd:option('-yLen', 68, 'sub-cube side length of brain data cube by y')
  cmd:option('-zLen', 68, 'sub-cube side length of brain data cube by z')

  cmd:option('-batchSize', 1, 'mini-batch size')
  cmd:option('-modelFile', './models/vdp_model.lua', 'File with architecture')
  cmd:option('-optimization', 'adam', 'optimization method: SGD')
  cmd:option('-loss', 'VolumetricCrossEntropyCriterion', 
    'type of loss function to minimize: VolumetricCrossEntropyCriterion')
  cmd:option('-weightInit', 'identity', 
    'Weight initilization of network layers: identity')
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
  utils.init_identity(net)
else
  print (opt.weightInit .. ' is not implemented')
end

net:cuda()
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

print 'Loading data'
local trainFold = utils.lines_from(opt.trainFold)
local validFold = utils.lines_from(opt.validFold)
local trainData = utils.load_brains(trainFold, extend)
local validData = utils.load_brains(validFold, extend)

local sizes = trainData[1].input:size()
-- define subvolumes sizes
local subsizes = {sizes[1], opt.zLen, opt.yLen, opt.xLen}
-- define mean and std for gaussian sampling
local mean = opt.mean or {sizes[2]/2,  sizes[3]/2,  sizes[4]/2}
local std = opt.std or {sizes[2]/6, sizes[3]/6, sizes[4]/6}

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
  utils.train(net, criterion, optimMethod, trainData, trainDataset, trainAmount, opt.nTrainPerEpoch, opt.batchSize, subsizes, lossInfo)
  -- validating
  utils.valid(net, criterion, validData, validDataset, validAmount, opt.nValidSubCubesPerBrain, opt.batchSize, subsizes, lossInfo)
  -- saving model
  torch.save(modelName .. modelFilenameAdd .. 'model_' .. i .. '.t7', net:clearState())
  -- saving tables with loss
  utils.save_loss_info_2_csv(lossInfo, logsFilename)
  print('train: ',lossInfo.trainMean[i], lossInfo.trainStd[i])
  print('valid: ',lossInfo.validMean[i], lossInfo.validStd[i])

  collectgarbage()
end
