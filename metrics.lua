require 'string'
utils = require 'utils'
npy4th = require 'npy4th'
---------------------------------------------------------------
-- for operating strings as arrays
getmetatable('').__call = string.sub
---------------------------------------------------------------
if not opt then
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Prediction')
  cmd:text()
  cmd:text('Options:')
  cmd:option('-foldList', '', 'Name of file with fold of brains')
  cmd:option('-modelFile', '', 'Name of file with model weights')
  cmd:option('-outputFile', 'metrics.csv', 'Output metrics csv name')
  cmd:option('-xLen', 68, 'sub-cube side length of brain data cube by x')
  cmd:option('-yLen', 68, 'sub-cube side length of brain data cube by y')
  cmd:option('-zLen', 68, 'sub-cube side length of brain data cube by z')
  cmd:option('-batchSize', 1, 'Mini-batch size (model-dependent')
  cmd:option('-nSubvolumes', 1024, 'Number of subvolumes')
  cmd:option('-gpuDevice', 1, 'GPU device id (starting from 1)')
  cmd:option('-predType', 'maxclass', 'maxclass or maxsoftmax')
  cmd:option('-seed', 123, 'seed')
  cmd:option('-nClasses', 3, 'Number of classes in labels')
  cmd:option('-sampleType', 'gaussian', 'Distribution for sampling subvolumes. gaussian')
  cmd:option('-std', {50, 50, 50}, 'std of gaussian sampling')
  cmd:text()
  opt = cmd:parse(arg or {})
end
print(opt)
---------------------------------------------------------------
-- set seed
torch.manualSeed(opt.seed)
-- set GPU device
cutorch.setDevice(opt.gpuDevice)
-- load brains
local foldList = utils.lines_from(opt.foldList)
print (foldList)
local brains = utils.load_brains(foldList)
-- load model weights
local model = utils.load_prediction_model(opt.modelFile)
-- calculate metrics
local brain_metrics = {}
for i = 1, #brains do
  print('Loading ' .. i .. 'th brain ' .. foldList[i])
  local segmentation, time = utils.predict(brains[i], model, opt)
  brain_metrics[i] = utils.calculate_metrics(segmentation, brains[i].target, opt.nClasses)
  brain_metrics[i].time = time
  collectgarbage()
end
print ('Saving metrics')
utils.save_metrics(foldList, brain_metrics, opt.nClasses, opt.outputFile)
