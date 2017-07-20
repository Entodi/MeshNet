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
  cmd:option('-brainPath', '', 'Path to input brain directory')
  cmd:option('-modelFile', '', 'Name of file with model weights')
  cmd:option('-outputFile', 'segmentation.npy', 'Output segmentation name')
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
  cmd:text()
  opt = cmd:parse(arg or {})
end
print(opt)
---------------------------------------------------------------
-- set seed
torch.manualSeed(opt.seed)
-- set GPU device
cutorch.setDevice(opt.gpuDevice)
-- load brain
local brain = utils.load_brain(opt.brainPath)
-- load model weights
local model = utils.load_prediction_model(opt.modelFile)
-- make prediction
segmentation, time = utils.predict(brain, model, opt)
-- save prediction
npy4th.savenpy(opt.brainPath .. opt.outputFile, segmentation - 1)