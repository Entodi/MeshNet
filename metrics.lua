require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'Dataframe'
require 'string'
utils = require 'utils'
npy4th = require 'npy4th'

getmetatable('').__call = string.sub
---------------------------------------------------------------
if not opt then
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Metrics')
  cmd:text()
  cmd:text('Options:')
  cmd:option('-foldList', '', 'Name of file with fold of brains')
  cmd:option('-modelFile', '', 'Name of file with model weights')
  cmd:option('-outputFile', 'metrics.csv', 'Name of file to save csv')
  cmd:option('-filenameAdd', '', 'Addition to filename with prediction and csv file with metrics')
  cmd:option('-xLen', 68, 'sub-cube side length of brain data cube by x')
  cmd:option('-yLen', 68, 'sub-cube side length of brain data cube by y')
  cmd:option('-zLen', 68, 'sub-cube side length of brain data cube by z')
  cmd:option('-MRI_zLen', 256, 'MRI x side length')
  cmd:option('-MRI_yLen', 256, 'MRI y side length')
  cmd:option('-MRI_xLen', 256, 'MRI x side length')
  cmd:option('-batchSize', 1, 'Mini-batch size (model-dependent')
  cmd:option('-nSubvolumes', 1024, 'Number of subvolumes')
  cmd:option('-gpuDevice', 1, 'GPU device id (starting from 1)')
  cmd:option('-predType', 'maxclass', 'maxclass or maxsoftmax')
  cmd:option('-savePred', 0, 'Flag to save prediction')
  cmd:option('-seed', 123, 'seed')
  cmd:option('-nClasses', 3, 'Number of classes in labels')
  cmd:option('-nModal', 1, 'The number of modalities')
  cmd:option('-sampleType', 'gaussian', 'Distribution for sampling subvolumes. gaussian')
  cmd:text()
  opt = cmd:parse(arg or {})
end
print(opt)
---------------------------------------------------------------
torch.manualSeed(opt.seed)
cutorch.setDevice(opt.gpuDevice)
local foldList = utils.lines_from(opt.foldList)
---------------------------------------------------------------
local gather_function = {}
if opt.predType == 'maxclass' then
  gather_function = utils.gather_maxclass
elseif opt.predType == 'maxsoftmax' then
  gather_function = utils.gather_maxsoftmax
else
  print('Invalid prediction type. Should be maxclass or maxsoftmax.')
end
---------------------------------------------------------------
subsizes = {opt.zLen, opt.yLen, opt.xLen}
extend = {{0, 0}, {0, 0}, {0, 0}}
sizes = {opt.nModal, opt.MRI_zLen + extend[1][1] + extend[1][2], opt.MRI_yLen + extend[2][1] + extend[2][2], opt.MRI_xLen + extend[3][1] + extend[3][2]}
mean = {opt.MRI_zLen/2,  opt.MRI_yLen/2,  opt.MRI_xLen/2}
std = {opt.MRI_zLen/6, opt.MRI_yLen/6, opt.MRI_xLen/6}
---------------------------------------------------------------
local softmax = cudnn.VolumetricLogSoftMax():cuda()
local model = utils.load_prediction_model(opt.modelFile)
local brains = utils.load_brains(foldList, opt.nModal, opt.MRI_zLen, opt.MRI_yLen, opt.MRI_xLen, extend)
opt.nSubvolumes = opt.nSubvolumes - opt.nSubvolumes % opt.batchSize
brain_metrics = {}
print('Processing brains')
for b = 1, #foldList do
  brain_metrics[b] = {}
  brain = brains[b]
  print('Loading ' .. b .. 'th brain ' .. foldList[b])
  print('Creating coordinate grid')
  coords_grid = coords_grid or utils.create_dataset_coords(
    sizes, opt.nSubvolumes, subsizes, extend, opt.sampleType, mean, std)     
  print('Prediction')
  local outputCube = {}
  if opt.predType == 'maxclass' then
    outputCube = torch.IntTensor(opt.nClasses, 
      sizes[2] + extend[1][1] + extend[1][2], 
      sizes[3] + extend[2][1] + extend[2][2], 
      sizes[4] + extend[3][1] + extend[3][2]):fill(0)
  elseif opt.predType == 'maxsoftmax' then
    outputCube = torch.DoubleTensor(opt.nClasses,
      sizes[2] + extend[1][1] + extend[1][2],
      sizes[2] + extend[2][1] + extend[2][2],
      sizes[4] + extend[3][1] + extend[3][2]):fill(0)
  else
    print('Invalid prediction type. Should be maxclass or maxsoftmax.')
  end
  local batchSize = opt.batchSize
  local time = sys.clock()
  local k = 0
  for i = 1, #coords_grid, opt.batchSize do
    local inputs, targets = utils.create_cuda_mini_batch(
      {brain}, coords_grid, opt.batchSize, 
      subsizes, opt.nModal, opt.nSubvolumes, i, 'valid')
    local outputs = model:forward(inputs)
    k = k + 1
    outputs = softmax:forward(outputs)
    outputCube = gather_function(outputs, outputCube, coords_grid, i)
  end
  local maxs, outputCube = torch.max(outputCube, 1)
  time = sys.clock() - time
  print (time, 'seconds')
  brain = utils.reduceData(brain, extend)
  outputCube = utils.reduceOutput(outputCube, extend)
  if opt.savePred == 1 then
    npy4th.savenpy(foldList[b](1, #foldList[b] - 3) .. opt.filenameAdd .. 'prediction.npy', outputCube[1] - 1)
  end
  print('Calculate metrics')
  local splitted_output = utils.split_classes(outputCube[1], opt.nClasses)  
  local splitted_target = utils.split_classes(brain.target, opt.nClasses)

  brain_metrics[b].f1_score = {}
  brain_metrics[b].avd = {}
  brain_metrics[b].time = time
  for c = 1, opt.nClasses do
    brain_metrics[b].f1_score[c] = 
      utils.f1_score(splitted_output[c], splitted_target[c])
    brain_metrics[b].avd[c] = 
      utils.average_volumetric_difference(splitted_output[c], splitted_target[c])
  end
  collectgarbage()
end

--print(model_metrics)

local model_csv = {}
model_csv['brain'] = {}
model_csv['time'] = {}
first_run = true
for c = 1, opt.nClasses do
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
df = Dataframe()
df:load_table{data=Df_Dict(model_csv)}
df:to_csv(opt.filenameAdd .. opt.outputFile)
