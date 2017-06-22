require 'cutorch'
require 'nn'
require 'cudnn'
require 'mattorch'
require 'image'
require 'optim'
require 'cunn'

debugger = require 'fb.debugger'
local utils = require 'utils'
local logger = optim.Logger('accuracy.log')
logger:setNames{'train top1', 'train top5', 'validation top1', 'validation top5'}


local optimState = {
    learningRate = 0.001
}
nClasses = 250
nChannels = 1
nBatchesPerEpoch = 100 
batchSize = 135 
nEpochs = 500
sampleSize = {1,225,225}

local net, criterion = require 'models/sketchanet' ()
parameters, gradParameters = net:getParameters()

local dataloader = require 'dataloader'
dataloader:loadDataIntoMemory()


for epochCounter = 1,nEpochs do
   print(('=========EPOCH #%d========='):format(epochCounter))

   print('=========Doing epoch on training data=========')
   local train_top1Sum = 0
   local train_top5Sum = 0
   for batchCounter = 1,nBatchesPerEpoch do
      cutorch.synchronize()

      -- set dropouts to training mode
      net:training()

      local batchInputs, batchLabels = dataloader:randomlySampleTrainingData(batchSize);

      local outputs, loss
      local feval = function()
	 net:zeroGradParameters()
	 outputs = net:forward(batchInputs)
	 loss = criterion:forward(outputs, batchLabels)
	 local dloss_doutputs = criterion:backward(outputs, batchLabels)
	 -- net:backwards implicitly calculates gradParameters
	 net:backward(batchInputs, dloss_doutputs)

	 return loss, gradParameters
      end

      optim.sgd(feval, parameters, optimState);

      local top1, top5 = utils.computeScore(outputs, batchLabels)
      train_top1Sum = train_top1Sum + top1
      train_top5Sum = train_top5Sum + top5

      print(('Training Epoch[%d/%d] Batch[%d/%d] Top1[%.2f (%.2f)] Top5[%.2f (%.2f)] Loss[%.5f]'):format(epochCounter, nEpochs, batchCounter, nBatchesPerEpoch, top1, train_top1Sum/batchCounter, top5, train_top5Sum/batchCounter, loss))
   end

   print('=========Doing epoch on validation data=========')
   local validation_top1Sum = 0
   local validation_top5Sum = 0
   local nValidationBatches = dataloader:getNumberOfValidationExamples()/batchSize
   for batchCounter = 1,nValidationBatches do
      cutorch.synchronize()

      -- set dropouts to testing mode
      net:evaluate()

      local startIndex = (batchCounter - 1) * batchSize + 1;
      local endIndex = startIndex + batchSize - 1;
      local batchInputs, batchLabels = dataloader:getValidationDataByIndex(startIndex, endIndex)

      local outputs = net:forward(batchInputs)
      local loss = criterion:forward(outputs, batchLabels)

      local top1, top5 = utils.computeScore(outputs, batchLabels)
      validation_top1Sum = validation_top1Sum + top1
      validation_top5Sum = validation_top5Sum + top5

      print(('Testing Epoch[%d/%d] Batch[%d/%d] Top1[%.2f (%.2f)] Top5[%.2f (%.2f)] Loss[%.5f]'):format(epochCounter, nEpochs, batchCounter, nValidationBatches, top1, validation_top1Sum/batchCounter, top5, validation_top5Sum/batchCounter, loss))
   end

   logger:add{train_top1Sum, train_top5Sum, validation_top1Sum, validation_top5Sum}
end
