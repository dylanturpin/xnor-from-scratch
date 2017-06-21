require 'nn'
require 'mattorch'
require 'image'
require 'optim'

debugger = require 'fb.debugger'

local utils = require 'utils'

nClasses = 250
nChannels = 1
batchSize = 100
sampleSize = {1,225,225}

local net, criterion = require 'models/sketchanet' ()
local dataloader = require 'dataloader'
dataloader:loadDataIntoMemory()

local top1Sum = 0
for i = 1,10 do
   -- set dropouts to training mode
   net:training()

   local inputs, labels = dataloader:randomlySampleTrainingData(batchSize);

   net:zeroGradParameters()

   outputs = net:forward(inputs)
   err = criterion:forward(outputs, labels)

   local gradOutputs = criterion:backward(outputs, labels)
   net:backward(inputs, gradOutputs)

   local feval = function()
      return err, gradParameters
   end

   parameters, gradParameters = net:getParameters()

   optim.sgd(feval, parameters, optimState);

   local pred = outputs

   local top1, top5 = util.computeScore(pred, labels)
   top1Sum = top1Sum + top1

   print(('%f\n'):format(top1Sum/i))
end


