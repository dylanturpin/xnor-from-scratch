require 'nn'
require 'mattorch'
require 'image'
require 'optim'

debugger = require('fb.debugger')

loaded = mattorch.load('/home/dylanturpin/data/sketchANet/lua_dataset_without_order_info_256.mat')

nClasses = 250
nChannels = 1
batchSize = 100

local net = nn.Sequential()
net:add(nn.SpatialConvolution(nChannels,64,15,15,3,3,0,0))
net:add(nn.ReLU(true))
net:add(nn.SpatialMaxPooling(3,3,2,2))
net:add(nn.SpatialConvolution(64,128,5,5,1,1,0,0))      
net:add(nn.ReLU(true))
net:add(nn.SpatialMaxPooling(3,3,2,2))
net:add(nn.SpatialConvolution(128,256,3,3,1,1,1,1))      
net:add(nn.ReLU(true))
net:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))      
net:add(nn.ReLU(true))
net:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))      
net:add(nn.ReLU(true))
net:add(nn.SpatialMaxPooling(3,3,2,2))
net:add(nn.SpatialConvolution(256,512,7,7,1,1,0,0))      
net:add(nn.ReLU(true))
net:add(nn.Dropout())
net:add(nn.SpatialConvolution(512,512,1,1,1,1,0,0))      
net:add(nn.ReLU(true))
net:add(nn.Dropout())
net:add(nn.SpatialConvolution(512,nClasses,1,1,1,1,0,0))
net:add(nn.View(nClasses))
net:add(nn.LogSoftMax())

local criterion = nn.ClassNLLCriterion()

function computeScore(output, target)
   -- Computes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))

   local top1 = correct:narrow(2, 1, 1):sum() / batchSize
   local top5 = correct:narrow(2, 1, 5):sum() / batchSize

   return top1 * 100, top5 * 100
end

local trainHook = function(input)
   local iW = input:size(3)
   local iH = input:size(2)

   -- do random crop
   local oW = sampleSize[3]
   local oH = sampleSize[2]
   local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
   local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
   local out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
   assert(out:size(3) == oW)
   assert(out:size(2) == oH)


   -- roll> 0.45 then rotate between -5 and 5 degrees...
   if torch.uniform() > 0.45 then
      degrees = torch.random(-5,5)
      out = image.rotate(out, math.rad(degrees), 'bilinear')
   end

   -- do hflip with probability 0.5
   if torch.uniform() > 0.5 then out = image.hflip(out) end

   return out
end

function sample(quantity)
   local data = torch.Tensor(quantity, sampleSize[1], sampleSize[2], sampleSize[3])
   local scalarLabels = torch.LongTensor(quantity)
   
   for i = 1,quantity do
      local randomIndex = torch.random(1, loaded.trainImages:size(1))

      local out = loaded.trainImages[{randomIndex,{},{},{}}]:float()
      out = trainHook(out)

      data[i] = out
      scalarLabels[i] = loaded.trainLabels[randomIndex]
   end

   
   return data, scalarLabels
end


sampleSize = {1,225,225}

local top1Sum = 0

for i = 1,10 do
   -- set dropouts to training mode
   net:training()

   local inputs, labels = sample(batchSize);

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

   local top1, top5 = computeScore(pred, labels)
   top1Sum = top1Sum + top1

   print(('%f\n'):format(top1Sum/i))
end


