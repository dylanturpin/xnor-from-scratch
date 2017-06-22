local D = {}

function D:loadDataIntoMemory()
   self.dataLoadedFromMatFile = mattorch.load('/u/dylanturpin/data/sketchANet/lua_dataset_without_order_info_256.mat')
end

local function trainHook(input)
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

local function testHook(input)
   local oH = sampleSize[2]
   local oW = sampleSize[3]
   local iW = input:size(3)
   local iH = input:size(2)
   local w1 = math.ceil((iW-oW)/2)
   local h1 = math.ceil((iH-oH)/2)
   local out = image.crop(input, w1, h1, w1+oW, h1+oH) -- center patch
   return out
end

function D:randomlySampleTrainingData(quantity)
   local data = torch.Tensor(quantity, sampleSize[1], sampleSize[2], sampleSize[3])
   local scalarLabels = torch.LongTensor(quantity)
   
   for i = 1,quantity do
      local randomIndex = torch.random(1, self.dataLoadedFromMatFile.trainImages:size(1))

      local out = self.dataLoadedFromMatFile.trainImages[{randomIndex,{},{},{}}]:float()
      out = trainHook(out)

      data[i] = out
      scalarLabels[i] = self.dataLoadedFromMatFile.trainLabels[randomIndex]
   end

   
   data = data:cuda()
   scalarLabels = scalarLabels:cuda()
   
   return data, scalarLabels

end

function D:getNumberOfValidationExamples()
   return self.dataLoadedFromMatFile.validationLabels:numel()
end

function D:getValidationDataByIndex(i1, i2)
   local indices = torch.range(i1, i2);
   local quantity = i2 - i1 + 1;
   assert(quantity > 0)
   -- now that indices has been initialized, get the samples
   local data = torch.Tensor(quantity, sampleSize[1], sampleSize[2], sampleSize[3])
   local scalarLabels = torch.LongTensor(quantity)
   for i=1,quantity do
      -- load the sample
      local out = self.dataLoadedFromMatFile.validationImages[{indices[i],{},{},{}}]:float()
      out = testHook(out)

      data[i] = out
      scalarLabels[i] = self.dataLoadedFromMatFile.validationLabels[indices[i]]
   end

   data = data:cuda()
   scalarLabels = scalarLabels:cuda()

   return data, scalarLabels
end


return D
