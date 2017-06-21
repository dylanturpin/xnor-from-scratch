local D = {}

function D:loadDataIntoMemory()
   self.dataLoadedFromMatFile = mattorch.load('/home/dylanturpin/data/sketchANet/lua_dataset_without_order_info_256.mat')
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

   
   return data, scalarLabels

end


return D
