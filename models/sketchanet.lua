local net = nn.Sequential();
net:add(cudnn.SpatialConvolution(nChannels,64,15,15,3,3,0,0))
-- inplace is set to true for all relus to save memory
net:add(cudnn.ReLU(true))
net:add(nn.SpatialMaxPooling(3,3,2,2))
net:add(cudnn.SpatialConvolution(64,128,5,5,1,1,0,0))      
net:add(cudnn.ReLU(true))
net:add(nn.SpatialMaxPooling(3,3,2,2))
net:add(cudnn.SpatialConvolution(128,256,3,3,1,1,1,1))      
net:add(cudnn.ReLU(true))
net:add(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1))      
net:add(cudnn.ReLU(true))
net:add(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1))      
net:add(cudnn.ReLU(true))
net:add(nn.SpatialMaxPooling(3,3,2,2))
net:add(cudnn.SpatialConvolution(256,512,7,7,1,1,0,0))      
net:add(cudnn.ReLU(true))
net:add(nn.Dropout())
net:add(cudnn.SpatialConvolution(512,512,1,1,1,1,0,0))      
net:add(cudnn.ReLU(true))
net:add(nn.Dropout())
net:add(cudnn.SpatialConvolution(512,nClasses,1,1,1,1,0,0))
net:add(nn.View(nClasses))
net:add(nn.LogSoftMax())

local criterion = nn.ClassNLLCriterion()

net = net:cuda()
criterion = criterion:cuda()

cudnn.fastest = true
cudnn.benchmark = true

-- When this module is required it can only return a single value.
-- The workaround is to return a function to be called like below.
-- > net, criterion = require 'models/sketchanet' ()
return function() return net, criterion end
