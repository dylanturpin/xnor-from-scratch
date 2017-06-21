local net = nn.Sequential();
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

-- When this module is required it can only return a single value.
-- The workaround is to return a function to be called like below.
-- > net, criterion = require 'models/sketchanet' ()
return function() return net, criterion end
